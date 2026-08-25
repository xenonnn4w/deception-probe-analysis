#!/usr/bin/env python3
"""Validate historical aggregates and regenerate descriptive outputs; no inference."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(line for line in stream if not line.startswith("#")))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for item in manifest["files"]:
        actual = hashlib.sha256((ROOT / item["file"]).read_bytes()).hexdigest()
        require(actual == item["sha256"], f"Input hash mismatch: {item['file']}")

    layers = read_csv(ROOT / "data/aggregate/layer_metrics.csv")
    scaling = read_csv(ROOT / "data/aggregate/scaling_summary.csv")
    ensembles = read_csv(ROOT / "data/aggregate/ensemble_comparison.csv")
    index = {}
    for row in layers:
        key = row["model"], row["dataset"], int(row["layer"])
        require(key not in index, f"Duplicate layer measurement: {key}")
        require(0 <= float(row["auroc"]) <= 1, f"Invalid AUROC: {key}")
        require(int(row["n_honest"]) + int(row["n_deceptive"]) == 100,
                f"Unexpected sample count: {key}")
        index[key] = row

    seen = set()
    comparisons = []
    for row in scaling:
        key = row["model"], row["dataset"]
        require(key not in seen, f"Duplicate scaling summary: {key}")
        seen.add(key)
        fixed = float(row["fixed65_auroc"])
        best = float(row["best_auroc"])
        raw = index[(*key, int(row["fixed65_layer"]))]
        require(math.isclose(fixed, float(raw["auroc"]), abs_tol=0.0001),
                f"Fixed-depth summary mismatch: {key}")
        best_raw = index[(*key, int(row["best_layer"]))]
        require(math.isclose(best, float(best_raw["auroc"]), abs_tol=0.0001),
                f"Best-layer summary mismatch: {key}")
        observed_best = max(float(v["auroc"]) for k, v in index.items() if k[:2] == key)
        require(math.isclose(best, observed_best, abs_tol=0.0001),
                f"Best layer is not the observed maximum: {key}")
        for prefix in ("fixed65", "best"):
            lo, hi = float(row[f"{prefix}_ci_lo"]), float(row[f"{prefix}_ci_hi"])
            require(0 <= lo <= hi <= 1, f"Invalid interval: {key}")
        comparisons.append({"model": key[0], "task": key[1],
                            "fixed_depth_auroc": fixed, "posthoc_best_auroc": best,
                            "selection_gap": best - fixed})

    models = sorted({r["model"] for r in layers}, key=lambda s: float(s.split("-")[1][:-1]))
    tasks = sorted({r["dataset"] for r in layers})
    require(len(seen) == len(models) * len(tasks), "Incomplete model/task summary")
    grouped = defaultdict(list)
    ensemble_seen = set()
    for row in ensembles:
        key = row["model"], row["dataset"]
        require(key not in ensemble_seen, f"Duplicate ensemble row: {key}")
        ensemble_seen.add(key)
        require(key in seen, f"Unknown ensemble task: {key}")
        for field in ("auroc_fixed65", "auroc_meanlayers", "auroc_lr_ensemble"):
            require(0 <= float(row[field]) <= 1, f"Invalid ensemble AUROC: {key}")
        fixed_row = next(r for r in scaling if (r["model"], r["dataset"]) == key)
        require(math.isclose(float(row["auroc_fixed65"]), float(fixed_row["fixed65_auroc"]),
                             abs_tol=0.0001), f"Ensemble baseline mismatch: {key}")
        grouped[row["model"]].append(row)
    for model, rows in grouped.items():
        require({r["dataset"] for r in rows} == set(tasks), f"Incomplete ensemble tasks: {model}")

    output = {"models": models, "tasks": tasks, "layer_measurements": len(layers),
              "model_task_pairs": len(seen), "examples_per_model_task": 100,
              "ensemble_mean_task_auroc": {
                  model: {field: mean(float(r[field]) for r in rows)
                          for field in ("auroc_fixed65", "auroc_meanlayers", "auroc_lr_ensemble")}
                  for model, rows in grouped.items()},
              "limitations": ["Historical aggregates; no new inference",
                              "Best-layer selection uses evaluation data",
                              "On-policy pilot labels were not independently verified",
                              "Confidence intervals are imported, not recomputed"]}
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    with (reports / "scaling_by_task.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    return output, scaling


def plot(output, scaling):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for task in output["tasks"]:
        rows = sorted((r for r in scaling if r["dataset"] == task), key=lambda r: float(r["params_b"]))
        x = [float(r["params_b"]) for r in rows]
        y = [float(r["fixed65_auroc"]) for r in rows]
        lo = [float(r["fixed65_ci_lo"]) for r in rows]
        hi = [float(r["fixed65_ci_hi"]) for r in rows]
        label = task.replace("liars_bench__", "").replace("-", " ")
        line, = ax.plot(x, y, marker="o", label=label)
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Model parameters (billions; log scale)", ylabel="AUROC",
           title="Fixed-depth probe transfer varies by task", ylim=(0, 1.03), xscale="log")
    ax.legend(fontsize=8, loc="lower left")
    ax.text(0, -0.2, "Historical results; shaded bands: saved 95% bootstrap CIs. Layer nearest 65% depth.",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "scaling_fixed_depth.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    groups = output["ensemble_mean_task_auroc"]
    methods = [("auroc_fixed65", "Fixed depth"), ("auroc_meanlayers", "Layer mean"),
               ("auroc_lr_ensemble", "LR stacking")]
    names = sorted(groups, key=lambda s: float(s.split("-")[1][:-1]))
    for j, (key, label) in enumerate(methods):
        x = [i + (j - 1) * 0.24 for i in range(len(names))]
        ax.bar(x, [groups[n][key] for n in names], width=0.23, label=label)
    ax.set_xticks(range(len(names)), names)
    ax.set(ylabel="Mean task AUROC", ylim=(0, 1), title="Layer fusion does not consistently improve transfer")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0, -0.16, "Arithmetic mean across five tasks; not pooled AUROC. No ensemble CIs available.",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "ensemble_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", action="store_true", help="Also generate figures; requires matplotlib")
    args = parser.parse_args()
    result, scaling_rows = analyze()
    if args.plots:
        plot(result, scaling_rows)
    print(json.dumps(result, indent=2))
