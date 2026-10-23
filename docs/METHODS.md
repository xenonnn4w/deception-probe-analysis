# Methods and reproducibility

## Historical experiments

- Qwen3 models: 0.6B, 1.7B, 4B, 8B, and 14B. No 32B result is included.
- Compute: one NVIDIA A100 PCIe GPU with 40 GB VRAM, as recorded in the original execution logs.
- Linear logistic-regression probes were trained using supplied RepE honesty data and evaluated across layers with the supplied pipeline.
- Evaluation tasks: instructed deception, convincing game, insider trading, harm-pressure choice, and harm-pressure knowledge report.
- Each model/task aggregate contains 100 labeled examples. Counts of honest/deceptive examples appear in the tables.
- AUROC treats deceptive labels as the positive class. It measures discrimination relative to those labels, not verified strategic intent.
- `scaling_summary.csv` reports the layer closest to 65% depth and the best observed evaluation layer. The summary describes 95% intervals from 1,000 stratified bootstrap resamples, seed 0.
- `ensemble_comparison.csv` compares a fixed-depth probe, an unweighted all-layer mean, and LR stacking for 0.6B and 14B. Its header states the stacking weights and standardization used RepE training scores rather than evaluation labels. This packaging pass has not independently audited the complete training implementation.

## Aggregate analysis implemented here

The script validates unique model/task/layer keys, AUROC bounds, label counts, input hashes, and consistency of the supplied summary with the layer table. It computes per-model arithmetic mean AUROC across the five tasks and per-task post-hoc selection gaps. It does not treat correlated layer rows as independent samples.

The README displays the three original experiment PNGs. Their hashes are recorded in `data/manifest.json`. Optional aggregate redraws are written separately to `figures/generated/` and do not replace the originals.

The scaling plots use the saved confidence intervals. They do not recompute intervals or account for best-layer selection. Best-layer intervals condition on the selected layer and are optimistic; they are not corrected estimates of out-of-sample layer selection. The original layer-geometry figure compares centroid-difference directions, not fitted logistic-regression weights. Its underlying arrays are unavailable in this package.

## What cannot be rerun from this folder

The original model inference, probe fitting, bootstrap resampling, ensemble fitting, and on-policy generation require missing inputs or code. The aggregate tables do not contain token scores, predictions, activations, prompts, quantization settings, complete environment metadata, or all random seeds. Avoid claiming exact end-to-end reproducibility.

To make a future full replication: use a licensed public implementation; record model/tokenizer/dataset revisions, precision, seeds, masks, splits and pooling; save per-example scores with permitted identifiers; reserve a validation split for layer selection and calibration; then rerun inference and analysis. No GPU experiments have been run during packaging.

## On-policy pilot

The historical pilot generated 20 responses with Qwen3-0.6B, scored them using a RepE-trained probe at layer 18, and reused expected dataset labels. Its own manual audit found examples inconsistent with those labels. Consequently, its AUROC and apparent gap from the off-policy baseline are not used as validated performance findings. The full generated text is not redistributed.

## References and attribution

- [Qwen3 model family](https://huggingface.co/Qwen).
- [Liars’ Bench dataset](https://huggingface.co/datasets/Cadenza-Labs/liars-bench). The original exercise specified revision `65299c5b10aa07adf75716ecb875c6713eed0dde`; the aggregate tables alone cannot independently verify the downloaded revision.
- RepE honesty prompt construction and the supplied `deception_detection` framework used for the original experiments. The framework is not redistributed here.
- The supplied preprint *Linear Probe Accuracy Scales with Model Size and Benefits from Multi-Layer Ensembling* motivated the replication. Its PDF and assessment materials are not redistributed here.
