# Findings and limitations

## Scaling partially replicates

The saved results cover five model sizes and five tasks. Explicitly instructed deception and convincing-game discrimination generally improve through 8B at the fixed-depth layer, but the trend is not uniformly monotonic. Other tasks show weaker or unstable transfer.

At 14B, harm-pressure knowledge-report AUROC is approximately 0.389 near 65% depth, whereas its best observed evaluation layer reaches approximately 0.868. This illustrates layer sensitivity, not a validated model-selection improvement: the best layer was selected using the evaluation set itself.

## Layer fusion does not universally improve transfer

| Model | Fixed-depth layer | All-layer mean | LR stacking |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 0.694 | 0.628 | 0.546 |
| Qwen3-14B | 0.768 | 0.795 | 0.750 |

Numbers are unweighted means of AUROC over the five tasks. They are not pooled AUROC or accuracy percentages. The 14B averaging improvement masks regressions on some tasks. There is no demonstrated universal advantage from stacking weights fitted on the training distribution.

## Labels are a prerequisite for the on-policy comparison

The historical 20-response pilot reused expected labels after generating new completions. Manual inspection identified behavior/label disagreements. The pilot is evidence of a methodological issue, not a reliable estimate of a performance change. A valid comparison needs independent behavioral labels, ambiguity handling, larger samples, and comparable prompt/task conditions.

## Interpretations versus demonstrated mechanisms

The honesty/dishonesty instruction contrast in training could encourage probes to learn persona, framing, or stylistic features rather than strategic deception. Differences in training and evaluation token masks and pooling may also affect transfer. These are plausible explanations from code inspection and task sensitivity; the included aggregates do not establish those mechanisms causally.

## Follow-up work — proposed, not completed

1. Evaluate content-matched true and false responses with explicit persona cues removed.
2. Harmonize training/evaluation token masks and pooling.
3. Select layers and calibrate fusion using independent validation data.
4. Independently grade new model generations before on-policy scoring.
5. Test transfer across scenarios and model families, with uncertainty estimates and failure analysis.

## Scope of conclusions

These experiments do not establish reliable detection of organic scheming, hidden goals, or deployment-time deception. Small benchmark subsets, task overlap, label validity, and post-hoc selection limit interpretation. No new model evaluations were performed when preparing this repository.
