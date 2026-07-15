# Model Artifact Policy

This repository is the source of truth for wake-word model artifacts.

## Output Layout

Trainer output should live under:

```text
output/<wake_word_slug>/
  <wake_word_slug>.onnx
  <wake_word_slug>.pt
  <wake_word_slug>_metrics.json
  <wake_word_slug>_eval.json
  <wake_word_slug>_det.png
```

The `.onnx` file is the primary artifact for downstream projects.

## Downstream Consumption

Other projects should consume models from this repo's released or explicitly
shared artifacts. They should not train their own copies of the same wake word
unless the model is intentionally forked.

When a downstream project copies a model, record:

- Wake-word slug.
- Source commit from this repo.
- Model path.
- Evaluation metrics used to accept the model.
- Any threshold chosen by the downstream runtime.

## Git Policy

Generated corpora and routine training outputs stay out of Git by default.

Commit model artifacts only when the user explicitly decides that a model is
ready to publish through Git. Otherwise, keep outputs in ignored local storage
or attach them to GitHub releases later.

## Acceptance Notes

Before promoting a model, capture:

- Training config path.
- Real-data bundle or dataset summary.
- False positives per hour.
- Recall.
- Suggested threshold.
- Known failure cases.
