# Wake-Word Trainer

This directory contains the training-only LiveKit wake-word pipeline copied from
`/data/claude_spawner_app/wakeword` and adapted for `/data/livekit_trainer`.

This project is the source of truth for produced wake-word model files. Runtime
consumers in other projects should consume exported artifacts from this
repository's `output/` tree.

## What's Here

- `Dockerfile.trainer` builds the CUDA training image with
  `livekit-wakeword[train,eval,export]`.
- `configs/` contains seed LiveKit YAML configs copied from the previous setup.
- `patches/0001-parallel-augmentation.patch` parallelizes the CPU-bound
  augmentation stage.
- `scripts/train.sh` builds the image if needed and runs setup plus training for
  one config.

The runtime sidecar from the old project was intentionally not copied. This
directory is for producing model artifacts, not serving them.

## Build

From the repo root:

```bash
docker build -f trainer/Dockerfile.trainer -t livekit-wakeword-trainer:latest trainer
```

## Train

From the repo root:

```bash
trainer/scripts/train.sh trainer/configs/bump.yaml
```

The script mounts the repo at `/work` in the container. Config paths should be
relative to the repo root. Model outputs land under:

```text
output/<model_name>/
```

Expected important artifacts:

- `<model_name>.onnx`
- `<model_name>.pt`
- `<model_name>_metrics.json`
- `<model_name>_eval.json`
- `<model_name>_det.png`

## Data

The seed configs use:

```yaml
data_dir: ./data
output_dir: ./output
```

Keep generated corpora, real voice recordings, and model outputs out of git.

Real Android-collected samples should eventually be imported into:

```text
data/real/<wake_word_slug>/
  positive/
  negative/
  background/
```

Richer labels such as `hard_negative`, `false_positive`, and `false_negative`
should be preserved in metadata and mapped into the categories required by the
current trainer.

## Calibration

`configs/bump_cal.yaml` is a short calibration config using the real model
shape but fewer steps. Use this before expensive full training runs.

## Smoke Test

For a tiny end-to-end pipeline check:

```bash
trainer/scripts/smoke_train.sh
```

This uses `configs/smoke.yaml`. It is meant to validate the container and CLI
path, not to produce a useful wake-word model.
