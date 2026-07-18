# Correction Batch Format

A correction batch turns a trained model's **mistakes** back into training data.
It closes the collection loop: after a model is trained and evaluated (or run in
a live scorer), the clips it got wrong are re-imported with distinct labels so
the next model sees them as targeted reinforcement.

`scripts/import_corrections.py` is the emitter. It reads a batch, compares each
clip's score against the detection threshold, and files only the mistakes into
`data/real/<slug>/{positive,negative}/`, preserving the richer label in
`metadata.jsonl`.

## Layout

```text
<batch>/
  corrections.json
  audio/
    <clip>.wav
```

Audio should be the same shape as the rest of the dataset: WAV, mono, 16 kHz,
16-bit PCM.

## `corrections.json`

```json
{
  "schema_version": 1,
  "wake_word": { "slug": "beep_beep", "phrase": "beep beep" },
  "model": { "name": "beep_beep", "version": "2026-07-18", "threshold": 0.5 },
  "corrections": [
    {
      "id": "clip_0001",
      "file": "audio/clip_0001.wav",
      "label": "positive",
      "score": 0.21,
      "spoken_phrase": "beep beep",
      "source": "eval",
      "notes": ""
    },
    {
      "id": "clip_0002",
      "file": "audio/clip_0002.wav",
      "label": "background",
      "score": 0.88,
      "source": "runtime"
    }
  ]
}
```

Each correction carries the clip's **ground-truth** `label` and the model
`score` that clip received. The importer derives the mistake:

| Ground-truth label family | Expected | Mistake when | Emitted label | Trains as |
| --- | --- | --- | --- | --- |
| `positive`, `false_negative` | fires | `score < threshold` | `false_negative` | `positive` |
| `negative`, `hard_negative`, `background`, `false_positive` | silent | `score >= threshold` | `false_positive` | `negative` |

Clips the model classified correctly are skipped. A clip may instead carry an
explicit `mistake` field (`false_positive` or `false_negative`) for a manually
triaged correction; that overrides score-based derivation.

The threshold comes from `model.threshold`, or `--threshold` on the command
line (which wins). Re-run with `--skip-existing` to make imports idempotent.

## Where the scores come from

This format is deliberately independent of *what* produced the scores. Today
that is expected to be a batch eval over held-out labeled clips or a runtime
scorer's log of live detections; either can write a `corrections.json`. When a
runtime scorer service lands under `runtime/`, it should emit this shape.
