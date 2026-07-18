# Making a Personal, Single-Voice Model

The goal of this project is a wake word tuned to *one* user's voice. That runs
against how the upstream trainer works, so this doc records how the two are
reconciled and the knobs that do it.

## How the trainer treats your real clips

From reading the upstream trainer (`livekit/livekit-wakeword`):

- Your recorded clips are copied into the same positive/negative/background
  split directories as the trainer's synthetic clips, renumbered so downstream
  stages **cannot tell them apart**. There is no real-vs-synthetic weighting.
- The trainer generates a **large synthetic positive pool** — `n_samples`
  defaults to 10,000–25,000 positives, produced by Piper TTS.
- Each clip (real or synthetic) is augmented only `augmentation.rounds` times —
  **1 by default, 3 in the prod config**. A real clip does *not* explode into
  thousands of variants.
- The batch sampler walks the whole positive array with wraparound, taking a
  fixed number of positives per batch. So a clip's influence is proportional to
  **its share of the total positive pool**.

The consequence: 300 real positives against a 20,000 synthetic pool is ~1.5% of
positives — effectively invisible. Collecting denser helps, but the pool ratio
is the real lever.

## Two different "overweight positives" knobs — don't confuse them

- **positive vs. negative balance** — `batch_n_per_class.positive`
  (`generate_config.py --positive-per-batch`). Negatives are pooled across every
  wake word and grow fast; raising the per-batch positive count keeps positive
  gradient signal strong. Duplicating positive *files* does **not** change this
  balance, because the batch still draws a fixed number of positives.
- **real vs. synthetic, within positives** — controlled by the size of the
  synthetic pool (`n_samples`) and how many real positive files exist.
  Duplicating your real positive files **does** change this: more real rows in
  the positive array means the sampler hits them more often relative to the
  synthetic ones. This is what personalizes the model.

## The levers this repo gives you

1. **Collect efficiently** — the app's **Dense** bulk-script style (Settings →
   Bulk script style) asks for short, prosodically varied repetitions of the
   wake phrase with frequent near misses, so you gather many true positives and
   hard negatives per minute instead of burying one positive in a paragraph.

2. **Replicate your positives** — `assemble_training_data.py --positive-boost N`
   places each of your own positives N times in the pooled tree, raising their
   weight against the synthetic pool.

3. **Shrink the synthetic pool** — `generate_config.py --personal` drops
   `n_samples` to 3,000 (from 20,000), sets `n_samples_val` to 600, and raises
   the per-batch positive count to 100.

Combined example: ~300 unique real positives, boosted ×8 (2,400 real rows)
against a 3,000-synthetic personal pool ≈ **45% of positives are your voice**,
versus ~1.5% at the defaults.

```bash
# after syncing recordings:
scripts/assemble_training_data.py --slug my_word --positive-boost 8
scripts/generate_config.py --slug my_word --phrase "my word" --personal
```

## Rough collection targets

There is no published minimum. As practical starting targets for a personal
model, aim for a few hundred real positives (≈300–500), 50–150 near-miss hard
negatives, and 15–30 minutes of real background from where you'll use it. Then
**train once and read the eval** — the mistakes drive the next batch (see
`docs/CORRECTION_BATCH_FORMAT.md`). Ordinary negatives are the one bucket you can
mostly leave to the trainer's downloaded corpora.
