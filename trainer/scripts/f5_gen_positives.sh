#!/usr/bin/env bash
# Voice-cloned wake-word positives via F5-TTS (speech_services stack).
#
# Stages clean reference clips of the user's voice into the running
# speech-f5tts container, generates a batch of cloned positives with the
# resident F5 model, then resamples them to 16 kHz mono 16-bit PCM into the
# trainer's real-samples layout.
#
# Usage:
#   trainer/scripts/f5_gen_positives.sh <slug> "<phrase>" [count]
#
# Example:
#   trainer/scripts/f5_gen_positives.sh all_set "all set" 300
#
# Reference clips are taken from data/real/<slug>/positive (real recordings of
# the user saying the phrase — their transcript is the phrase, which is exactly
# the ref_text F5 needs). Output lands in a DEDICATED synth bucket
# (data/synth_f5/<slug>/positive) so it can be weighted or removed independently
# and never silently masquerades as a real recording. Add it to training by
# pointing real_samples_dir at it, or copying it into data/train/<slug>/positive.
set -euo pipefail

SLUG="${1:?usage: f5_gen_positives.sh <slug> \"<phrase>\" [count]}"
PHRASE="${2:?phrase required}"
COUNT="${3:-200}"
CONTAINER="${F5_CONTAINER:-speech-f5tts}"
# Fidelity knobs (F5 defaults). Raise NFE_STEP for a sharper, more faithful
# render (slower); raise CFG_STRENGTH to hew closer to your timbre.
NFE_STEP="${F5_NFE_STEP:-32}"
CFG_STRENGTH="${F5_CFG_STRENGTH:-2.0}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_HOST="$REPO/data/synth_f5/$SLUG/positive"

# Prefer SHORT real positive clips as the cloning reference (ref_text = the
# phrase): F5 sizes its output from the reference's rate, so a long enrollment
# passage starves the short wake phrase and crushes it. Fall back to the
# enrollment read only if no positives exist yet.
POS_HOST="$REPO/data/real/$SLUG/positive"
ENROLL_HOST="$REPO/data/real/$SLUG/enrollment"
if ls "$POS_HOST"/*.wav >/dev/null 2>&1; then
  REFS_HOST="$POS_HOST"
  echo "using positive-clip reference(s) from $POS_HOST"
elif ls "$ENROLL_HOST"/*.wav >/dev/null 2>&1; then
  REFS_HOST="$ENROLL_HOST"
  echo "no positives; falling back to enrollment reference(s) from $ENROLL_HOST"
else
  REFS_HOST="$POS_HOST"
fi
[ -d "$REFS_HOST" ] || { echo "no reference clips at $REFS_HOST" >&2; exit 1; }
mkdir -p "$OUT_HOST"

# Use a small, clean subset as references (rotated inside the python). More than
# ~8 adds little timbre variety and slows staging. Stage any sidecar .txt too so
# an enrollment reference keeps its exact ref_text.
STAMP="$(date +%s)"
CREFS="/tmp/f5gen_$STAMP/refs"
COUT="/tmp/f5gen_$STAMP/out"
docker exec "$CONTAINER" mkdir -p "$CREFS" "$COUT"
n=0
for f in "$REFS_HOST"/*.wav; do
  docker cp "$f" "$CONTAINER:$CREFS/$(basename "$f")"
  sidecar="${f%.wav}.txt"
  [ -f "$sidecar" ] && docker cp "$sidecar" "$CONTAINER:$CREFS/$(basename "$sidecar")"
  n=$((n+1)); [ "$n" -ge 8 ] && break
done
echo "staged $n reference clips into $CONTAINER"

# Stage and run the resident-model generator.
docker cp "$REPO/trainer/scripts/f5_gen_positives.py" "$CONTAINER:/tmp/f5gen_$STAMP/gen.py"
docker exec "$CONTAINER" python3 "/tmp/f5gen_$STAMP/gen.py" \
  --refs-dir "$CREFS" \
  --ref-text "$PHRASE" \
  --gen-text "$PHRASE" \
  --out-dir "$COUT" \
  --count "$COUNT" \
  --nfe-step "$NFE_STEP" \
  --cfg-strength "$CFG_STRENGTH"

# Pull the 24 kHz output back and resample to the trainer's 16 kHz mono 16-bit.
TMP_PULL="/tmp/f5gen_pull_$STAMP"; mkdir -p "$TMP_PULL"
docker cp "$CONTAINER:$COUT/." "$TMP_PULL/"
for f in "$TMP_PULL"/*.wav; do
  base="$(basename "$f")"
  ffmpeg -v error -y -i "$f" -ar 16000 -ac 1 -sample_fmt s16 "$OUT_HOST/$base"
done
count_out=$(ls "$OUT_HOST"/*.wav 2>/dev/null | wc -l)
echo "wrote $count_out cloned positives to $OUT_HOST"

# Cleanup container scratch.
docker exec "$CONTAINER" rm -rf "/tmp/f5gen_$STAMP" || true
rm -rf "$TMP_PULL"
