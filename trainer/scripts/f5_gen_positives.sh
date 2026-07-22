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

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
REFS_HOST="$REPO/data/real/$SLUG/positive"
OUT_HOST="$REPO/data/synth_f5/$SLUG/positive"
[ -d "$REFS_HOST" ] || { echo "no reference clips at $REFS_HOST" >&2; exit 1; }
mkdir -p "$OUT_HOST"

# Use a small, clean subset of real positives as references (rotated inside the
# python). More than ~8 adds little timbre variety and slows staging.
STAMP="$(date +%s)"
CREFS="/tmp/f5gen_$STAMP/refs"
COUT="/tmp/f5gen_$STAMP/out"
docker exec "$CONTAINER" mkdir -p "$CREFS" "$COUT"
n=0
for f in "$REFS_HOST"/*.wav; do
  docker cp "$f" "$CONTAINER:$CREFS/$(basename "$f")"
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
  --count "$COUNT"

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
