#!/usr/bin/env bash
# Full training run, orchestrated inside the trainer container.
#
# Runs the whole pipeline for one wake word in a single container: assemble the
# pooled real-samples tree, generate the YAML config from the chosen
# hyperparameters, then `livekit-wakeword setup` + `run`. Progress and the final
# outcome are written to output/<slug>/train_status.json and the full log to
# output/<slug>/train.log, which the sync-server serves back to the app.
#
# Every input arrives as an environment variable so the sync-server can launch
# this with `docker run -e KEY=VALUE ...` (argv, never shell-interpolated) and
# stay injection-safe. Required: SLUG, PHRASE. The rest have defaults matching
# scripts/generate_config.py.
set -uo pipefail

SLUG="${SLUG:?SLUG required}"
PHRASE="${PHRASE:?PHRASE required}"
STEPS="${STEPS:-50000}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
TARGET_FP_PER_HOUR="${TARGET_FP_PER_HOUR:-0.2}"
PERSONAL="${PERSONAL:-0}"
POSITIVE_BOOST="${POSITIVE_BOOST:-1}"
N_SAMPLES="${N_SAMPLES:-}"
N_SAMPLES_VAL="${N_SAMPLES_VAL:-}"
POSITIVE_PER_BATCH="${POSITIVE_PER_BATCH:-}"
REAL_SAMPLES_DIR="${REAL_SAMPLES_DIR:-./data/train}"

cd /work

OUT_DIR="output/$SLUG"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/train.log"
STATUS="$OUT_DIR/train_status.json"

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
STARTED="$(now)"

# Escape a value for embedding inside a JSON string (backslash and quote only;
# the fields here never carry control characters).
json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

# write_status STATE STEP EXIT_CODE MESSAGE
write_status() {
  local esc_msg esc_phrase
  esc_msg="$(json_escape "$4")"
  esc_phrase="$(json_escape "$PHRASE")"
  cat > "$STATUS" <<EOF
{"slug":"$SLUG","phrase":"$esc_phrase","state":"$1","step":"$2","exit_code":$3,"message":"$esc_msg","steps":$STEPS,"model_size":"$MODEL_SIZE","personal":$([ "$PERSONAL" = "1" ] && echo true || echo false),"started_at":"$STARTED","updated_at":"$(now)"}
EOF
}

fail() {
  write_status "failed" "$1" "$2" "$3"
  echo "== FAILED at step '$1' (exit $2): $3 ==" | tee -a "$LOG"
  exit "$2"
}

: > "$LOG"
echo "== train_job slug=$SLUG phrase='$PHRASE' steps=$STEPS size=$MODEL_SIZE personal=$PERSONAL boost=$POSITIVE_BOOST started $STARTED ==" | tee -a "$LOG"

# 1. Assemble the pooled real-samples tree.
write_status "running" "assemble" 0 "assembling pooled data"
assemble_args=(--slug "$SLUG" --positive-boost "$POSITIVE_BOOST")
python3 scripts/assemble_training_data.py "${assemble_args[@]}" >>"$LOG" 2>&1 \
  || fail assemble $? "assemble_training_data failed"

# 2. Generate the YAML config from the chosen hyperparameters.
write_status "running" "generate" 0 "generating config"
CONFIG="trainer/configs/$SLUG.yaml"
gen_args=(--slug "$SLUG" --phrase "$PHRASE" --out "$CONFIG"
  --steps "$STEPS" --model-size "$MODEL_SIZE" --target-fp-per-hour "$TARGET_FP_PER_HOUR"
  --real-samples-dir "$REAL_SAMPLES_DIR")
[ "$PERSONAL" = "1" ] && gen_args+=(--personal)
[ -n "$N_SAMPLES" ] && gen_args+=(--n-samples "$N_SAMPLES")
[ -n "$N_SAMPLES_VAL" ] && gen_args+=(--n-samples-val "$N_SAMPLES_VAL")
[ -n "$POSITIVE_PER_BATCH" ] && gen_args+=(--positive-per-batch "$POSITIVE_PER_BATCH")
python3 scripts/generate_config.py "${gen_args[@]}" >>"$LOG" 2>&1 \
  || fail generate $? "generate_config failed"

# 3. Setup: synthesize positives/negatives and fetch corpora.
write_status "running" "setup" 0 "livekit-wakeword setup"
livekit-wakeword setup -c "$CONFIG" >>"$LOG" 2>&1 || fail setup $? "setup failed"

# 4. Train.
write_status "running" "train" 0 "training ($STEPS steps)"
livekit-wakeword run "$CONFIG" >>"$LOG" 2>&1 || fail train $? "training failed"

write_status "succeeded" "done" 0 "model written to $OUT_DIR"
echo "== done: model in $OUT_DIR ==" | tee -a "$LOG"
