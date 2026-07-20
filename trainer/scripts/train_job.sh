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
# start | end. Decides the leading-context recipe: an end token (end of speech,
# e.g. "all set") trains with prior speech in front; a start token begins an
# utterance from a quiet room, so only ambient noise fills the lead. See
# scripts/generate_config.py --token-type.
TOKEN_TYPE="${TOKEN_TYPE:-end}"

cd /work

OUT_DIR="output/$SLUG"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/train.log"
STATUS="$OUT_DIR/train_status.json"

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
STARTED="$(now)"
# Compact, filesystem-safe run id derived from the start instant, e.g.
# 20260719T153000Z. Each run archives its artifacts under output/<slug>/runs/<RUNID>/
# so retraining never overwrites a prior model.
RUNID="$(printf '%s' "$STARTED" | tr -d ':-')"

# Escape a value for embedding inside a JSON string (backslash and quote only;
# the fields here never carry control characters).
json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

# write_status STATE STEP EXIT_CODE MESSAGE
write_status() {
  local esc_msg esc_phrase
  esc_msg="$(json_escape "$4")"
  esc_phrase="$(json_escape "$PHRASE")"
  cat > "$STATUS" <<EOF
{"slug":"$SLUG","phrase":"$esc_phrase","state":"$1","step":"$2","exit_code":$3,"message":"$esc_msg","steps":$STEPS,"model_size":"$MODEL_SIZE","personal":$([ "$PERSONAL" = "1" ] && echo true || echo false),"run_id":"$RUNID","started_at":"$STARTED","updated_at":"$(now)"}
EOF
}

fail() {
  write_status "failed" "$1" "$2" "$3"
  echo "== FAILED at step '$1' (exit $2): $3 ==" | tee -a "$LOG"
  exit "$2"
}

: > "$LOG"
echo "== train_job slug=$SLUG phrase='$PHRASE' token=$TOKEN_TYPE steps=$STEPS size=$MODEL_SIZE personal=$PERSONAL boost=$POSITIVE_BOOST started $STARTED ==" | tee -a "$LOG"

# 1. Assemble the pooled real-samples tree.
write_status "running" "assemble" 0 "assembling pooled data"
ASSEMBLE_SUMMARY="$OUT_DIR/assemble_summary.json"
assemble_args=(--slug "$SLUG" --positive-boost "$POSITIVE_BOOST"
  --summary-json "$ASSEMBLE_SUMMARY")
python3 scripts/assemble_training_data.py "${assemble_args[@]}" >>"$LOG" 2>&1 \
  || fail assemble $? "assemble_training_data failed"

# 2. Generate the YAML config from the chosen hyperparameters.
write_status "running" "generate" 0 "generating config"
CONFIG="trainer/configs/$SLUG.yaml"
CONFIG_PARAMS="$OUT_DIR/config_params.json"
gen_args=(--slug "$SLUG" --phrase "$PHRASE" --out "$CONFIG"
  --steps "$STEPS" --model-size "$MODEL_SIZE" --target-fp-per-hour "$TARGET_FP_PER_HOUR"
  --real-samples-dir "$REAL_SAMPLES_DIR" --token-type "$TOKEN_TYPE"
  --params-json "$CONFIG_PARAMS")
# Durable hard negatives: the app regenerates this config on every run, so the
# curated near-miss list lives in a sidecar file and is always fed back in
# rather than being silently overwritten to empty.
NEGATIVES_FILE="trainer/configs/$SLUG.negatives.txt"
[ -f "$NEGATIVES_FILE" ] && gen_args+=(--negatives-file "$NEGATIVES_FILE")
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

# 5. Archive this run: snapshot the model artifacts into a timestamped folder
# with checksums and a manifest, so the series of models for this wake word is
# retained and comparable. The live path (output/<slug>/<slug>.onnx) is left in
# place so the scorer, app, and status endpoints keep reading it unchanged; a
# CURRENT marker records which run it corresponds to.
write_status "running" "archive" 0 "archiving run $RUNID"
RUN_DIR="$OUT_DIR/runs/$RUNID"
mkdir -p "$RUN_DIR"

sha256() { [ -f "$1" ] && sha256sum "$1" | awk '{print $1}' || true; }

for f in "$SLUG.onnx" "$SLUG.pt" "${SLUG}_metrics.json" "${SLUG}_eval.json" "${SLUG}_det.png"; do
  [ -f "$OUT_DIR/$f" ] && cp -p "$OUT_DIR/$f" "$RUN_DIR/"
done
[ -f "$CONFIG" ] && cp -p "$CONFIG" "$RUN_DIR/config.yaml"
[ -f "$NEGATIVES_FILE" ] && cp -p "$NEGATIVES_FILE" "$RUN_DIR/negatives.txt"

ONNX_SHA="$(sha256 "$OUT_DIR/$SLUG.onnx")"
PT_SHA="$(sha256 "$OUT_DIR/$SLUG.pt")"
ONNX_BYTES="$([ -f "$OUT_DIR/$SLUG.onnx" ] && stat -c %s "$OUT_DIR/$SLUG.onnx" || echo 0)"
# The repo is bind-mounted from the host and owned by a different uid, so git
# refuses to operate on it ("dubious ownership") and the commit stamped "unknown".
# Marking it a safe directory lets the container read the exact code revision.
git config --global --add safe.directory /work 2>/dev/null || true
GIT_COMMIT="$(git -C /work rev-parse HEAD 2>/dev/null || echo unknown)"
FINISHED="$(now)"

# Assemble the complete provenance manifest: every knob the run turned, the real
# vs synthetic clip counts, the resolved hyperparameters, the eval metrics, the
# code revision, and the model checksums. Built in python from the sidecars so
# the record is a single source of truth the sync-server folds into the models
# table. Robust to any sidecar being absent (older/failed step).
MANIFEST_ENV_SLUG="$SLUG" MANIFEST_ENV_PHRASE="$PHRASE" \
  MANIFEST_ENV_RUNID="$RUNID" MANIFEST_ENV_ONNX_SHA="$ONNX_SHA" \
  MANIFEST_ENV_PT_SHA="$PT_SHA" MANIFEST_ENV_ONNX_BYTES="$ONNX_BYTES" \
  MANIFEST_ENV_STEPS="$STEPS" MANIFEST_ENV_MODEL_SIZE="$MODEL_SIZE" \
  MANIFEST_ENV_PERSONAL="$PERSONAL" MANIFEST_ENV_POSITIVE_BOOST="$POSITIVE_BOOST" \
  MANIFEST_ENV_TARGET_FP="$TARGET_FP_PER_HOUR" MANIFEST_ENV_TOKEN_TYPE="$TOKEN_TYPE" \
  MANIFEST_ENV_GIT_COMMIT="$GIT_COMMIT" MANIFEST_ENV_STARTED="$STARTED" \
  MANIFEST_ENV_FINISHED="$FINISHED" MANIFEST_ENV_IMAGE="${TRAINER_IMAGE:-}" \
  MANIFEST_ENV_PARAMS="$CONFIG_PARAMS" MANIFEST_ENV_SUMMARY="$ASSEMBLE_SUMMARY" \
  MANIFEST_ENV_EVAL="$OUT_DIR/${SLUG}_eval.json" \
  MANIFEST_ENV_METRICS="$OUT_DIR/${SLUG}_metrics.json" \
  MANIFEST_ENV_OUT="$RUN_DIR/manifest.json" \
  python3 scripts/write_manifest.py >>"$LOG" 2>&1 \
  || echo "== WARNING: write_manifest failed; model kept, provenance incomplete ==" | tee -a "$LOG"

printf '%s\n' "$RUNID" > "$OUT_DIR/CURRENT"
echo "== archived run $RUNID (onnx sha256 ${ONNX_SHA:-none}) ==" | tee -a "$LOG"

write_status "succeeded" "done" 0 "model written to $OUT_DIR (run $RUNID)"
echo "== done: model in $OUT_DIR ==" | tee -a "$LOG"
