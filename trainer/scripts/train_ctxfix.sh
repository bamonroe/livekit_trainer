#!/usr/bin/env bash
# Train a config with the leading-context augmentation fix mounted over the
# installed trainer module. Same as train.sh but bind-mounts
# trainer/patches/augment_ctxfix.py over livekit.wakeword.data.augment so the
# leading silence in front of positives is filled with real background/speech
# instead of zeros. See trainer/configs/all_set_ctx.yaml and
# streaming-recall-gap for why.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: trainer/scripts/train_ctxfix.sh <config-yaml-relative-to-repo-root>" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config="$1"
image="${LIVEKIT_WAKEWORD_TRAINER_IMAGE:-livekit-wakeword-trainer:latest}"
patch="$repo_root/trainer/patches/augment_ctxfix.py"
pkg_augment="/opt/conda/lib/python3.11/site-packages/livekit/wakeword/data/augment.py"
gpu_args=()

if [[ "${LIVEKIT_WAKEWORD_USE_GPU:-1}" != "0" ]]; then
  gpu_args=(--gpus all)
fi

if [[ ! -f "$patch" ]]; then
  echo "missing patch: $patch" >&2
  exit 1
fi

mkdir -p "$repo_root/data" "$repo_root/output"

docker run --rm "${gpu_args[@]}" \
  -v "$repo_root:/work" \
  -v "$patch:$pkg_augment:ro" \
  "$image" \
  bash -lc "livekit-wakeword setup -c '$config' && livekit-wakeword run '$config'"
