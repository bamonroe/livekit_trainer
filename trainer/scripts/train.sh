#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: trainer/scripts/train.sh <config-yaml-relative-to-repo-root>" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config="$1"
image="${LIVEKIT_WAKEWORD_TRAINER_IMAGE:-livekit-wakeword-trainer:latest}"
gpu_args=()

if [[ "${LIVEKIT_WAKEWORD_USE_GPU:-1}" != "0" ]]; then
  gpu_args=(--gpus all)
fi

mkdir -p "$repo_root/data" "$repo_root/output"

docker build -f "$repo_root/trainer/Dockerfile.trainer" -t "$image" "$repo_root/trainer"

docker run --rm "${gpu_args[@]}" \
  -v "$repo_root:/work" \
  "$image" \
  bash -lc "livekit-wakeword setup -c '$config' && livekit-wakeword run -c '$config'"
