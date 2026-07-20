#!/usr/bin/env bash
# Train a config with the REALISTIC-positive augmentation patch mounted over the
# installed trainer module. Like train_ctxfix.sh, but the mounted augment module
# composites each positive as one clear voice on a background bed (optional
# filler speech -> gap -> wake word -> room-tone margin) instead of overlapping
# speech on the phrase. See trainer/patches/augment_realistic.py and
# trainer/configs/all_set_realistic.yaml.
#
# usage: trainer/scripts/train_realistic.sh <config.yaml> [realistic_sidecar.yaml]
#
# Both paths are relative to the repo root. If the sidecar is omitted it is
# guessed as configs/<model_config_stem>.realistic.yaml next to the config.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: trainer/scripts/train_realistic.sh <config.yaml> [realistic_sidecar.yaml]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config="$1"

if [[ $# -eq 2 ]]; then
  sidecar="$2"
else
  # Guess: strip a trailing _realistic from the stem, then <stem>.realistic.yaml
  dir="$(dirname "$config")"
  stem="$(basename "$config" .yaml)"
  stem="${stem%_realistic}"
  sidecar="$dir/$stem.realistic.yaml"
fi

image="${LIVEKIT_WAKEWORD_TRAINER_IMAGE:-livekit-wakeword-trainer:latest}"
patch="$repo_root/trainer/patches/augment_realistic.py"
pkg_augment="/opt/conda/lib/python3.11/site-packages/livekit/wakeword/data/augment.py"
gpu_args=()

if [[ "${LIVEKIT_WAKEWORD_USE_GPU:-1}" != "0" ]]; then
  gpu_args=(--gpus all)
fi

for f in "$patch" "$repo_root/$config" "$repo_root/$sidecar"; do
  if [[ ! -f "$f" ]]; then
    echo "missing file: $f" >&2
    exit 1
  fi
done

echo "config:   $config"
echo "sidecar:  $sidecar  (AUG_REALISTIC_CONFIG=/work/$sidecar)"

mkdir -p "$repo_root/data" "$repo_root/output"

docker run --rm "${gpu_args[@]}" \
  -v "$repo_root:/work" \
  -v "$patch:$pkg_augment:ro" \
  -e "AUG_REALISTIC_CONFIG=/work/$sidecar" \
  "$image" \
  bash -lc "livekit-wakeword setup -c '$config' && livekit-wakeword run '$config'"
