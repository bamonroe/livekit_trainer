#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$repo_root/trainer/scripts/train.sh" trainer/configs/smoke.yaml
