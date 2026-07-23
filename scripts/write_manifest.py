#!/usr/bin/env python3
"""Assemble a trained model's complete provenance manifest.

Run at the end of ``train_job.sh`` once the model artifacts exist. Merges three
sources into a single ``manifest.json`` that the sync-server folds verbatim into
the ``models`` table (as ``params_json``) while promoting the key knobs to typed
columns:

1. Runtime facts passed as ``MANIFEST_ENV_*`` environment variables (slug,
   phrase, run id, checksums, git commit, trainer image, timestamps, and the
   raw knob values the job was launched with).
2. ``config_params.json`` — the *resolved* hyperparameters emitted by
   ``generate_config.py`` (effective ``n_samples`` etc. after preset defaults).
3. ``assemble_summary.json`` — the real vs synthetic clip counts emitted by
   ``assemble_training_data.py`` (how much of the user's own voice went in).
4. ``<slug>_eval.json`` / ``<slug>_metrics.json`` — the trainer's own scores.

Every source is optional: a missing or unreadable sidecar simply omits its
fields, so an interrupted run still writes whatever provenance it has.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def env(name: str, default: str = "") -> str:
    return os.environ.get(f"MANIFEST_ENV_{name}", default)


def load_json(path_str: str) -> dict | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def to_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _total_positive(params: dict, summary: dict) -> int | None:
    """The model's total positive input across all three sources.

    kokoro (built-in TTS pool) + own real (already boosted) + F5 synth. Returns
    None only if the built-in pool count is unknown, since it dominates."""
    kokoro = params.get("n_samples")
    if not isinstance(kokoro, int):
        return None
    own = summary.get("own_positive") or summary.get("positive") or 0
    synth = summary.get("synth_positive") or 0
    return kokoro + own + synth


def main() -> int:
    slug = env("SLUG")
    run_id = env("RUNID")

    params = load_json(env("PARAMS")) or {}
    summary = load_json(env("SUMMARY")) or {}
    eval_metrics = load_json(env("EVAL"))
    train_metrics = load_json(env("METRICS"))

    manifest: dict = {
        # Identity + artifact.
        "slug": slug,
        "phrase": env("PHRASE"),
        "run_id": run_id,
        "onnx_path": f"runs/{run_id}/{slug}.onnx",
        "onnx_sha256": env("ONNX_SHA") or None,
        "pt_sha256": env("PT_SHA") or None,
        "onnx_bytes": to_int(env("ONNX_BYTES")),
        # Code + environment provenance.
        "git_commit": env("GIT_COMMIT") or None,
        "trainer_image": env("IMAGE") or None,
        "started_at": env("STARTED") or None,
        "finished_at": env("FINISHED") or None,
        # Knobs the job was launched with.
        "steps": to_int(env("STEPS")),
        "model_size": env("MODEL_SIZE") or None,
        "personal": env("PERSONAL") == "1",
        "positive_boost": to_int(env("POSITIVE_BOOST"), 1),
        "target_fp_per_hour": to_float(env("TARGET_FP")),
        "token_type": env("TOKEN_TYPE") or None,
        # Resolved hyperparameters (effective values after preset defaults).
        "model_type": params.get("model_type"),
        "n_samples": params.get("n_samples"),
        "n_samples_val": params.get("n_samples_val"),
        "positive_per_batch": params.get("positive_per_batch"),
        "context_fix": params.get("context_fix"),
        "background_paths": params.get("background_paths"),
        "custom_negative_count": params.get("custom_negative_count"),
        "target_phrases": params.get("target_phrases"),
        # Positive training data by source. The model's total positive input is
        # three streams: the trainer's built-in TTS pool (kokoro_positive =
        # n_samples), the user's own real recordings replicated x positive_boost
        # (real_positive), and the F5 voice-cloned clips (synth_positive).
        "real_positive": summary.get("own_positive", summary.get("positive")),
        "real_positive_unique": summary.get("positive_unique"),
        "synth_positive": summary.get("synth_positive"),
        "kokoro_positive": params.get("n_samples"),
        "total_positive_input": _total_positive(params, summary),
        "real_negative": summary.get("negative"),
        "real_background": summary.get("background"),
        "own_negative": summary.get("own_negative"),
        "own_background": summary.get("own_background"),
        "borrowed_positive": summary.get("borrowed_positive"),
        "borrowed_negative": summary.get("borrowed_negative"),
        "borrowed_background": summary.get("borrowed_background"),
        "pooled_from": summary.get("other_slugs"),
        # The trainer's own scores.
        "eval": eval_metrics,
        "metrics": train_metrics,
    }

    out = Path(env("OUT"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote provenance manifest {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
