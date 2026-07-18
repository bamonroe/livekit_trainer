#!/usr/bin/env python3
"""Turn model evaluation mistakes into false_positive / false_negative clips.

This is the emitter for the correction half of the collection loop. A trained
model (or a runtime scorer) is run over a set of already-labeled clips, and the
score for each clip is written into a corrections batch alongside its audio.
This script compares each score against the detection threshold and files only
the *mistakes* into the real-data tree, tagged with the distinct label the rest
of the pipeline already understands:

- A clip that should fire (`positive` / `false_negative`) but scored below the
  threshold is a miss -> `false_negative` -> trains as an extra `positive`.
- A clip that should stay silent (`negative` / `hard_negative` / `background` /
  `false_positive`) but scored at or above the threshold is a wrongful trigger
  -> `false_positive` -> trains as an extra `negative`.

Clips the model got right are skipped. The richer mistake label and the score
that produced it are preserved in `metadata.jsonl` so later tools can inspect or
rebalance the correction set.

The batch layout mirrors an Android bundle:

    <batch>/corrections.json
    <batch>/audio/<clip>.wav
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


# Ground-truth label -> whether the model is expected to fire on that clip.
# Mirrors the label families the trainer already collapses (see import_bundle).
SHOULD_FIRE = {
    "positive": True,
    "false_negative": True,
    "negative": False,
    "hard_negative": False,
    "false_positive": False,
    "background": False,
}

# Mistake label -> LiveKit training category it feeds.
MISTAKE_TO_CATEGORY = {
    "false_negative": "positive",
    "false_positive": "negative",
}

SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, help="Corrections batch directory")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/real"),
        help="Destination real-data root, default: data/real",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the detection threshold from the manifest",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rewrite clip files or metadata records that already exist",
    )
    args = parser.parse_args()

    batch = args.batch.resolve()
    manifest_path = batch / "corrections.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing corrections manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)

    threshold = args.threshold
    if threshold is None:
        threshold = manifest.get("model", {}).get("threshold")
    if threshold is None:
        raise SystemExit(
            "no threshold given; set model.threshold in the manifest or pass --threshold"
        )
    threshold = float(threshold)

    slug = manifest["wake_word"]["slug"]
    dest_root = args.data_root / slug
    dest_root.mkdir(parents=True, exist_ok=True)
    metadata_path = dest_root / "metadata.jsonl"

    imported = {"false_positive": 0, "false_negative": 0}
    correct = 0
    with metadata_path.open("a", encoding="utf-8") as metadata:
        for index, entry in enumerate(manifest["corrections"]):
            mistake = classify(entry, threshold, index)
            if mistake is None:
                correct += 1
                continue

            category = MISTAKE_TO_CATEGORY[mistake]
            src = resolve_batch_file(batch, entry["file"])
            dest_dir = dest_root / category
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_name = f"{entry['id']}_{safe_filename(entry.get('spoken_phrase') or mistake)}.wav"
            dest = dest_dir / dest_name
            if args.skip_existing and dest.exists():
                continue
            shutil.copy2(src, dest)

            record = {
                "source_batch": str(batch),
                "source_file": entry["file"],
                "imported_file": str(dest),
                "livekit_category": category,
                "original_label": mistake,
                "ground_truth_label": entry.get("label"),
                "score": entry.get("score"),
                "threshold": threshold,
                "model": manifest.get("model", {}),
                "wake_word": manifest["wake_word"],
                "correction": entry,
            }
            metadata.write(json.dumps(record, sort_keys=True) + "\n")
            imported[mistake] += 1

    print(
        f"Imported {imported['false_negative']} false_negative + "
        f"{imported['false_positive']} false_positive corrections into {dest_root} "
        f"({correct} correctly-classified clips skipped)"
    )
    return 0


def classify(entry: dict[str, Any], threshold: float, index: int) -> str | None:
    """The mistake label for a scored clip, or None if the model got it right.

    An explicit ``mistake`` field wins (a manually triaged correction); otherwise
    the mistake is derived from the ground-truth label and the score.
    """
    explicit = entry.get("mistake")
    if explicit:
        if explicit not in MISTAKE_TO_CATEGORY:
            raise SystemExit(
                f"correction {index} has unknown mistake {explicit!r}"
            )
        return explicit

    label = entry.get("label")
    if label not in SHOULD_FIRE:
        raise SystemExit(
            f"correction {index} needs a known label to derive a mistake; got {label!r}"
        )
    score = entry.get("score")
    if not isinstance(score, (int, float)):
        raise SystemExit(f"correction {index} missing numeric score")

    fired = float(score) >= threshold
    if SHOULD_FIRE[label] and not fired:
        return "false_negative"
    if not SHOULD_FIRE[label] and fired:
        return "false_positive"
    return None


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise SystemExit(f"unsupported schema_version: {manifest.get('schema_version')}")

    wake_word = manifest.get("wake_word")
    if not isinstance(wake_word, dict):
        raise SystemExit("manifest wake_word must be an object")

    slug = wake_word.get("slug")
    if not isinstance(slug, str) or not SAFE_SLUG.fullmatch(slug):
        raise SystemExit(f"unsafe or missing wake_word.slug: {slug!r}")

    corrections = manifest.get("corrections")
    if not isinstance(corrections, list):
        raise SystemExit("manifest corrections must be a list")

    for index, entry in enumerate(corrections):
        if not isinstance(entry, dict):
            raise SystemExit(f"correction {index} must be an object")
        for field in ("id", "file"):
            if not entry.get(field):
                raise SystemExit(f"correction {index} missing {field}")


def resolve_batch_file(batch: Path, relative: str) -> Path:
    candidate = (batch / relative).resolve()
    if batch not in candidate.parents:
        raise SystemExit(f"correction path escapes batch: {relative}")
    if not candidate.is_file():
        raise SystemExit(f"missing correction file: {candidate}")
    return candidate


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "clip"


if __name__ == "__main__":
    raise SystemExit(main())
