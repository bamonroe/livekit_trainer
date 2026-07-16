#!/usr/bin/env python3
"""Import an Android training bundle into the LiveKit real-data layout."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


LABEL_TO_CATEGORY = {
    "positive": "positive",
    "false_negative": "positive",
    "negative": "negative",
    "hard_negative": "negative",
    "false_positive": "negative",
    "background": "background",
}

SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Unzipped Android training bundle")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/real"),
        help="Destination real-data root, default: data/real",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rewrite clip files or metadata records that already exist",
    )
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)

    slug = manifest["wake_word"]["slug"]
    dest_root = args.data_root / slug
    dest_root.mkdir(parents=True, exist_ok=True)

    metadata_path = dest_root / "metadata.jsonl"
    imported = 0
    with metadata_path.open("a", encoding="utf-8") as metadata:
        for clip in manifest["clips"]:
            label = clip["label"]
            category = LABEL_TO_CATEGORY[label]
            src = resolve_bundle_file(bundle, clip["file"])
            dest_dir = dest_root / category
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_name = f"{clip['id']}_{safe_filename(clip.get('spoken_phrase') or label)}.wav"
            dest = dest_dir / dest_name
            if args.skip_existing and dest.exists():
                continue
            shutil.copy2(src, dest)

            record = {
                "source_bundle": str(bundle),
                "source_file": clip["file"],
                "imported_file": str(dest),
                "livekit_category": category,
                "original_label": label,
                "wake_word": manifest["wake_word"],
                "clip": clip,
            }
            metadata.write(json.dumps(record, sort_keys=True) + "\n")
            imported += 1

    print(f"Imported {imported} clips into {dest_root}")
    return 0


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise SystemExit(f"unsupported schema_version: {manifest.get('schema_version')}")

    wake_word = manifest.get("wake_word")
    if not isinstance(wake_word, dict):
        raise SystemExit("manifest wake_word must be an object")

    slug = wake_word.get("slug")
    if not isinstance(slug, str) or not SAFE_SLUG.fullmatch(slug):
        raise SystemExit(f"unsafe or missing wake_word.slug: {slug!r}")

    clips = manifest.get("clips")
    if not isinstance(clips, list):
        raise SystemExit("manifest clips must be a list")

    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            raise SystemExit(f"clip {index} must be an object")
        for field in ("id", "file", "label"):
            if not clip.get(field):
                raise SystemExit(f"clip {index} missing {field}")
        if clip["label"] not in LABEL_TO_CATEGORY:
            raise SystemExit(f"clip {index} has unknown label {clip['label']!r}")


def resolve_bundle_file(bundle: Path, relative: str) -> Path:
    candidate = (bundle / relative).resolve()
    if bundle not in candidate.parents:
        raise SystemExit(f"clip path escapes bundle: {relative}")
    if not candidate.is_file():
        raise SystemExit(f"missing clip file: {candidate}")
    return candidate


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "clip"


if __name__ == "__main__":
    raise SystemExit(main())
