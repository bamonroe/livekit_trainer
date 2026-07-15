#!/usr/bin/env python3
"""Validate an Android training bundle manifest and WAV files."""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

from import_bundle import LABEL_TO_CATEGORY, resolve_bundle_file, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Unzipped Android training bundle")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    validate_manifest(manifest)

    warnings: list[str] = []
    for clip in manifest["clips"]:
        src = resolve_bundle_file(bundle, clip["file"])
        warnings.extend(validate_wav(src, clip["id"]))
        if clip["label"] not in LABEL_TO_CATEGORY:
            warnings.append(f"{clip['id']}: unknown label {clip['label']}")

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    print(f"Validated {len(manifest['clips'])} clips")
    return 0


def validate_wav(path: Path, clip_id: str) -> list[str]:
    warnings: list[str] = []
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        audio = wav.readframes(frames)

    duration = frames / sample_rate if sample_rate else 0
    if channels != 1:
        warnings.append(f"{clip_id}: expected mono, got {channels} channels")
    if sample_width != 2:
        warnings.append(f"{clip_id}: expected 16-bit PCM, got sample width {sample_width}")
    if sample_rate != 16_000:
        warnings.append(f"{clip_id}: expected 16000 Hz, got {sample_rate}")
    if duration <= 0:
        warnings.append(f"{clip_id}: zero duration")
    if duration > 5:
        warnings.append(f"{clip_id}: long clip {duration:.2f}s")

    peak, rms = pcm16_peak_rms(audio)
    if peak >= 32760:
        warnings.append(f"{clip_id}: possible clipping, peak={peak}")
    if rms < 50 and duration > 0:
        warnings.append(f"{clip_id}: very quiet audio, rms={rms:.1f}")
    return warnings


def pcm16_peak_rms(audio: bytes) -> tuple[int, float]:
    if len(audio) < 2:
        return 0, 0.0
    count = len(audio) // 2
    peak = 0
    total = 0
    for index in range(0, count * 2, 2):
        sample = int.from_bytes(audio[index : index + 2], "little", signed=True)
        absolute = abs(sample)
        peak = max(peak, absolute)
        total += sample * sample
    return peak, math.sqrt(total / count)


if __name__ == "__main__":
    raise SystemExit(main())
