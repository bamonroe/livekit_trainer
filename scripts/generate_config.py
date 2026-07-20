#!/usr/bin/env python3
"""Generate a LiveKit wake-word YAML config."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Android bundle manifest.json")
    parser.add_argument("--slug", help="Wake-word slug, required without --manifest")
    parser.add_argument("--phrase", help="Target phrase, required without --manifest")
    parser.add_argument(
        "--negative",
        action="append",
        default=[],
        help="Custom negative phrase; may be repeated",
    )
    parser.add_argument(
        "--negatives-file",
        type=Path,
        default=None,
        help=(
            "Path to a durable list of hard-negative phrases, one per line "
            "(blank lines and lines starting with # are ignored). These are the "
            "curated near-misses that must never trigger; keeping them in a "
            "sidecar file means regenerating the config from the app can't drop "
            "them. Merged with any --negative flags and manifest negatives."
        ),
    )
    parser.add_argument("--out", type=Path, help="Output YAML path")
    parser.add_argument(
        "--real-samples-dir",
        default="./data/train",
        help=(
            "Root of the real recorded clips the trainer injects, laid out as "
            "<dir>/<slug>/{positive,negative,background}. Defaults to ./data/train, "
            "the pooled tree built by assemble_training_data.py. Use ./data/real "
            "for this project's own clips only, or empty to disable injection."
        ),
    )
    parser.add_argument(
        "--positive-per-batch",
        type=int,
        default=None,
        help=(
            "Overweight positives by raising the per-batch positive slot count "
            "(trainer default 50). Since negatives are pooled across wake words and "
            "grow far faster than positives, bumping this keeps positive gradient "
            "signal strong. Setting it emits an explicit batch_n_per_class block."
        ),
    )
    parser.add_argument("--adversarial-negative-per-batch", type=int, default=50)
    parser.add_argument("--acav-per-batch", type=int, default=1024)
    parser.add_argument("--background-noise-per-batch", type=int, default=50)
    parser.add_argument(
        "--personal",
        action="store_true",
        help=(
            "Preset for a personal, single-voice model: shrink the synthetic "
            "positive pool (n_samples) and raise the per-batch positive count so "
            "your own recorded positives — ideally replicated with "
            "assemble_training_data.py --positive-boost — dominate training. Any "
            "explicit --n-samples / --n-samples-val / --positive-per-batch wins."
        ),
    )
    # Sample-count knobs default to None so --personal vs standard presets can
    # supply the base value while an explicit flag still overrides either.
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-samples-val", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--target-fp-per-hour", type=float, default=0.2)
    parser.add_argument("--model-size", default="medium")
    parser.add_argument(
        "--no-context-fix",
        dest="context_fix",
        action="store_false",
        help=(
            "Disable the leading-context augmentation recipe. By default the "
            "config sets augmentation.background_paths so the trainer re-mixes "
            "real noise and ordinary-speech negatives into the ~1s in front of "
            "every positive instead of the zero-fill that align_clip_to_end "
            "leaves. That zero lead is a cue that never occurs on a live mic and "
            "is the cause of the streaming-recall gap (99%% padded eval, ~1%% "
            "mid-sentence). The recipe only bites if the trainer image carries "
            "trainer/patches/augment_ctxfix.py over livekit.wakeword.data.augment."
        ),
    )
    parser.add_argument(
        "--token-type",
        choices=["start", "end"],
        default="end",
        help=(
            "Where the phrase sits in an utterance, which decides the leading "
            "context the recipe trains on. 'end' (default, e.g. \"all set\") lands "
            "at the end of speech, so its realistic lead is prior words — the "
            "ordinary-speech negatives are mixed into the pre-phrase window. "
            "'start' begins an utterance from a quiet-but-noisy room with no "
            "speech before it, so only ambient background is mixed in; leading "
            "speech is deliberately withheld so a start token never learns to "
            "expect words in front of the trigger."
        ),
    )
    parser.set_defaults(context_fix=True)
    args = parser.parse_args()

    config = config_from_args(args)
    out = args.out or Path("trainer/configs") / f"{config['model_name']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_yaml(config), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        wake_word = manifest["wake_word"]
        slug = wake_word["slug"]
        target_phrases = wake_word.get("target_phrases") or [wake_word["phrase"]]
        negatives = list(wake_word.get("negative_phrases") or [])
    else:
        if not args.slug or not args.phrase:
            raise SystemExit("--slug and --phrase are required without --manifest")
        slug = args.slug
        target_phrases = [args.phrase]
        negatives = []

    if not SAFE_SLUG.fullmatch(slug):
        raise SystemExit(f"unsafe slug: {slug!r}")

    if args.negatives_file and args.negatives_file.exists():
        for line in args.negatives_file.read_text(encoding="utf-8").splitlines():
            phrase = line.strip()
            if phrase and not phrase.startswith("#"):
                negatives.append(phrase)

    negatives.extend(args.negative)
    negatives = dedupe([item.strip() for item in negatives if item.strip()])

    # Base sample counts from the chosen preset; explicit flags override below.
    base_n_samples = 3_000 if args.personal else 20_000
    base_n_samples_val = 600 if args.personal else 4_000
    base_positive_per_batch = 100 if args.personal else None

    n_samples = args.n_samples if args.n_samples is not None else base_n_samples
    n_samples_val = args.n_samples_val if args.n_samples_val is not None else base_n_samples_val
    positive_per_batch = (
        args.positive_per_batch if args.positive_per_batch is not None else base_positive_per_batch
    )

    batch_n_per_class = None
    if positive_per_batch is not None:
        batch_n_per_class = {
            "positive": positive_per_batch,
            "adversarial_negative": args.adversarial_negative_per_batch,
            "ACAV100M_sample": args.acav_per_batch,
            "background_noise": args.background_noise_per_batch,
        }

    header_comment = None
    if args.personal:
        header_comment = (
            "# personal preset: real recorded positives weighted against a reduced\n"
            "# synthetic pool. Pair with assemble_training_data.py --positive-boost."
        )

    real_samples_dir = (args.real_samples_dir or "").strip() or None

    # Leading-context augmentation recipe: fill the pre-phrase window with real
    # sound instead of the zero-fill align_clip_to_end leaves. Always include the
    # trainer's ambient corpus and our own ambient background takes. Whether the
    # ordinary-speech negatives join depends on token type: an END token (end of
    # speech) has prior words in front, so it trains on leading speech; a START
    # token begins an utterance from a quiet room, so speech is withheld and only
    # ambient noise fills the lead. mix_with_background picks one at random per
    # clip.
    background_paths = None
    if args.context_fix:
        background_paths = ["./data/backgrounds"]
        if args.token_type == "end" and real_samples_dir:
            background_paths.append(f"{real_samples_dir.rstrip('/')}/{slug}/negative")
        background_paths.append(f"./data/real/{slug}/background")

    return {
        "model_name": slug,
        "target_phrases": dedupe(target_phrases),
        "custom_negative_phrases": negatives,
        "data_dir": "./data",
        "output_dir": "./output",
        "real_samples_dir": real_samples_dir,
        "background_paths": background_paths,
        "token_type": args.token_type,
        "batch_n_per_class": batch_n_per_class,
        "header_comment": header_comment,
        "model": {
            "model_type": "conv_attention",
            "model_size": args.model_size,
        },
        "n_samples": n_samples,
        "n_samples_val": n_samples_val,
        "steps": args.steps,
        "target_fp_per_hour": args.target_fp_per_hour,
    }


def render_yaml(config: dict[str, Any]) -> str:
    lines: list[str] = []
    if config.get("header_comment"):
        lines.append(config["header_comment"])
        lines.append("")
    lines.extend(
        [
            f"model_name: {yaml_string(config['model_name'])}",
            "target_phrases:",
        ]
    )
    lines.extend(f"  - {yaml_string(item)}" for item in config["target_phrases"])
    lines.append("")
    lines.append("custom_negative_phrases:")
    if config["custom_negative_phrases"]:
        lines.extend(f"  - {yaml_string(item)}" for item in config["custom_negative_phrases"])
    else:
        lines.append("  []")
    lines.extend(
        [
            "",
            f"data_dir: {yaml_string(config['data_dir'])}",
            f"output_dir: {yaml_string(config['output_dir'])}",
        ],
    )
    if config.get("real_samples_dir"):
        lines.append(f"real_samples_dir: {yaml_string(config['real_samples_dir'])}")
    if config.get("batch_n_per_class"):
        lines.append("")
        lines.append("batch_n_per_class:")
        for key, value in config["batch_n_per_class"].items():
            lines.append(f"  {key}: {value}")
    if config.get("background_paths"):
        token_type = config.get("token_type", "end")
        lead = "noise + prior speech" if token_type == "end" else "ambient noise only (no leading speech)"
        lines.append("")
        lines.append("augmentation:")
        lines.append(f"  # Leading-context fix ({token_type} token: {lead}): re-mix these into")
        lines.append("  # the full 2s window after placement so the pre-phrase region carries")
        lines.append("  # real sound, not zeros. Needs the augment_ctxfix patch in the image.")
        lines.append("  background_paths:")
        for path in config["background_paths"]:
            lines.append(f"    - {yaml_string(path)}")
    lines.extend(
        [
            "",
            "model:",
            f"  model_type: {yaml_string(config['model']['model_type'])}",
            f"  model_size: {yaml_string(config['model']['model_size'])}",
            "",
            f"n_samples: {config['n_samples']}",
            f"n_samples_val: {config['n_samples_val']}",
            f"steps: {config['steps']}",
            f"target_fp_per_hour: {config['target_fp_per_hour']}",
            "",
        ],
    )
    return "\n".join(lines)


def yaml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
