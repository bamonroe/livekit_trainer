#!/usr/bin/env python3
"""Assemble a pooled real-samples tree for one wake word.

The trainer reads real clips from ``<real_samples_dir>/<slug>/{positive,negative,
background}``. Every project's own clips already live under ``data/real/<slug>``,
but the negatives collected for one wake word are ordinary speech and noise that
serve just as well as negatives for any *other* wake word. A wake word's
*positives* are also valuable negatives for a different word: they are real
speech that sounds like a wake phrase but is not the target.

This script builds a merged tree at ``<out>/<slug>`` for a single target slug:

  positive/    the target slug's own positives (never borrowed)
  negative/    the target's own negatives + the target's own hard negatives
               + every other slug's negatives
               + every other slug's positives (reused as hard negatives)
  background/  the target's own background + every other slug's background

Plain negatives and background are ordinary speech and noise, so they pool
across every project. Hard negatives are near-miss phrases specific to *this*
wake word, so the target's own ``hard_negative`` clips train only its own model
and are never borrowed into another project's pool.

Borrowed clips are linked, not copied, so the pool stays in sync with the
canonical ``data/real`` tree and costs no extra disk. Nothing under the source
data root is modified. Point the training config's ``real_samples_dir`` at the
``<out>`` directory (default ``./data/train``) to train on the pooled set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9_]*$")
CATEGORIES = ("positive", "negative", "background")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slug", required=True, help="Target wake-word slug to assemble for")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/real"),
        help="Root of per-slug canonical clips (default: data/real)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/train"),
        help="Root of the assembled real_samples tree (default: data/train)",
    )
    parser.add_argument(
        "--no-borrow-positives",
        action="store_true",
        help="Do not reuse other wake words' positives as negatives",
    )
    parser.add_argument(
        "--no-borrow-background",
        action="store_true",
        help="Do not reuse other wake words' background clips",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy clips instead of symlinking (slower, uses disk)",
    )
    parser.add_argument(
        "--positive-boost",
        type=int,
        default=1,
        help=(
            "Replicate the target's own positives N times so your real voice "
            "carries more weight against the trainer's large synthetic positive "
            "pool (default: 1 = no boost)"
        ),
    )
    parser.add_argument(
        "--synth-root",
        type=Path,
        default=Path("data/synth_f5"),
        help=(
            "Root of the F5 voice-cloned synthetic positives, laid out as "
            "<root>/<slug>/positive (default: data/synth_f5). Up to --synth-count "
            "of these are pooled into the target's positives as a second synthetic "
            "source that carries the user's timbre."
        ),
    )
    parser.add_argument(
        "--synth-count",
        type=int,
        default=0,
        help=(
            "How many F5 voice-cloned positives to fold into training (0 = none). "
            "The trainer's own built-in TTS pool (n_samples) and the user's real "
            "positives (x --positive-boost) are the other two positive sources; "
            "together they are the model's total positive input."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help=(
            "Also write the pooled-clip counts (own/borrowed positive, negative, "
            "background) to this JSON file, so the training pipeline can record "
            "exactly how much real voice went into the model."
        ),
    )
    args = parser.parse_args()

    if not SAFE_SLUG.fullmatch(args.slug):
        raise SystemExit(f"unsafe slug: {args.slug!r}")
    if args.positive_boost < 1:
        raise SystemExit("--positive-boost must be 1 or greater")
    if args.synth_count < 0:
        raise SystemExit("--synth-count must be 0 or greater")
    # No own recordings is allowed: the trainer synthesizes positives/negatives
    # from the phrase, and other wake words' clips are still pooled in as
    # negatives/background. Warn but continue so a brand-new word can train on a
    # purely synthetic pool.
    if not (args.data_root / args.slug).is_dir():
        print(
            f"note: no recorded clips for slug {args.slug!r} under {args.data_root}; "
            "assembling a synthetic-only pool (borrowed negatives/background only)"
        )

    summary = assemble(
        slug=args.slug,
        data_root=args.data_root,
        out_root=args.out,
        borrow_positives=not args.no_borrow_positives,
        borrow_background=not args.no_borrow_background,
        copy=args.copy,
        positive_boost=args.positive_boost,
        synth_root=args.synth_root,
        synth_count=args.synth_count,
    )

    dest = args.out / args.slug
    print(f"Assembled {dest}")
    boost_note = (
        f" = {summary['positive_unique']} unique x{args.positive_boost}"
        if args.positive_boost > 1
        else ""
    )
    synth_note = (
        f", F5 synth {summary['synth_positive']}" if summary["synth_positive"] else ""
    )
    print(
        f"  positive:   {summary['positive']} "
        f"(own {summary['own_positive']}{boost_note}{synth_note})"
    )
    print(
        f"  negative:   {summary['negative']} "
        f"(own {summary['own_negative']}, own hard negatives {summary['own_hard_negative']}, "
        f"borrowed negatives {summary['borrowed_negative']}, "
        f"borrowed positives {summary['borrowed_positive']})"
    )
    print(
        f"  background: {summary['background']} "
        f"(own {summary['own_background']}, borrowed {summary['borrowed_background']})"
    )
    if summary["other_slugs"]:
        print(f"  pooled from: {', '.join(summary['other_slugs'])}")
    if args.summary_json is not None:
        # A machine-readable copy of the counts above, so the training pipeline
        # can fold the exact real-clip totals (and the boost multiplier) into the
        # model's provenance record.
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(
                {
                    **summary,
                    "positive_boost": args.positive_boost,
                    "synth_count": args.synth_count,
                },
                indent=2,
            )
        )
    return 0


def other_slugs(data_root: Path, slug: str) -> list[str]:
    """Every other project slug with clips under *data_root*, sorted."""
    if not data_root.is_dir():
        return []
    result = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir() or child.name == slug:
            continue
        if not SAFE_SLUG.fullmatch(child.name):
            continue
        result.append(child.name)
    return result


def assemble(
    slug: str,
    data_root: Path,
    out_root: Path,
    borrow_positives: bool = True,
    borrow_background: bool = True,
    copy: bool = False,
    positive_boost: int = 1,
    synth_root: Path | None = None,
    synth_count: int = 0,
) -> dict:
    dest = out_root / slug
    if dest.exists():
        shutil.rmtree(dest)
    for category in CATEGORIES:
        (dest / category).mkdir(parents=True, exist_ok=True)

    others = other_slugs(data_root, slug)
    counts = {k: 0 for k in ("own_negative", "own_hard_negative", "borrowed_negative", "borrowed_positive", "own_background", "borrowed_background")}
    contributing: set[str] = set()

    # Positives come from up to three sources that together are the model's total
    # positive input:
    #   1. the user's own real recordings, optionally replicated `positive_boost`
    #      times so they weigh more against the synthetic pools;
    #   2. up to `synth_count` F5 voice-cloned clips (data/synth_f5/<slug>), a
    #      synthetic source that carries the user's timbre;
    #   3. the trainer's own built-in TTS pool (n_samples), synthesized later by
    #      `livekit-wakeword setup`, not pooled here.
    n_pos_unique = _count_wavs(data_root / slug / "positive")
    own_pos = _link_category(
        data_root / slug / "positive", dest / "positive", slug, copy, boost=positive_boost
    )
    synth_pos = 0
    if synth_root is not None and synth_count > 0:
        synth_pos = _link_category(
            synth_root / slug / "positive",
            dest / "positive",
            f"{slug}_f5synth",
            copy,
            limit=synth_count,
        )
    n_pos = own_pos + synth_pos

    # Negatives: own negatives, own hard negatives (project-scoped, never
    # borrowed from or lent to another slug), then every other slug's negatives,
    # then (optionally) every other slug's positives reused as hard negatives.
    counts["own_negative"] = _link_category(data_root / slug / "negative", dest / "negative", slug, copy)
    counts["own_hard_negative"] = _link_category(
        data_root / slug / "hard_negative", dest / "negative", f"{slug}_hardneg", copy
    )
    for other in others:
        got = _link_category(data_root / other / "negative", dest / "negative", other, copy)
        counts["borrowed_negative"] += got
        if got:
            contributing.add(other)
        if borrow_positives:
            got = _link_category(data_root / other / "positive", dest / "negative", other, copy)
            counts["borrowed_positive"] += got
            if got:
                contributing.add(other)

    # Background: own, then (optionally) every other slug's background.
    counts["own_background"] = _link_category(data_root / slug / "background", dest / "background", slug, copy)
    if borrow_background:
        for other in others:
            got = _link_category(data_root / other / "background", dest / "background", other, copy)
            counts["borrowed_background"] += got
            if got:
                contributing.add(other)

    return {
        "positive": n_pos,
        "positive_unique": n_pos_unique,
        "own_positive": own_pos,
        "synth_positive": synth_pos,
        "negative": counts["own_negative"] + counts["own_hard_negative"] + counts["borrowed_negative"] + counts["borrowed_positive"],
        "background": counts["own_background"] + counts["borrowed_background"],
        "own_negative": counts["own_negative"],
        "own_hard_negative": counts["own_hard_negative"],
        "borrowed_negative": counts["borrowed_negative"],
        "borrowed_positive": counts["borrowed_positive"],
        "own_background": counts["own_background"],
        "borrowed_background": counts["borrowed_background"],
        "other_slugs": sorted(contributing),
    }


def _count_wavs(src_dir: Path) -> int:
    return len(list(src_dir.glob("*.wav"))) if src_dir.is_dir() else 0


def _link_category(
    src_dir: Path, dest_dir: Path, src_slug: str, copy: bool, boost: int = 1, limit: int | None = None
) -> int:
    """Link/copy every ``*.wav`` in *src_dir* into *dest_dir*, namespaced by slug.

    Returns the number of clips placed. Names are prefixed with the source slug
    so clips pooled from different wake words never collide in the flat dir. When
    *boost* > 1 each source clip is placed *boost* times with a distinct suffix,
    so the trainer treats every replica as its own clip and samples/augments it,
    raising that source's weight in the pool. When *limit* is set, at most that
    many source clips are placed (used to cap the F5 synth pool to the count the
    user asked training for even if more clips exist on disk).
    """
    if not src_dir.is_dir():
        return 0
    placed = 0
    clips = sorted(src_dir.glob("*.wav"))
    if limit is not None:
        clips = clips[:limit]
    for clip in clips:
        for replica in range(boost):
            suffix = "" if boost == 1 else f"__x{replica}"
            target = dest_dir / f"{src_slug}__{clip.stem}{suffix}{clip.suffix}"
            if copy:
                shutil.copyfile(clip, target)
            else:
                # Relative symlink so the pool resolves inside the trainer's
                # /work bind mount regardless of the host path.
                rel = os.path.relpath(clip.resolve(), dest_dir.resolve())
                target.symlink_to(rel)
            placed += 1
    return placed


if __name__ == "__main__":
    raise SystemExit(main())
