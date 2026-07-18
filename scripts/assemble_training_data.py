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
  negative/    the target's own negatives + every other slug's negatives
               + every other slug's positives (reused as hard negatives)
  background/  the target's own background + every other slug's background

Borrowed clips are linked, not copied, so the pool stays in sync with the
canonical ``data/real`` tree and costs no extra disk. Nothing under the source
data root is modified. Point the training config's ``real_samples_dir`` at the
``<out>`` directory (default ``./data/train``) to train on the pooled set.
"""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    if not SAFE_SLUG.fullmatch(args.slug):
        raise SystemExit(f"unsafe slug: {args.slug!r}")
    if not (args.data_root / args.slug).is_dir():
        raise SystemExit(f"no data for slug {args.slug!r} under {args.data_root}")

    summary = assemble(
        slug=args.slug,
        data_root=args.data_root,
        out_root=args.out,
        borrow_positives=not args.no_borrow_positives,
        borrow_background=not args.no_borrow_background,
        copy=args.copy,
    )

    dest = args.out / args.slug
    print(f"Assembled {dest}")
    print(f"  positive:   {summary['positive']} (own)")
    print(
        f"  negative:   {summary['negative']} "
        f"(own {summary['own_negative']}, borrowed negatives {summary['borrowed_negative']}, "
        f"borrowed positives {summary['borrowed_positive']})"
    )
    print(
        f"  background: {summary['background']} "
        f"(own {summary['own_background']}, borrowed {summary['borrowed_background']})"
    )
    if summary["other_slugs"]:
        print(f"  pooled from: {', '.join(summary['other_slugs'])}")
    return 0


def other_slugs(data_root: Path, slug: str) -> list[str]:
    """Every other project slug with clips under *data_root*, sorted."""
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
) -> dict:
    dest = out_root / slug
    if dest.exists():
        shutil.rmtree(dest)
    for category in CATEGORIES:
        (dest / category).mkdir(parents=True, exist_ok=True)

    others = other_slugs(data_root, slug)
    counts = {k: 0 for k in ("own_negative", "borrowed_negative", "borrowed_positive", "own_background", "borrowed_background")}
    contributing: set[str] = set()

    # Positives: own only.
    n_pos = _link_category(data_root / slug / "positive", dest / "positive", slug, copy)

    # Negatives: own negatives, then every other slug's negatives, then
    # (optionally) every other slug's positives reused as hard negatives.
    counts["own_negative"] = _link_category(data_root / slug / "negative", dest / "negative", slug, copy)
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
        "negative": counts["own_negative"] + counts["borrowed_negative"] + counts["borrowed_positive"],
        "background": counts["own_background"] + counts["borrowed_background"],
        "own_negative": counts["own_negative"],
        "borrowed_negative": counts["borrowed_negative"],
        "borrowed_positive": counts["borrowed_positive"],
        "own_background": counts["own_background"],
        "borrowed_background": counts["borrowed_background"],
        "other_slugs": sorted(contributing),
    }


def _link_category(src_dir: Path, dest_dir: Path, src_slug: str, copy: bool) -> int:
    """Link/copy every ``*.wav`` in *src_dir* into *dest_dir*, namespaced by slug.

    Returns the number of clips placed. Names are prefixed with the source slug
    so clips pooled from different wake words never collide in the flat dir.
    """
    if not src_dir.is_dir():
        return 0
    placed = 0
    for clip in sorted(src_dir.glob("*.wav")):
        target = dest_dir / f"{src_slug}__{clip.name}"
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
