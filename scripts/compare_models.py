#!/usr/bin/env python3
"""Head-to-head wake-word model comparison suite.

Replays the stored *test* takes (the `test_` recordings the app collects on its
own channel, which never enter the training pool) through several trained models
at once via the scorer's ``POST /compare`` endpoint, then scores each model's
detection curve against the Whisper transcript ground truth. Prints a per-model
recall / false-fire table so a freshly trained model can be judged against the
one it is meant to replace instead of eyeballed one curve at a time.

Ground truth comes from ``data/trainer.db``: each recording's current
transcript, searched for the project's target phrase; the tail (``end_ms``) of
each match is where a tail-aligned model should fire. Detection accounting
mirrors the app's Test tab: a hit is any score >= threshold within
[end-0.6s, end+0.5s]; a false fire is an above-threshold run whose peak sits
more than 0.7s from every target tail.

Stdlib only, so it runs from the host, the dev container, or anywhere that can
reach the scorer (default http://localhost:8780).

Example:
    python3 scripts/compare_models.py \
        --slug all_set \
        --models all_set,all_set_prev_silencepad \
        --mode full --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sqlite3
import sys
import urllib.request
import uuid
from pathlib import Path


# ---- ground truth ------------------------------------------------------

def norm(w: str) -> str:
    return "".join(c for c in w.lower() if c.isalpha())


def target_ends_ms(words: list[tuple[str, int, int]], phrase_tokens: list[str]) -> list[float]:
    """Tail (end_ms/1000, in seconds) of every consecutive run of words matching
    the target phrase tokens."""
    n = len(phrase_tokens)
    ends: list[float] = []
    normed = [norm(w) for w, _, _ in words]
    for i in range(len(words) - n + 1):
        if normed[i:i + n] == phrase_tokens:
            ends.append(words[i + n - 1][2] / 1000.0)
    return ends


def load_test_recordings(db: Path, slug: str, prefix: str) -> list[dict]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    phrase = conn.execute(
        "SELECT phrase FROM projects WHERE slug=?", (slug,)
    ).fetchone()
    if phrase is None:
        conn.close()
        sys.exit(f"no project with slug {slug!r} in {db}")
    phrase_tokens = [norm(t) for t in phrase["phrase"].split() if norm(t)]

    rows = conn.execute(
        "SELECT id, source_wav, duration_ms FROM bulk_recordings "
        "WHERE project_slug=? AND id LIKE ? ORDER BY imported_at_ms",
        (slug, f"{prefix}%"),
    ).fetchall()

    out: list[dict] = []
    for r in rows:
        t = conn.execute(
            "SELECT id FROM transcripts WHERE recording_id=? AND is_current=1 "
            "ORDER BY version DESC LIMIT 1",
            (r["id"],),
        ).fetchone()
        ends: list[float] = []
        if t is not None:
            words = conn.execute(
                "SELECT word, start_ms, end_ms FROM transcript_words "
                "WHERE transcript_id=? ORDER BY ordinal",
                (t["id"],),
            ).fetchall()
            ends = target_ends_ms([(w["word"], w["start_ms"], w["end_ms"]) for w in words],
                                  phrase_tokens)
        out.append({
            "id": r["id"],
            "source_wav": r["source_wav"],
            "duration_ms": r["duration_ms"],
            "target_ends": ends,
        })
    conn.close()
    return out, phrase["phrase"]


def local_wav_path(source_wav: str, data_root: Path) -> Path:
    """Map a stored container path (/data/real/...) onto the local data root."""
    p = source_wav
    for prefix in ("/data/", "/work/data/"):
        if p.startswith(prefix):
            return data_root / p[len(prefix):]
    return Path(p)


# ---- scorer client -----------------------------------------------------

def post_compare(scorer_url: str, wav: Path, models: list[str],
                 mode: str, step_ms: int, keep_ms: int) -> dict:
    boundary = uuid.uuid4().hex
    fields = {"models": ",".join(models), "mode": mode,
              "step_ms": str(step_ms), "keep_ms": str(keep_ms)}
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        parts.append(f"{v}\r\n".encode())
    ctype = mimetypes.guess_type(str(wav))[0] or "audio/wav"
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{wav.name}"\r\n'.encode())
    parts.append(f"Content-Type: {ctype}\r\n\r\n".encode())
    parts.append(wav.read_bytes())
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        f"{scorer_url.rstrip('/')}/compare", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


# ---- detection accounting (mirrors the app's Test tab) -----------------

def account(times_ms: list[float], scores: list[float], target_ends: list[float],
            threshold: float, pre_s: float, post_s: float, fp_gap_s: float):
    times = [t / 1000.0 for t in times_ms]

    hits = 0
    peak_at_target: list[float] = []
    for se in target_ends:
        win = [s for t, s in zip(times, scores) if se - pre_s <= t <= se + post_s]
        pk = max(win) if win else 0.0
        peak_at_target.append(pk)
        if pk >= threshold:
            hits += 1

    false_fires = 0
    i = 0
    n = len(scores)
    while i < n:
        if scores[i] >= threshold:
            j = i
            while j + 1 < n and scores[j + 1] >= threshold:
                j += 1
            seg = list(range(i, j + 1))
            pk_idx = max(seg, key=lambda k: scores[k])
            pk_t = times[pk_idx]
            if not any(abs(pk_t - se) < fp_gap_s for se in target_ends):
                false_fires += 1
            i = j + 1
        else:
            i += 1

    return {
        "targets": len(target_ends),
        "hits": hits,
        "false_fires": false_fires,
        "peak_at_target": peak_at_target,
    }


# ---- reporting ---------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", default="all_set", help="wake-word project slug")
    ap.add_argument("--models", default="",
                    help="comma-separated model dir names; empty = all available")
    ap.add_argument("--scorer-url", default="http://localhost:8780")
    ap.add_argument("--db", default="data/trainer.db")
    ap.add_argument("--data-root", default="data",
                    help="local dir the stored /data/... wav paths map onto")
    ap.add_argument("--id-prefix", default="test_",
                    help="only replay recordings whose id starts with this "
                         "(test_ keeps evaluation off the training pool)")
    ap.add_argument("--mode", default="full", choices=["full", "reset"])
    ap.add_argument("--step-ms", type=int, default=20)
    ap.add_argument("--keep-ms", type=int, default=700)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--pre-s", type=float, default=0.6)
    ap.add_argument("--post-s", type=float, default=0.5)
    ap.add_argument("--fp-gap-s", type=float, default=0.7)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    db = Path(args.db)
    data_root = Path(args.data_root)
    if not db.exists():
        sys.exit(f"db not found: {db}")

    recs, phrase = load_test_recordings(db, args.slug, args.id_prefix)
    if not recs:
        sys.exit(f"no '{args.id_prefix}' recordings for slug {args.slug!r}")

    requested = [m.strip() for m in args.models.split(",") if m.strip()]

    # Per-model running totals.
    totals: dict[str, dict] = {}
    per_rec: list[dict] = []
    seen_models: list[str] = []

    for rec in recs:
        wav = local_wav_path(rec["source_wav"], data_root)
        if not wav.exists():
            print(f"! missing wav, skipping {rec['id']}: {wav}", file=sys.stderr)
            continue
        try:
            resp = post_compare(args.scorer_url, wav, requested,
                                 args.mode, args.step_ms, args.keep_ms)
        except Exception as e:  # noqa: BLE001 - surface any transport error per take
            print(f"! scorer error on {rec['id']}: {e}", file=sys.stderr)
            continue
        if resp.get("errors"):
            for name, msg in resp["errors"].items():
                print(f"! model {name}: {msg}", file=sys.stderr)

        times_ms = resp["times_ms"]
        rec_row = {"id": rec["id"], "targets": len(rec["target_ends"]), "models": {}}
        for name, curve in resp["results"].items():
            if name not in seen_models:
                seen_models.append(name)
                totals[name] = {"targets": 0, "hits": 0, "false_fires": 0, "peaks": []}
            a = account(times_ms, curve["scores"], rec["target_ends"],
                        args.threshold, args.pre_s, args.post_s, args.fp_gap_s)
            totals[name]["targets"] += a["targets"]
            totals[name]["hits"] += a["hits"]
            totals[name]["false_fires"] += a["false_fires"]
            totals[name]["peaks"].extend(a["peak_at_target"])
            rec_row["models"][name] = {
                "hits": a["hits"], "false_fires": a["false_fires"],
                "targets": a["targets"],
            }
        per_rec.append(rec_row)

    if args.json:
        print(json.dumps({
            "slug": args.slug, "phrase": phrase, "mode": args.mode,
            "threshold": args.threshold, "step_ms": args.step_ms,
            "per_recording": per_rec, "totals": totals,
        }, indent=2))
        return

    print(f"\nComparison: slug={args.slug!r} phrase={phrase!r}  mode={args.mode} "
          f"step={args.step_ms}ms thr={args.threshold}")
    print(f"Takes: {len(per_rec)}   (id prefix {args.id_prefix!r})\n")

    name_w = max([len(n) for n in seen_models] + [10])
    header = f"{'model'.ljust(name_w)}  recall        false-fires   mean-peak"
    print(header)
    print("-" * len(header))
    for name in seen_models:
        t = totals[name]
        rec_pct = (100.0 * t["hits"] / t["targets"]) if t["targets"] else 0.0
        mean_peak = (sum(t["peaks"]) / len(t["peaks"])) if t["peaks"] else 0.0
        print(f"{name.ljust(name_w)}  {t['hits']:3d}/{t['targets']:<3d} "
              f"{rec_pct:5.1f}%   {t['false_fires']:<11d}   {mean_peak:.3f}")
    print()


if __name__ == "__main__":
    main()
