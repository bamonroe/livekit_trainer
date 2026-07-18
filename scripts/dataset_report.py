#!/usr/bin/env python3
"""Summarize the collected training data recorded in the sync server database.

Reports, per wake word and pooled across all of them, how many active slices
have been cut, broken down by training category and by the richer slice label
(so hard negatives are visible separately from plain negatives). When the
per-recording capture provenance is present (device, input route, session), it
also breaks the slices down by source so the dataset can be inspected before an
expensive training run.

Reads the SQLite database the sync server writes (default: data/trainer.db).
Tolerates older databases that predate the capture columns.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


CATEGORIES = ("positive", "negative", "background")

# Capture columns joined from bulk_recordings, labeled for the report grouping.
PROVENANCE = (
    ("device", "capture_device_model"),
    ("input route", "capture_input_route"),
    ("session", "capture_session_id"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/trainer.db"),
        help="Sync server SQLite database, default: data/trainer.db",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    if not args.db.is_file():
        raise SystemExit(f"database not found: {args.db}")

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    has_capture = column_exists(conn, "bulk_recordings", "capture_session_id")

    report = build_report(conn, has_capture)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report, str(args.db), has_capture))
    return 0


def build_report(conn: sqlite3.Connection, has_capture: bool) -> dict[str, Any]:
    projects: list[dict[str, Any]] = []
    for row in conn.execute(
        "SELECT slug, phrase FROM projects ORDER BY slug"
    ).fetchall():
        slug = row["slug"]
        projects.append(
            {
                "slug": slug,
                "phrase": row["phrase"],
                "categories": category_counts(conn, slug),
                "labels": label_counts(conn, slug),
                "recordings": recording_count(conn, slug),
                "provenance": provenance_counts(conn, slug) if has_capture else None,
            }
        )

    totals = {category: 0 for category in CATEGORIES}
    for project in projects:
        for category in CATEGORIES:
            totals[category] += project["categories"].get(category, 0)
    totals["total"] = sum(totals[category] for category in CATEGORIES)

    return {"wake_words": projects, "totals": totals}


def category_counts(conn: sqlite3.Connection, slug: str) -> dict[str, int]:
    rows = conn.execute(
        "SELECT category, COUNT(*) AS n FROM slices "
        "WHERE project_slug = ? AND status = 'active' GROUP BY category",
        (slug,),
    ).fetchall()
    return {row["category"]: row["n"] for row in rows}


def label_counts(conn: sqlite3.Connection, slug: str) -> dict[str, int]:
    rows = conn.execute(
        "SELECT label, COUNT(*) AS n FROM slices "
        "WHERE project_slug = ? AND status = 'active' GROUP BY label",
        (slug,),
    ).fetchall()
    return {row["label"]: row["n"] for row in rows}


def recording_count(conn: sqlite3.Connection, slug: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM bulk_recordings WHERE project_slug = ?", (slug,)
    ).fetchone()[0]


def provenance_counts(conn: sqlite3.Connection, slug: str) -> dict[str, dict[str, int]]:
    """Active slice counts grouped by each capture dimension of their source."""
    result: dict[str, dict[str, int]] = {}
    for name, column in PROVENANCE:
        rows = conn.execute(
            f"SELECT COALESCE(NULLIF(r.{column}, ''), '(unknown)') AS key, "
            "COUNT(*) AS n FROM slices s "
            "JOIN bulk_recordings r ON r.id = s.recording_id "
            "WHERE s.project_slug = ? AND s.status = 'active' "
            "GROUP BY key ORDER BY n DESC, key",
            (slug,),
        ).fetchall()
        result[name] = {row["key"]: row["n"] for row in rows}
    return result


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(
        row["name"] == column
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    )


def render(report: dict[str, Any], db_path: str, has_capture: bool) -> str:
    lines = [f"Dataset report — {db_path}", ""]
    wake_words = report["wake_words"]
    if not wake_words:
        lines.append("No wake words recorded yet.")
        return "\n".join(lines)

    for project in wake_words:
        categories = project["categories"]
        active = sum(categories.get(category, 0) for category in CATEGORIES)
        lines.append(f'{project["slug"]}  "{project["phrase"]}"')
        lines.append(
            f"  slices: {active} active   ("
            + " · ".join(f"{c} {categories.get(c, 0)}" for c in CATEGORIES)
            + ")"
        )
        labels = project["labels"]
        if labels:
            lines.append(
                "  labels: "
                + " · ".join(f"{label} {n}" for label, n in sorted(labels.items()))
            )
        lines.append(f'  source recordings: {project["recordings"]}')
        if project["provenance"] is not None:
            for name, counts in project["provenance"].items():
                rendered = " · ".join(f"{key} {n}" for key, n in counts.items())
                lines.append(f"  by {name}: {rendered or '(none)'}")
        lines.append("")

    if not has_capture:
        lines.append(
            "Capture provenance columns are absent; re-sync with the current "
            "server to populate device/route/session breakdowns."
        )
        lines.append("")

    totals = report["totals"]
    count = len(wake_words)
    noun = "wake word" if count == 1 else "wake words"
    lines.append(
        f"Totals across {count} {noun}: {totals['total']} active slices ("
        + " · ".join(f"{c} {totals[c]}" for c in CATEGORIES)
        + ")"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
