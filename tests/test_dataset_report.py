import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DatasetReportTest(unittest.TestCase):
    def test_reports_categories_labels_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "trainer.db"
            build_db(
                db,
                with_capture=True,
                recordings=[
                    ("rec_a", "Pixel 8a", "builtin_mic", "sess-1"),
                    ("rec_b", "SM-P620", "usb_device", "sess-2"),
                ],
                slices=[
                    ("rec_a", "positive", "positive"),
                    ("rec_a", "negative", "negative"),
                    ("rec_a", "negative", "hard_negative"),
                    ("rec_b", "background", "background"),
                    # A deleted slice must not be counted anywhere.
                    ("rec_b", "positive", "positive", "deleted"),
                ],
            )
            report = json.loads(run(db, "--json"))

        word = report["wake_words"][0]
        self.assertEqual({"positive": 1, "negative": 2, "background": 1}, word["categories"])
        self.assertEqual(
            {"positive": 1, "negative": 1, "hard_negative": 1, "background": 1},
            word["labels"],
        )
        self.assertEqual(2, word["recordings"])
        device = word["provenance"]["device"]
        self.assertEqual(3, device["Pixel 8a"])
        self.assertEqual(1, device["SM-P620"])
        self.assertEqual(
            {"positive": 1, "negative": 2, "background": 1, "total": 4},
            report["totals"],
        )

    def test_tolerates_database_without_capture_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "trainer.db"
            build_db(
                db,
                with_capture=False,
                recordings=[("rec_a", None, None, None)],
                slices=[("rec_a", "positive", "positive")],
            )
            text = run(db)
            self.assertIn("Capture provenance columns are absent", text)
            self.assertIn("positive 1", text)

            report = json.loads(run(db, "--json"))
            self.assertIsNone(report["wake_words"][0]["provenance"])


def run(db: Path, *extra: str) -> str:
    result = subprocess.run(
        [str(ROOT / "scripts" / "dataset_report.py"), "--db", str(db), *extra],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout


def build_db(db: Path, with_capture: bool, recordings: list, slices: list) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE projects (slug TEXT PRIMARY KEY, phrase TEXT, "
        "external_id TEXT, created_at_ms INTEGER)"
    )
    capture_cols = (
        ", capture_device_model TEXT, capture_input_route TEXT, capture_session_id TEXT"
        if with_capture
        else ""
    )
    conn.execute(
        "CREATE TABLE bulk_recordings (id TEXT PRIMARY KEY, project_slug TEXT, "
        f"script TEXT{capture_cols})"
    )
    conn.execute(
        "CREATE TABLE slices (id TEXT PRIMARY KEY, recording_id TEXT, "
        "project_slug TEXT, label TEXT, category TEXT, status TEXT)"
    )
    conn.execute(
        "INSERT INTO projects VALUES ('test_word', 'test word', 'test', 0)"
    )
    for rec_id, device, route, session in recordings:
        if with_capture:
            conn.execute(
                "INSERT INTO bulk_recordings VALUES (?, 'test_word', 'script', ?, ?, ?)",
                (rec_id, device, route, session),
            )
        else:
            conn.execute(
                "INSERT INTO bulk_recordings VALUES (?, 'test_word', 'script')",
                (rec_id,),
            )
    for index, row in enumerate(slices):
        rec_id, category, label = row[0], row[1], row[2]
        status = row[3] if len(row) > 3 else "active"
        conn.execute(
            "INSERT INTO slices VALUES (?, ?, 'test_word', ?, ?, ?)",
            (f"slice_{index}", rec_id, label, category, status),
        )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    unittest.main()
