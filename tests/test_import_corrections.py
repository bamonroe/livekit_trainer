import json
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ImportCorrectionsTest(unittest.TestCase):
    def test_emits_false_positive_and_false_negative_and_skips_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            batch = write_batch(
                tmp_path,
                threshold=0.5,
                corrections=[
                    # positive scored low -> missed -> false_negative -> positive/
                    entry("miss", "positive", 0.20),
                    # negative scored high -> wrongful fire -> false_positive -> negative/
                    entry("fire", "negative", 0.90),
                    # background scored high -> false_positive -> negative/
                    entry("noise", "background", 0.80),
                    # positive scored high -> correct -> skipped
                    entry("good_pos", "positive", 0.95),
                    # negative scored low -> correct -> skipped
                    entry("good_neg", "negative", 0.10),
                ],
            )
            data_root = tmp_path / "data" / "real"
            run(batch, data_root)

            slug_root = data_root / "test_word"
            self.assertTrue((slug_root / "positive" / "miss_test-word.wav").is_file())
            self.assertTrue((slug_root / "negative" / "fire_test-word.wav").is_file())
            self.assertTrue((slug_root / "negative" / "noise_test-word.wav").is_file())

            records = [
                json.loads(line)
                for line in (slug_root / "metadata.jsonl").read_text().splitlines()
            ]
            self.assertEqual(3, len(records))
            labels = sorted(r["original_label"] for r in records)
            self.assertEqual(
                ["false_negative", "false_positive", "false_positive"], labels
            )
            categories = {r["original_label"]: r["livekit_category"] for r in records}
            self.assertEqual("positive", categories["false_negative"])
            self.assertEqual("negative", categories["false_positive"])

    def test_explicit_mistake_overrides_derivation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            override = entry("manual", "negative", 0.10)
            override["mistake"] = "false_positive"
            batch = write_batch(tmp_path, threshold=0.5, corrections=[override])
            data_root = tmp_path / "data" / "real"
            run(batch, data_root)
            self.assertTrue(
                (data_root / "test_word" / "negative" / "manual_test-word.wav").is_file()
            )

    def test_cli_threshold_overrides_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Score 0.60: correct at threshold 0.5, a miss at threshold 0.7.
            batch = write_batch(
                tmp_path, threshold=0.5, corrections=[entry("edge", "positive", 0.60)]
            )
            data_root = tmp_path / "data" / "real"
            run(batch, data_root, "--threshold", "0.7")
            self.assertTrue(
                (data_root / "test_word" / "positive" / "edge_test-word.wav").is_file()
            )

    def test_skip_existing_does_not_duplicate_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            batch = write_batch(
                tmp_path, threshold=0.5, corrections=[entry("miss", "positive", 0.20)]
            )
            data_root = tmp_path / "data" / "real"
            run(batch, data_root, "--skip-existing")
            run(batch, data_root, "--skip-existing")
            metadata = data_root / "test_word" / "metadata.jsonl"
            self.assertEqual(1, len(metadata.read_text().splitlines()))


def entry(clip_id: str, label: str, score: float) -> dict:
    return {
        "id": clip_id,
        "file": f"audio/{clip_id}.wav",
        "label": label,
        "score": score,
        "spoken_phrase": "test word",
    }


def run(batch: Path, data_root: Path, *extra: str) -> None:
    subprocess.run(
        [
            str(ROOT / "scripts" / "import_corrections.py"),
            str(batch),
            "--data-root",
            str(data_root),
            *extra,
        ],
        check=True,
        cwd=ROOT,
    )


def write_batch(tmp_path: Path, threshold: float, corrections: list[dict]) -> Path:
    batch = tmp_path / "batch"
    audio = batch / "audio"
    audio.mkdir(parents=True)
    for correction in corrections:
        write_wav(batch / correction["file"])
    (batch / "corrections.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "wake_word": {"slug": "test_word", "phrase": "test word"},
                "model": {"name": "test_word", "threshold": threshold},
                "corrections": corrections,
            },
        ),
    )
    return batch


def write_wav(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x10" * 1600)


if __name__ == "__main__":
    unittest.main()
