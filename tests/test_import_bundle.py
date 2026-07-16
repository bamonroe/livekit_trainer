import json
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ImportBundleTest(unittest.TestCase):
    def test_imports_sample_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = write_bundle(tmp_path)
            data_root = tmp_path / "data" / "real"
            subprocess.run(
                [
                    str(ROOT / "scripts" / "import_bundle.py"),
                    str(bundle),
                    "--data-root",
                    str(data_root),
                ],
                check=True,
                cwd=ROOT,
            )
            imported = data_root / "test_word" / "positive" / "clip_1_test-word.wav"
            self.assertTrue(imported.is_file())
            self.assertTrue((data_root / "test_word" / "metadata.jsonl").is_file())

    def test_skip_existing_does_not_duplicate_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = write_bundle(tmp_path)
            data_root = tmp_path / "data" / "real"
            command = [
                str(ROOT / "scripts" / "import_bundle.py"),
                str(bundle),
                "--data-root",
                str(data_root),
                "--skip-existing",
            ]

            subprocess.run(command, check=True, cwd=ROOT)
            subprocess.run(command, check=True, cwd=ROOT)

            metadata = data_root / "test_word" / "metadata.jsonl"
            self.assertEqual(1, len(metadata.read_text().splitlines()))


def write_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    audio = bundle / "audio"
    audio.mkdir(parents=True)
    write_wav(audio / "clip.wav")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "wake_word": {
                    "id": "test",
                    "slug": "test_word",
                    "phrase": "test word",
                    "target_phrases": ["test word"],
                    "negative_phrases": [],
                },
                "clips": [
                    {
                        "id": "clip_1",
                        "file": "audio/clip.wav",
                        "label": "positive",
                        "prompt": "Say test word",
                        "spoken_phrase": "test word",
                        "recorded_at": "2026-07-15T00:00:00Z",
                        "duration_ms": 100,
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "encoding": "pcm_s16le",
                        "session_id": "test",
                        "notes": "",
                    },
                ],
            },
        ),
    )
    return bundle


def write_wav(path: Path):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x10" * 1600)


if __name__ == "__main__":
    unittest.main()
