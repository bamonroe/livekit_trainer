import struct
import tempfile
import unittest
import wave
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import assemble_training_data as ata  # noqa: E402


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(struct.pack("<8h", *([0] * 8)))


def seed(data_root: Path, slug: str, positives=0, negatives=0, background=0) -> None:
    for i in range(positives):
        write_wav(data_root / slug / "positive" / f"p{i}.wav")
    for i in range(negatives):
        write_wav(data_root / slug / "negative" / f"n{i}.wav")
    for i in range(background):
        write_wav(data_root / slug / "background" / f"b{i}.wav")


class AssembleTrainingDataTest(unittest.TestCase):
    def test_pools_other_negatives_and_positives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data" / "real"
            out_root = root / "data" / "train"
            seed(data_root, "hey_tester", positives=3, negatives=2, background=1)
            seed(data_root, "hey_jarvis", positives=4, negatives=5, background=2)

            s = ata.assemble("hey_tester", data_root, out_root)

            # Positives are never borrowed.
            self.assertEqual(3, s["positive"])
            # Negatives = own 2 + jarvis negatives 5 + jarvis positives 4 as neg.
            self.assertEqual(2, s["own_negative"])
            self.assertEqual(5, s["borrowed_negative"])
            self.assertEqual(4, s["borrowed_positive"])
            self.assertEqual(11, s["negative"])
            # Background = own 1 + jarvis 2.
            self.assertEqual(3, s["background"])
            self.assertEqual(["hey_jarvis"], s["other_slugs"])

            dest = out_root / "hey_tester"
            self.assertEqual(3, len(list((dest / "positive").glob("*.wav"))))
            self.assertEqual(11, len(list((dest / "negative").glob("*.wav"))))
            self.assertEqual(3, len(list((dest / "background").glob("*.wav"))))
            # Borrowed clips are symlinks that resolve to a real file.
            borrowed = next((dest / "negative").glob("hey_jarvis__*.wav"))
            self.assertTrue(borrowed.is_symlink())
            self.assertTrue(borrowed.resolve().is_file())

    def test_no_borrow_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data" / "real"
            out_root = root / "data" / "train"
            seed(data_root, "a_word", positives=1, negatives=1, background=1)
            seed(data_root, "b_word", positives=2, negatives=2, background=2)

            s = ata.assemble(
                "a_word", data_root, out_root,
                borrow_positives=False, borrow_background=False,
            )
            self.assertEqual(0, s["borrowed_positive"])
            self.assertEqual(0, s["borrowed_background"])
            self.assertEqual(2, s["borrowed_negative"])
            self.assertEqual(1, s["background"])  # own only

    def test_rerun_is_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data" / "real"
            out_root = root / "data" / "train"
            seed(data_root, "w1", positives=1, negatives=1)
            seed(data_root, "w2", negatives=3)

            first = ata.assemble("w1", data_root, out_root)
            second = ata.assemble("w1", data_root, out_root)
            self.assertEqual(first, second)
            dest = out_root / "w1"
            self.assertEqual(4, len(list((dest / "negative").glob("*.wav"))))


if __name__ == "__main__":
    unittest.main()
