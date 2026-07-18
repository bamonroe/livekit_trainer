import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_config as gc  # noqa: E402


def make_args(**overrides) -> argparse.Namespace:
    base = dict(
        manifest=None,
        slug="demo_word",
        phrase="demo word",
        negative=[],
        real_samples_dir="./data/train",
        positive_per_batch=None,
        adversarial_negative_per_batch=50,
        acav_per_batch=1024,
        background_noise_per_batch=50,
        personal=False,
        n_samples=None,
        n_samples_val=None,
        steps=50_000,
        target_fp_per_hour=0.2,
        model_size="medium",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class GenerateConfigTest(unittest.TestCase):
    def test_standard_defaults(self):
        config = gc.config_from_args(make_args())
        self.assertEqual(20_000, config["n_samples"])
        self.assertEqual(4_000, config["n_samples_val"])
        # No batch block unless positives are overweighted.
        self.assertIsNone(config["batch_n_per_class"])
        self.assertIsNone(config["header_comment"])

    def test_personal_preset_shrinks_pool_and_overweights_positives(self):
        config = gc.config_from_args(make_args(personal=True))
        self.assertEqual(3_000, config["n_samples"])
        self.assertEqual(600, config["n_samples_val"])
        self.assertEqual(100, config["batch_n_per_class"]["positive"])
        self.assertIsNotNone(config["header_comment"])
        yaml = gc.render_yaml(config)
        self.assertTrue(yaml.startswith("# personal preset"))
        self.assertIn("n_samples: 3000", yaml)

    def test_explicit_flags_override_personal_preset(self):
        config = gc.config_from_args(
            make_args(personal=True, n_samples=8_000, positive_per_batch=70)
        )
        self.assertEqual(8_000, config["n_samples"])
        self.assertEqual(600, config["n_samples_val"])  # still preset
        self.assertEqual(70, config["batch_n_per_class"]["positive"])


if __name__ == "__main__":
    unittest.main()
