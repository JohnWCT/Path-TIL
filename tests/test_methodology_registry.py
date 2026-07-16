#!/usr/bin/env python3
"""Unit tests for path_til.experiment_registry."""

import tempfile
import unittest
from pathlib import Path

from path_til.experiment_registry import (
    CANDIDATE_REFERENCE,
    keep_or_drop,
    load_method_config,
)


class RegistryTests(unittest.TestCase):
    def test_load_method_config_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "method_demo.yaml"
            path.write_text(
                "loss: focal_gamma2\naug: heavy\n", encoding="utf-8"
            )
            config = load_method_config(path)
            self.assertEqual(config["loss"], "focal_gamma2")
            self.assertEqual(config["hne_norm"], "off")
            self.assertEqual(config["name"], "method_demo")

    def test_keep_or_drop_success_criteria(self):
        better = keep_or_drop(
            {
                "positive_auc": CANDIDATE_REFERENCE["positive_auc"] + 0.01,
                "hard_til_mae": CANDIDATE_REFERENCE["hard_til_mae"] - 0.01,
                "macro_ovr_auc": CANDIDATE_REFERENCE["macro_ovr_auc"],
                "weighted_ovr_auc": CANDIDATE_REFERENCE["weighted_ovr_auc"],
            }
        )
        self.assertEqual(better["decision"], "keep")
        worse = keep_or_drop(
            {
                "positive_auc": 0.80,
                "hard_til_mae": 0.20,
                "macro_ovr_auc": 0.80,
                "weighted_ovr_auc": 0.80,
            }
        )
        self.assertEqual(worse["decision"], "drop")
        self.assertIn("positive_auc_not_improved", worse["reasons"])


if __name__ == "__main__":
    unittest.main()
