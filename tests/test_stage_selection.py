#!/usr/bin/env python3
"""Unit tests for path_til.stage_selection."""

import unittest

from path_til.stage_selection import select_stage


class StageSelectionTests(unittest.TestCase):
    def setUp(self):
        self.stage_metrics = {
            0: {
                "val": {
                    "positive_auc": 0.70,
                    "macro_ovr_auc": 0.72,
                }
            },
            1: {
                "val": {
                    "positive_auc": 0.84,
                    "macro_ovr_auc": 0.88,
                }
            },
            2: {
                "val": {
                    "positive_auc": 0.86,
                    "macro_ovr_auc": 0.85,
                }
            },
        }
        self.keras_auc = {0: 0.71, 1: 0.90, 2: 0.89}

    def test_fixed_policies(self):
        self.assertEqual(
            select_stage("fixed_stage1", self.stage_metrics), 1
        )
        self.assertEqual(
            select_stage("fixed_stage2", self.stage_metrics), 2
        )

    def test_validation_multiclass_auc(self):
        self.assertEqual(
            select_stage(
                "validation_multiclass_auc",
                self.stage_metrics,
                self.keras_auc,
            ),
            1,
        )

    def test_positive_and_composite(self):
        self.assertEqual(
            select_stage(
                "validation_positive_auc", self.stage_metrics
            ),
            2,
        )
        self.assertEqual(
            select_stage(
                "validation_macro_ovr_auc", self.stage_metrics
            ),
            1,
        )
        # 0.7*0.86 + 0.3*0.85 = 0.857; stage1 composite = 0.852
        self.assertEqual(
            select_stage(
                "composite_positive_macro", self.stage_metrics
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
