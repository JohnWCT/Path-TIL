#!/usr/bin/env python3
"""Unit tests for path_til.til_threshold."""

import unittest

import numpy as np
import pandas as pd

from path_til.til_threshold import (
    apply_threshold_to_frame,
    hard_til_from_labels,
    predict_labels_with_threshold,
    tune_positive_threshold,
)


class TilThresholdTests(unittest.TestCase):
    def test_hard_til_score(self):
        self.assertAlmostEqual(
            hard_til_from_labels(["positive", "negative", "other"]),
            0.5,
        )
        self.assertTrue(np.isnan(hard_til_from_labels(["other", "other"])))

    def test_predict_labels_with_threshold(self):
        probabilities = np.array(
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
            ]
        )
        high = predict_labels_with_threshold(probabilities, 0.5)
        low = predict_labels_with_threshold(probabilities, 0.8)
        np.testing.assert_array_equal(high, ["positive", "negative"])
        np.testing.assert_array_equal(low, ["negative", "negative"])

    def test_tune_on_validation_only(self):
        frame = pd.DataFrame(
            {
                "y_true_label": [
                    "positive",
                    "negative",
                    "positive",
                    "negative",
                ],
                "prob_positive": [0.9, 0.2, 0.55, 0.4],
                "prob_negative": [0.05, 0.7, 0.35, 0.5],
                "prob_other": [0.05, 0.1, 0.1, 0.1],
            }
        )
        threshold, error, grid = tune_positive_threshold(frame)
        self.assertTrue(0.1 <= threshold <= 0.9)
        self.assertTrue(np.isfinite(error))
        self.assertFalse(grid.empty)
        applied = apply_threshold_to_frame(frame, threshold)
        self.assertIn("positive_threshold", applied.columns)
        self.assertEqual(len(applied), len(frame))


if __name__ == "__main__":
    unittest.main()
