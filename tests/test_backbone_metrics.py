import tempfile
import unittest
from pathlib import Path

from path_til.backbone_metrics import (
    has_all_full5_folds,
    required_full5_folds,
    summarize_smoke_folds,
)
from path_til.class_weighting import scale_class_weight


class ClassWeightingTests(unittest.TestCase):
    def test_scale_positive_class_weight(self):
        weights = {0: 3.0, 1: 1.0, 2: 0.8}
        scaled = scale_class_weight(weights, positive_class_index=0, positive_scale=0.75)
        self.assertEqual(scaled[0], 2.25)
        self.assertEqual(scaled[1], 1.0)
        self.assertEqual(scaled[2], 0.8)

    def test_original_class_weight_not_modified(self):
        weights = {0: 3.0, 1: 1.0, 2: 0.8}
        scaled = scale_class_weight(weights, positive_class_index=0, positive_scale=0.75)
        self.assertEqual(weights[0], 3.0)
        self.assertEqual(scaled[0], 2.25)


class BackboneMetricsTests(unittest.TestCase):
    def test_summarize_smoke_folds_computes_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for fold, auc, prc in ((0, 0.86, 0.50), (1, 0.94, 0.60)):
                fold_dir = root / "fold{0:02d}".format(fold)
                fold_dir.mkdir()
                (fold_dir / "fold_metrics.json").write_text(
                    (
                        '{{"selected_metrics": {{"test": {{'
                        '"positive_auc": {0}, "positive_prc": {1}, '
                        '"macro_ovr_auc": 0.90, "weighted_ovr_auc": 0.91, '
                        '"negative_auc": 0.88, "other_auc": 0.89}}}}}}'
                    ).format(auc, prc),
                    encoding="utf-8",
                )
            summary = summarize_smoke_folds(root, folds=[0, 1], experiment_name="demo")
            self.assertAlmostEqual(summary["positive_auc"], 0.90)
            self.assertAlmostEqual(summary["positive_prc"], 0.55)
            self.assertAlmostEqual(summary["fold_auc_gap"], 0.08)
            self.assertAlmostEqual(summary["fold_prc_gap"], 0.10)

    def test_full5_fold_set(self):
        self.assertEqual(required_full5_folds(), {0, 1, 2, 3, 4})
        self.assertTrue(has_all_full5_folds([0, 1, 2, 3, 4]))
        self.assertFalse(has_all_full5_folds([0, 1]))


if __name__ == "__main__":
    unittest.main()
