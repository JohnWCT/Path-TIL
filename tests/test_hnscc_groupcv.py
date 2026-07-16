import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from path_til.hnscc import (
    LABELS,
    PREDICTION_COLUMNS,
    balanced_class_weights,
    build_fold_assignments,
    cross_fitted_linear_til_calibration,
    load_hnscc_csv,
    patch_metric_summary,
    slide_til_score_summary,
    validate_oof_predictions,
)


def synthetic_manifest():
    rows = []
    for case_number in range(10):
        case_id = "case_{0:02d}".format(case_number)
        for label in LABELS:
            rows.append(
                {
                    "case_id": case_id,
                    "image_path": "/synthetic/{0}/{1}.png".format(case_id, label),
                    "label": label,
                }
            )
    return pd.DataFrame(rows, columns=("case_id", "image_path", "label"))


def synthetic_assignments(frame):
    assignments, _ = build_fold_assignments(frame, n_folds=5, seed=123)
    return assignments


def synthetic_oof(frame, assignments):
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    test_folds = (
        assignments[assignments["role"] == "test"]
        .set_index("case_id")["fold"]
        .to_dict()
    )
    rows = []
    for row in frame.itertuples(index=False):
        true_index = label_to_index[row.label]
        probabilities = [0.05, 0.05, 0.05]
        probabilities[true_index] = 0.90
        rows.append(
            {
                "patch_id": row.image_path,
                "case_id": row.case_id,
                "image_path": row.image_path,
                "fold": test_folds[row.case_id],
                "split": "test",
                "y_true_idx": true_index,
                "y_true_label": row.label,
                "y_pred_idx": true_index,
                "y_pred_label": row.label,
                "prob_positive": probabilities[0],
                "prob_negative": probabilities[1],
                "prob_other": probabilities[2],
                "confidence": 0.90,
                "correct": True,
            }
        )
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


class FoldAssignmentTests(unittest.TestCase):
    def test_assignments_are_deterministic_grouped_and_complete(self):
        frame = synthetic_manifest()
        first, first_objective = build_fold_assignments(frame, n_folds=5, seed=77)
        second, second_objective = build_fold_assignments(frame, n_folds=5, seed=77)

        assert_frame_equal(first, second)
        self.assertEqual(first_objective, second_objective)
        self.assertEqual(set(first["fold"]), set(range(5)))

        test_counts = first[first["role"] == "test"]["case_id"].value_counts()
        self.assertEqual(set(test_counts.index), set(frame["case_id"]))
        self.assertTrue((test_counts == 1).all())

        for fold in range(5):
            fold_rows = first[first["fold"] == fold]
            self.assertEqual(
                fold_rows["role"].value_counts().to_dict(),
                {"train": 7, "test": 2, "val": 1},
            )
            role_cases = [
                set(fold_rows.loc[fold_rows["role"] == role, "case_id"])
                for role in ("train", "val", "test")
            ]
            self.assertFalse(role_cases[0] & role_cases[1])
            self.assertFalse(role_cases[0] & role_cases[2])
            self.assertFalse(role_cases[1] & role_cases[2])
            self.assertEqual(set.union(*role_cases), set(frame["case_id"]))

    def test_other_seed_still_produces_valid_assignment(self):
        frame = synthetic_manifest()
        assignments, _ = build_fold_assignments(frame, n_folds=5, seed=78)
        for fold in range(5):
            counts = assignments[assignments["fold"] == fold]["role"].value_counts()
            self.assertEqual(counts.to_dict(), {"train": 7, "test": 2, "val": 1})


class CsvValidationTests(unittest.TestCase):
    def assert_manifest_rejected(self, frame):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "manifest.csv")
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_hnscc_csv(path)

    def test_rejects_duplicate_image_path(self):
        frame = synthetic_manifest()
        frame.loc[1, "image_path"] = frame.loc[0, "image_path"]
        self.assert_manifest_rejected(frame)

    def test_rejects_case_path_mismatch(self):
        frame = synthetic_manifest()
        frame.loc[0, "image_path"] = "/synthetic/case_01/wrong.png"
        self.assert_manifest_rejected(frame)

    def test_rejects_case_missing_a_class(self):
        frame = synthetic_manifest()
        frame = frame.drop(
            frame[(frame["case_id"] == "case_00") & (frame["label"] == "other")].index
        )
        self.assert_manifest_rejected(frame)


class MetricTests(unittest.TestCase):
    def test_balanced_class_weight_formula(self):
        labels = ["positive"] * 2 + ["negative"] * 3 + ["other"]
        weights = balanced_class_weights(labels)
        self.assertEqual(set(weights), {0, 1, 2})
        self.assertAlmostEqual(weights[0], 6.0 / (3.0 * 2.0))
        self.assertAlmostEqual(weights[1], 6.0 / (3.0 * 3.0))
        self.assertAlmostEqual(weights[2], 6.0 / (3.0 * 1.0))

    def test_patch_metric_summary_with_all_classes(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        probabilities = np.array(
            [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.05, 0.05, 0.90],
                [0.80, 0.10, 0.10],
                [0.10, 0.80, 0.10],
                [0.10, 0.10, 0.80],
            ]
        )
        summary = patch_metric_summary(y_true, probabilities)
        class_auc = summary[
            (summary["metric"] == "ovr_auc") & (summary["average"] == "none")
        ]
        self.assertEqual(set(class_auc["class"]), set(LABELS))
        self.assertTrue((class_auc["status"] == "ok").all())
        self.assertTrue(np.allclose(class_auc["value"], 1.0))
        aggregate = summary[
            (summary["metric"] == "ovr_auc")
            & (summary["average"].isin(("macro", "weighted")))
        ]
        self.assertTrue((aggregate["status"] == "ok").all())
        self.assertTrue(np.allclose(aggregate["value"], 1.0))

    def test_patch_metric_summary_marks_missing_class_auc_undefined(self):
        y_true = np.array([0, 1, 0, 1])
        probabilities = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
            ]
        )
        summary = patch_metric_summary(y_true, probabilities)
        other = summary[
            (summary["metric"] == "ovr_auc") & (summary["class"] == "other")
        ].iloc[0]
        self.assertEqual(other["status"], "single_class")
        self.assertTrue(np.isnan(other["value"]))
        aggregate = summary[
            (summary["metric"] == "ovr_auc")
            & (summary["average"].isin(("macro", "weighted")))
        ]
        self.assertTrue((aggregate["status"] == "single_class").all())
        self.assertTrue(aggregate["value"].isna().all())


class SlideTilSummaryTests(unittest.TestCase):
    def test_computes_normal_til_scores(self):
        predictions = pd.DataFrame(
            [
                ("case_a", 0, "positive", "positive"),
                ("case_a", 0, "negative", "negative"),
                ("case_a", 0, "negative", "positive"),
                ("case_b", 1, "positive", "positive"),
                ("case_b", 1, "positive", "negative"),
                ("case_b", 1, "negative", "negative"),
            ],
            columns=("case_id", "fold", "y_true_label", "y_pred_label"),
        )
        summary = slide_til_score_summary(predictions)
        case_a = summary[summary["case_id"] == "case_a"].iloc[0]
        self.assertAlmostEqual(case_a["gt_til_score"], 1.0 / 3.0)
        self.assertAlmostEqual(case_a["pred_til_score"], 2.0 / 3.0)
        overall = summary[summary["row_type"] == "overall"].iloc[0]
        self.assertEqual(overall["n_valid_slides"], 2)
        self.assertEqual(overall["status"], "ok")

    def test_marks_zero_predicted_denominator(self):
        predictions = pd.DataFrame(
            [
                ("case_a", 0, "positive", "other"),
                ("case_a", 0, "negative", "other"),
            ],
            columns=("case_id", "fold", "y_true_label", "y_pred_label"),
        )
        summary = slide_til_score_summary(predictions)
        case_row = summary[summary["row_type"] == "case"].iloc[0]
        overall = summary[summary["row_type"] == "overall"].iloc[0]
        self.assertTrue(np.isnan(case_row["pred_til_score"]))
        self.assertEqual(case_row["status"], "zero_denominator")
        self.assertEqual(overall["status"], "no_valid_slides")

    def test_marks_constant_correlation_input(self):
        predictions = pd.DataFrame(
            [
                ("case_a", 0, "positive", "positive"),
                ("case_a", 0, "negative", "negative"),
                ("case_b", 1, "positive", "positive"),
                ("case_b", 1, "negative", "negative"),
            ],
            columns=("case_id", "fold", "y_true_label", "y_pred_label"),
        )
        overall = slide_til_score_summary(predictions)
        overall = overall[overall["row_type"] == "overall"].iloc[0]
        self.assertEqual(overall["status"], "constant_input")
        self.assertTrue(np.isnan(overall["pearson_r"]))
        self.assertTrue(np.isnan(overall["spearman_r"]))

    def test_computes_probability_weighted_til_score(self):
        predictions = pd.DataFrame(
            [
                ("case_a", 0, "positive", "positive", 0.8, 0.1),
                ("case_a", 0, "negative", "other", 0.2, 0.4),
            ],
            columns=(
                "case_id",
                "fold",
                "y_true_label",
                "y_pred_label",
                "prob_positive",
                "prob_negative",
            ),
        )
        summary = slide_til_score_summary(predictions)
        case_row = summary[summary["row_type"] == "case"].iloc[0]
        self.assertAlmostEqual(case_row["soft_pred_til_score"], 1.0 / 1.5)
        self.assertAlmostEqual(case_row["soft_abs_error"], 1.0 / 6.0)

    def test_cross_fitted_linear_calibration(self):
        cases = pd.DataFrame(
            {
                "row_type": ["case"] * 5,
                "case_id": ["a", "b", "c", "d", "e"],
                "gt_til_score": [0.1, 0.2, 0.3, 0.4, 0.5],
                "pred_til_score": [0.2, 0.4, 0.6, 0.8, 1.0],
                "soft_pred_til_score": [0.15, 0.25, 0.35, 0.45, 0.55],
            }
        )
        calibrated = cross_fitted_linear_til_calibration(cases)
        self.assertEqual(len(calibrated), 5)
        self.assertTrue((calibrated["hard_status"] == "ok").all())
        self.assertTrue((calibrated["soft_status"] == "ok").all())
        self.assertTrue(
            np.allclose(
                calibrated["hard_calibrated"],
                calibrated["gt_til_score"],
            )
        )


class OofValidationTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_manifest()
        self.assignments = synthetic_assignments(self.frame)
        self.predictions = synthetic_oof(self.frame, self.assignments)

    def test_accepts_complete_synthetic_oof(self):
        validated = validate_oof_predictions(
            self.frame, self.assignments, self.predictions
        )
        self.assertEqual(len(validated), len(self.frame))
        self.assertEqual(set(validated["image_path"]), set(self.frame["image_path"]))
        self.assertTrue(validated["correct"].all())

    def test_rejects_duplicate_patch_id(self):
        predictions = self.predictions.copy()
        predictions.loc[1, "patch_id"] = predictions.loc[0, "patch_id"]
        with self.assertRaises(ValueError):
            validate_oof_predictions(self.frame, self.assignments, predictions)

    def test_rejects_probability_sum_error(self):
        predictions = self.predictions.copy()
        predictions.loc[0, "prob_positive"] = 0.80
        with self.assertRaises(ValueError):
            validate_oof_predictions(self.frame, self.assignments, predictions)

    def test_rejects_case_fold_mismatch(self):
        predictions = self.predictions.copy()
        predictions.loc[0, "fold"] = (int(predictions.loc[0, "fold"]) + 1) % 5
        with self.assertRaises(ValueError):
            validate_oof_predictions(self.frame, self.assignments, predictions)


if __name__ == "__main__":
    unittest.main()
