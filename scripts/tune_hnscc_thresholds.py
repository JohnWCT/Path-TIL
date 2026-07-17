#!/usr/bin/env python3
"""Tune positive thresholds on validation and apply to held-out test only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.hnscc import (  # noqa: E402
    patch_metric_summary,
    slide_til_score_summary,
    load_hnscc_csv,
    validate_fold_assignments,
    ASSIGNMENT_COLUMNS,
    LABELS,
)
from path_til.til_threshold import (  # noqa: E402
    apply_threshold_to_frame,
    tune_positive_threshold,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--fold-csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_assignments(path):
    assignments = pd.read_csv(path)
    if list(assignments.columns) != list(ASSIGNMENT_COLUMNS):
        raise ValueError("Unexpected fold CSV columns")
    assignments["fold"] = assignments["fold"].astype(int)
    assignments["case_id"] = assignments["case_id"].astype(str)
    assignments["role"] = assignments["role"].astype(str)
    return assignments


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_hnscc_csv(args.csv, expected_cases=10)
    assignments = load_assignments(args.fold_csv)
    validate_fold_assignments(manifest, assignments, n_folds=5)

    tuned_rows = []
    test_frames = []
    for fold in range(5):
        fold_dir = pred_dir / "fold{0:02d}".format(fold)
        validation = pd.read_csv(fold_dir / "val_predictions.csv")
        test = pd.read_csv(fold_dir / "test_predictions.csv")
        threshold, val_error, grid = tune_positive_threshold(validation)
        grid.to_csv(
            output / "fold{0:02d}_threshold_grid.csv".format(fold),
            index=False,
            float_format="%.8f",
        )
        applied = apply_threshold_to_frame(test, threshold)
        applied.to_csv(
            output / "fold{0:02d}_test_threshold_predictions.csv".format(fold),
            index=False,
            float_format="%.8f",
        )
        test_frames.append(applied)
        tuned_rows.append(
            {
                "fold": fold,
                "threshold": threshold,
                "validation_abs_error": val_error,
                "n_validation": int(len(validation)),
                "n_test": int(len(test)),
            }
        )
        print(
            "Fold {0}: threshold={1:.4f} val_abs_error={2:.6f}".format(
                fold, threshold, val_error
            )
        )

    predictions = pd.concat(test_frames, ignore_index=True)
    # Thresholded hard labels intentionally differ from probability argmax, so
    # skip validate_oof_predictions' argmax consistency check. Still enforce
    # coverage and fold uniqueness manually.
    if predictions["patch_id"].duplicated().any():
        raise ValueError("Duplicate patch_id in thresholded OOF predictions")
    if set(predictions["case_id"].astype(str)) != set(
        manifest["case_id"].astype(str)
    ):
        raise ValueError("Thresholded OOF cases do not match the manifest")
    if len(predictions) != len(manifest):
        raise ValueError(
            "Expected {0} patches, found {1}".format(
                len(manifest), len(predictions)
            )
        )
    probabilities = predictions[
        ["prob_{0}".format(label) for label in LABELS]
    ].to_numpy()
    patch_summary = patch_metric_summary(
        predictions["y_true_idx"].to_numpy(), probabilities
    )
    # Rebuild hard labels from thresholded predictions for TIL summary.
    slide_summary = slide_til_score_summary(predictions)
    predictions.to_csv(
        output / "oof_threshold_predictions.csv",
        index=False,
        float_format="%.8f",
    )
    patch_summary.to_csv(
        output / "patch_metric_summary.csv", index=False, float_format="%.8f"
    )
    slide_summary.to_csv(
        output / "slide_til_score_summary.csv",
        index=False,
        float_format="%.8f",
    )
    pd.DataFrame(tuned_rows).to_csv(
        output / "fold_thresholds.csv", index=False, float_format="%.8f"
    )
    overall = slide_summary[slide_summary["row_type"] == "overall"].iloc[0]
    patch_map = {}
    for _, row in patch_summary.iterrows():
        key_parts = [str(row["metric"])]
        if row["class"]:
            key_parts.append(str(row["class"]))
        if row["average"]:
            key_parts.append(str(row["average"]))
        patch_map["_".join(key_parts)] = row["value"]
    summary = {
        "n_patches": int(len(predictions)),
        "n_cases": int(predictions["case_id"].nunique()),
        "positive_auc": float(
            patch_map["positive_vs_rest_auc_positive_binary"]
        ),
        "positive_prc": float(
            patch_map["positive_vs_rest_average_precision_positive_binary"]
        ),
        "macro_ovr_auc": float(patch_map["ovr_auc_macro"]),
        "weighted_ovr_auc": float(patch_map["ovr_auc_weighted"]),
        "macro_f1": float(patch_map["f1_macro"]),
        "hard_til_mae": None
        if pd.isna(overall["mae"])
        else float(overall["mae"]),
        "soft_til_mae": None
        if pd.isna(overall["soft_mae"])
        else float(overall["soft_mae"]),
        "pearson": None
        if pd.isna(overall["pearson_r"])
        else float(overall["pearson_r"]),
        "spearman": None
        if pd.isna(overall["spearman_r"])
        else float(overall["spearman_r"]),
        "fold_thresholds": tuned_rows,
        "note": "Thresholds tuned on validation only; applied to held-out test",
    }
    # Patch metrics from probabilities remain ranking metrics; hard labels changed.
    accuracy = float(predictions["correct"].astype(bool).mean())
    summary["accuracy"] = accuracy
    with (output / "threshold_til_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        "Threshold TIL complete: hard_til_mae={0} accuracy={1} -> {2}".format(
            summary["hard_til_mae"], summary["accuracy"], output
        )
    )


if __name__ == "__main__":
    main()
