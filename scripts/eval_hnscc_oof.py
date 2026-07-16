#!/usr/bin/env python3
"""Validate and evaluate complete HNSCC out-of-fold test predictions."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.hnscc import (  # noqa: E402
    ASSIGNMENT_COLUMNS,
    LABELS,
    PREDICTION_COLUMNS,
    cross_fitted_linear_til_calibration,
    load_hnscc_csv,
    patch_metric_summary,
    slide_til_score_summary,
    validate_fold_assignments,
    validate_oof_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate complete HNSCC fold test predictions and produce "
            "patch-level and slide-level OOF evaluation summaries."
        )
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Training output root containing fold*/test_predictions.csv",
    )
    parser.add_argument("--csv", required=True, help="Original HNSCC patch manifest")
    parser.add_argument(
        "--fold-csv",
        required=True,
        help="Case-level fold,case_id,role CSV used during training",
    )
    parser.add_argument("--output", required=True, help="Evaluation output directory")
    parser.add_argument(
        "--stage",
        choices=("selected", "0", "1", "2"),
        default="selected",
        help="Prediction stage to evaluate (default: selected)",
    )
    return parser.parse_args()


def load_assignments(path):
    assignments = pd.read_csv(path)
    if list(assignments.columns) != list(ASSIGNMENT_COLUMNS):
        raise ValueError(
            "Fold CSV columns must be exactly {0}; found {1}".format(
                ASSIGNMENT_COLUMNS, assignments.columns.tolist()
            )
        )
    if assignments.empty:
        raise ValueError("Fold CSV contains no rows")
    if assignments.isnull().any().any():
        raise ValueError("Fold CSV contains null values")
    fold_values = pd.to_numeric(assignments["fold"], errors="coerce")
    if fold_values.isnull().any() or not np.equal(
        fold_values, np.floor(fold_values)
    ).all():
        raise ValueError("Fold CSV fold values must be integers")
    assignments = assignments.copy()
    assignments["fold"] = fold_values.astype(np.int64)
    assignments["case_id"] = assignments["case_id"].astype(str)
    assignments["role"] = assignments["role"].astype(str)
    for column in ("case_id", "role"):
        if assignments[column].str.strip().eq("").any():
            raise ValueError("Fold CSV {0} contains empty values".format(column))
    return assignments


def discover_prediction_files(pred_dir, n_folds=5, stage="selected"):
    root = Path(pred_dir)
    if not root.is_dir():
        raise FileNotFoundError("Prediction directory does not exist: {0}".format(root))
    filename = (
        "test_predictions.csv"
        if stage == "selected"
        else "stage{0}_test_predictions.csv".format(stage)
    )
    paths = sorted(root.glob("fold*/{0}".format(filename)))
    if not paths:
        raise FileNotFoundError(
            "No fold*/{0} files found under {1}".format(filename, root)
        )
    by_fold = {}
    for path in paths:
        match = re.fullmatch(r"fold(\d+)", path.parent.name)
        if match is None:
            raise ValueError(
                "Prediction parent directory must be named fold<integer>: {0}".format(
                    path.parent
                )
            )
        fold = int(match.group(1))
        if fold in by_fold:
            raise ValueError(
                "Multiple test prediction files represent fold {0}: {1}, {2}".format(
                    fold, by_fold[fold], path
                )
            )
        by_fold[fold] = path
    expected = set(range(n_folds))
    actual = set(by_fold)
    if actual != expected:
        raise ValueError(
            "Prediction files must cover folds 0..{0} exactly; missing={1}, "
            "unknown={2}".format(
                n_folds - 1, sorted(expected - actual), sorted(actual - expected)
            )
        )
    return [(fold, by_fold[fold]) for fold in range(n_folds)]


def load_predictions(pred_dir, n_folds=5, stage="selected"):
    frames = []
    for folder_fold, path in discover_prediction_files(
        pred_dir, n_folds=n_folds, stage=stage
    ):
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError("Prediction file contains no rows: {0}".format(path))
        missing = [column for column in PREDICTION_COLUMNS if column not in frame]
        extra = [column for column in frame if column not in PREDICTION_COLUMNS]
        if missing or extra:
            raise ValueError(
                "Prediction schema error in {0}; missing={1}, extra={2}".format(
                    path, missing, extra
                )
            )
        file_folds = pd.to_numeric(frame["fold"], errors="coerce")
        if file_folds.isnull().any() or set(file_folds.tolist()) != {folder_fold}:
            raise ValueError(
                "Prediction fold column does not match directory fold {0}: {1}".format(
                    folder_fold, path
                )
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def json_number(value):
    return None if pd.isna(value) else float(value)


def build_json_summary(predictions, patch_summary, slide_summary):
    overall = slide_summary.loc[slide_summary["row_type"] == "overall"].iloc[0]
    patch_values = {}
    for _, row in patch_summary.iterrows():
        key_parts = [str(row["metric"])]
        if row["class"]:
            key_parts.append(str(row["class"]))
        if row["average"]:
            key_parts.append(str(row["average"]))
        patch_values["_".join(key_parts)] = {
            "value": json_number(row["value"]),
            "status": row["status"],
        }
    return {
        "n_patches": int(len(predictions)),
        "n_cases": int(predictions["case_id"].nunique()),
        "folds": sorted(int(value) for value in predictions["fold"].unique()),
        "labels": list(LABELS),
        "patch_metrics": patch_values,
        "slide_metrics": {
            "n_valid_slides": int(overall["n_valid_slides"]),
            "mae": json_number(overall["mae"]),
            "median_ae": json_number(overall["median_ae"]),
            "spearman_r": json_number(overall["spearman_r"]),
            "pearson_r": json_number(overall["pearson_r"]),
            "soft_mae": json_number(overall["soft_mae"]),
            "soft_median_ae": json_number(overall["soft_median_ae"]),
            "soft_spearman_r": json_number(overall["soft_spearman_r"]),
            "soft_pearson_r": json_number(overall["soft_pearson_r"]),
            "soft_status": overall["soft_status"],
            "status": overall["status"],
        },
    }


def calibration_summary(calibration):
    result = {}
    gt = calibration["gt_til_score"].to_numpy(dtype=np.float64)
    for prefix in ("hard", "soft"):
        calibrated = calibration[
            "{0}_calibrated".format(prefix)
        ].to_numpy(dtype=np.float64)
        valid = np.isfinite(gt) & np.isfinite(calibrated)
        if not valid.any():
            result[prefix] = {
                "n_cases": 0,
                "mae": None,
                "pearson_r": None,
                "spearman_r": None,
                "status": "no_valid_cases",
            }
            continue
        error = np.abs(gt[valid] - calibrated[valid])
        pearson = None
        spearman = None
        status = "ok"
        if valid.sum() < 2:
            status = "insufficient_pairs"
        elif np.ptp(gt[valid]) == 0.0 or np.ptp(calibrated[valid]) == 0.0:
            status = "constant_input"
        else:
            pearson = float(np.corrcoef(gt[valid], calibrated[valid])[0, 1])
            gt_rank = pd.Series(gt[valid]).rank(method="average").to_numpy()
            pred_rank = pd.Series(calibrated[valid]).rank(
                method="average"
            ).to_numpy()
            spearman = float(np.corrcoef(gt_rank, pred_rank)[0, 1])
        result[prefix] = {
            "n_cases": int(valid.sum()),
            "mae": float(error.mean()),
            "pearson_r": pearson,
            "spearman_r": spearman,
            "status": status,
        }
    return result


def main():
    args = parse_args()
    manifest_path = Path(args.csv)
    fold_path = Path(args.fold_csv)
    if not manifest_path.is_file():
        raise FileNotFoundError("Manifest CSV does not exist: {0}".format(manifest_path))
    if not fold_path.is_file():
        raise FileNotFoundError("Fold CSV does not exist: {0}".format(fold_path))

    manifest = load_hnscc_csv(manifest_path, expected_cases=10)
    assignments = load_assignments(fold_path)
    validate_fold_assignments(manifest, assignments, n_folds=5)
    raw_predictions = load_predictions(
        args.pred_dir, n_folds=5, stage=args.stage
    )
    predictions = validate_oof_predictions(
        manifest, assignments, raw_predictions, n_folds=5
    )

    probabilities = predictions[
        ["prob_{0}".format(label) for label in LABELS]
    ].to_numpy(dtype=np.float64)
    patch_summary = patch_metric_summary(
        predictions["y_true_idx"].to_numpy(dtype=np.int64), probabilities
    )
    slide_summary = slide_til_score_summary(predictions)
    calibration = cross_fitted_linear_til_calibration(slide_summary)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        output / "oof_predictions.csv", index=False, float_format="%.8f"
    )
    patch_summary.to_csv(
        output / "patch_auc_summary.csv", index=False, float_format="%.8f"
    )
    slide_summary.to_csv(
        output / "slide_til_score_summary.csv", index=False, float_format="%.8f"
    )
    calibration.to_csv(
        output / "slide_til_calibration_summary.csv",
        index=False,
        float_format="%.8f",
    )
    json_summary = build_json_summary(
        predictions, patch_summary, slide_summary
    )
    json_summary["stage"] = args.stage
    json_summary["cross_fitted_linear_calibration"] = calibration_summary(
        calibration
    )
    with (output / "eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            json_summary,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(
        "OOF evaluation complete: {0} patches, {1} cases -> {2}".format(
            len(predictions), predictions["case_id"].nunique(), output
        )
    )


if __name__ == "__main__":
    main()
