#!/usr/bin/env python3
"""Compare hard, soft, and cross-fitted slide TIL estimators."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="NAME=OOF_OUTPUT_DIR; repeat for each experiment",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def correlation(gt, predicted, rank=False):
    if len(gt) < 2 or np.ptp(gt) == 0.0 or np.ptp(predicted) == 0.0:
        return np.nan
    if rank:
        gt = pd.Series(gt).rank(method="average").to_numpy()
        predicted = pd.Series(predicted).rank(method="average").to_numpy()
    return float(np.corrcoef(gt, predicted)[0, 1])


def load_estimators(path):
    slides = pd.read_csv(path / "slide_til_score_summary.csv")
    slides = slides[slides["row_type"] == "case"].copy()
    calibration = pd.read_csv(path / "slide_til_calibration_summary.csv")
    merged = slides.merge(
        calibration[
            ["case_id", "hard_calibrated", "soft_calibrated"]
        ],
        on="case_id",
        validate="one_to_one",
    )
    return merged.rename(
        columns={
            "pred_til_score": "hard_raw",
            "soft_pred_til_score": "soft_raw",
        }
    )


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    summary_rows = []
    bootstrap_rows = []
    for value in args.experiment:
        if "=" not in value:
            raise ValueError("Expected NAME=OOF_OUTPUT_DIR: {0}".format(value))
        name, raw_path = value.split("=", 1)
        frame = load_estimators(Path(raw_path))
        gt = frame["gt_til_score"].to_numpy(dtype=np.float64)
        estimators = (
            "hard_raw",
            "soft_raw",
            "hard_calibrated",
            "soft_calibrated",
        )
        errors = {}
        for estimator in estimators:
            predicted = frame[estimator].to_numpy(dtype=np.float64)
            valid = np.isfinite(gt) & np.isfinite(predicted)
            error = np.abs(gt[valid] - predicted[valid])
            errors[estimator] = error
            summary_rows.append(
                {
                    "experiment": name,
                    "estimator": estimator,
                    "n_cases": int(valid.sum()),
                    "mae": float(error.mean()),
                    "median_ae": float(np.median(error)),
                    "pearson_r": correlation(gt[valid], predicted[valid]),
                    "spearman_r": correlation(
                        gt[valid], predicted[valid], rank=True
                    ),
                }
            )
        n_cases = len(gt)
        samples = rng.randint(0, n_cases, size=(args.bootstrap, n_cases))
        baseline = errors["hard_raw"]
        for estimator in estimators[1:]:
            delta = (
                errors[estimator][samples].mean(axis=1)
                - baseline[samples].mean(axis=1)
            )
            bootstrap_rows.append(
                {
                    "experiment": name,
                    "baseline_estimator": "hard_raw",
                    "estimator": estimator,
                    "metric": "mae",
                    "direction": "lower_is_better",
                    "delta_median": float(np.percentile(delta, 50.0)),
                    "delta_ci_low": float(np.percentile(delta, 2.5)),
                    "delta_ci_high": float(np.percentile(delta, 97.5)),
                    "bootstrap_repetitions": args.bootstrap,
                    "seed": args.seed,
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        output / "til_estimator_comparison.csv",
        index=False,
        float_format="%.8f",
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        output / "til_estimator_bootstrap.csv",
        index=False,
        float_format="%.8f",
    )
    print("TIL estimator comparison complete -> {0}".format(output))


if __name__ == "__main__":
    main()
