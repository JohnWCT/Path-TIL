#!/usr/bin/env python3
"""Compare HNSCC experiments with paired case-cluster bootstrap intervals."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="NAME=OOF_OUTPUT_DIR; repeat for each experiment",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_experiments(values):
    experiments = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Expected NAME=OOF_OUTPUT_DIR: {0}".format(value))
        name, path = value.split("=", 1)
        if not name or name in experiments:
            raise ValueError("Experiment names must be unique and non-empty")
        experiments[name] = Path(path)
    return experiments


def read_experiment(name, path):
    with (path / "eval_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    predictions = pd.read_csv(path / "oof_predictions.csv")
    slides = pd.read_csv(path / "slide_til_score_summary.csv")
    slides = slides[slides["row_type"] == "case"].copy()
    predictions["case_id"] = predictions["case_id"].astype(str)
    slides["case_id"] = slides["case_id"].astype(str)
    return {
        "name": name,
        "summary": summary,
        "predictions": predictions,
        "slides": slides,
    }


def metric_value(summary, key):
    return summary["patch_metrics"][key]["value"]


def summary_row(experiment):
    summary = experiment["summary"]
    slide = summary["slide_metrics"]
    calibration = summary["cross_fitted_linear_calibration"]
    return {
        "experiment": experiment["name"],
        "accuracy": metric_value(summary, "accuracy"),
        "macro_f1": metric_value(summary, "f1_macro"),
        "positive_auc": metric_value(
            summary, "positive_vs_rest_auc_positive_binary"
        ),
        "macro_ovr_auc": metric_value(summary, "ovr_auc_macro"),
        "weighted_ovr_auc": metric_value(summary, "ovr_auc_weighted"),
        "hard_til_mae": slide["mae"],
        "soft_til_mae": slide["soft_mae"],
        "hard_calibrated_mae": calibration["hard"]["mae"],
        "soft_calibrated_mae": calibration["soft"]["mae"],
        "hard_til_pearson": slide["pearson_r"],
        "hard_til_spearman": slide["spearman_r"],
    }


def sampled_metrics(experiment, sampled_cases):
    predictions = experiment["predictions"]
    slides = experiment["slides"].set_index("case_id")
    patch_groups = {
        case_id: group for case_id, group in predictions.groupby("case_id")
    }
    sampled_predictions = pd.concat(
        [patch_groups[case_id] for case_id in sampled_cases],
        ignore_index=True,
    )
    y_true = (sampled_predictions["y_true_label"] == "positive").astype(int)
    auc = roc_auc_score(y_true, sampled_predictions["prob_positive"])
    errors = [
        float(slides.loc[case_id, "abs_error"]) for case_id in sampled_cases
    ]
    return auc, float(np.mean(errors))


def percentile_interval(values):
    return {
        "median": float(np.percentile(values, 50.0)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def paired_bootstrap(experiments, baseline_name, repetitions, seed):
    baseline = experiments[baseline_name]
    cases = sorted(baseline["slides"]["case_id"].astype(str).tolist())
    expected = set(cases)
    for name, experiment in experiments.items():
        actual = set(experiment["slides"]["case_id"].astype(str))
        if actual != expected:
            raise ValueError(
                "{0} case set differs from baseline: {1}".format(
                    name, sorted(actual.symmetric_difference(expected))
                )
            )
    rng = np.random.RandomState(seed)
    differences = {
        name: {"positive_auc": [], "hard_til_mae": []}
        for name in experiments
        if name != baseline_name
    }
    for _ in range(repetitions):
        sampled = rng.choice(cases, size=len(cases), replace=True).tolist()
        baseline_auc, baseline_mae = sampled_metrics(baseline, sampled)
        for name, experiment in experiments.items():
            if name == baseline_name:
                continue
            auc, mae = sampled_metrics(experiment, sampled)
            differences[name]["positive_auc"].append(auc - baseline_auc)
            differences[name]["hard_til_mae"].append(mae - baseline_mae)

    rows = []
    for name, metrics in differences.items():
        for metric, values in metrics.items():
            interval = percentile_interval(np.asarray(values))
            rows.append(
                {
                    "baseline": baseline_name,
                    "experiment": name,
                    "metric": metric,
                    "direction": (
                        "higher_is_better"
                        if metric == "positive_auc"
                        else "lower_is_better"
                    ),
                    "delta_median": interval["median"],
                    "delta_ci_low": interval["ci_low"],
                    "delta_ci_high": interval["ci_high"],
                    "bootstrap_repetitions": repetitions,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    paths = parse_experiments(args.experiment)
    if args.baseline not in paths:
        raise ValueError("--baseline must match an experiment name")
    experiments = {
        name: read_experiment(name, path) for name, path in paths.items()
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [summary_row(experiment) for experiment in experiments.values()]
    )
    summary.to_csv(
        output / "experiment_comparison.csv",
        index=False,
        float_format="%.8f",
    )
    bootstrap = paired_bootstrap(
        experiments, args.baseline, args.bootstrap, args.seed
    )
    bootstrap.to_csv(
        output / "paired_case_bootstrap.csv",
        index=False,
        float_format="%.8f",
    )
    print(
        "Compared {0} experiments -> {1}".format(len(experiments), output)
    )


if __name__ == "__main__":
    main()
