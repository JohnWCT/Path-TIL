#!/usr/bin/env python3
"""Compare methodology OOF summaries against the current candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.experiment_registry import (  # noqa: E402
    CANDIDATE_REFERENCE,
    keep_or_drop,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Reference OOF dir")
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment OOF dirs or summary JSON files",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_metrics(path):
    path = Path(path)
    summary_path = path
    experiment_name = path.name
    if path.is_dir():
        for name in (
            "eval_summary.json",
            "threshold_til_summary.json",
            "oof_summary/eval_summary.json",
        ):
            candidate = path / name
            if candidate.is_file():
                summary_path = candidate
                break
        else:
            raise FileNotFoundError("No summary JSON under {0}".format(path))
        if path.name == "oof_summary":
            experiment_name = path.parent.name
        else:
            experiment_name = path.name
    with summary_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if "patch_metrics" in payload:
        patch = payload["patch_metrics"]
        slide = payload["slide_metrics"]
        return {
            "experiment_name": experiment_name,
            "positive_auc": patch["positive_vs_rest_auc_positive_binary"]["value"],
            "macro_ovr_auc": patch["ovr_auc_macro"]["value"],
            "weighted_ovr_auc": patch["ovr_auc_weighted"]["value"],
            "accuracy": patch["accuracy"]["value"],
            "macro_f1": patch["f1_macro"]["value"],
            "hard_til_mae": slide["mae"],
            "soft_til_mae": slide.get("soft_mae"),
            "pearson": slide.get("pearson_r"),
            "spearman": slide.get("spearman_r"),
            "summary_path": str(summary_path),
        }
    # threshold summary style
    return {
        "experiment_name": experiment_name,
        "positive_auc": payload.get("positive_auc"),
        "macro_ovr_auc": payload.get("macro_ovr_auc"),
        "weighted_ovr_auc": payload.get("weighted_ovr_auc"),
        "accuracy": payload.get("accuracy"),
        "macro_f1": payload.get("macro_f1"),
        "hard_til_mae": payload.get("hard_til_mae"),
        "soft_til_mae": payload.get("soft_til_mae"),
        "pearson": payload.get("pearson"),
        "spearman": payload.get("spearman"),
        "summary_path": str(summary_path),
    }


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reference = load_metrics(args.reference)
    rows = []
    ref_metrics = {
        "positive_auc": reference["positive_auc"],
        "hard_til_mae": reference["hard_til_mae"],
        "macro_ovr_auc": reference["macro_ovr_auc"],
        "weighted_ovr_auc": reference["weighted_ovr_auc"],
    }
    # Ensure reference row uses candidate constants when available.
    decision_ref = keep_or_drop(ref_metrics, CANDIDATE_REFERENCE)
    reference_row = dict(reference)
    reference_row.update(
        {
            "method_type": "reference",
            "delta_positive_auc": decision_ref["delta_positive_auc"],
            "delta_hard_til_mae": decision_ref["delta_hard_til_mae"],
            "keep_or_drop": "reference",
            "notes": "current candidate reference",
        }
    )
    rows.append(reference_row)

    for experiment in args.experiments:
        metrics = load_metrics(experiment)
        usable = {
            key: metrics[key]
            for key in (
                "positive_auc",
                "hard_til_mae",
                "macro_ovr_auc",
                "weighted_ovr_auc",
            )
            if metrics.get(key) is not None
        }
        if len(usable) == 4:
            decision = keep_or_drop(usable, CANDIDATE_REFERENCE)
            metrics["delta_positive_auc"] = decision["delta_positive_auc"]
            metrics["delta_hard_til_mae"] = decision["delta_hard_til_mae"]
            metrics["keep_or_drop"] = decision["decision"]
            metrics["notes"] = ";".join(decision["reasons"]) or "meets_criteria"
        else:
            metrics["delta_positive_auc"] = None
            metrics["delta_hard_til_mae"] = None
            metrics["keep_or_drop"] = "incomplete"
            metrics["notes"] = "missing_metrics_for_keep_or_drop"
        rows.append(metrics)

    frame = pd.DataFrame(rows)
    frame.to_csv(
        output / "methodology_comparison.csv", index=False, float_format="%.8f"
    )
    with (output / "methodology_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("Wrote methodology comparison -> {0}".format(output))


if __name__ == "__main__":
    main()
