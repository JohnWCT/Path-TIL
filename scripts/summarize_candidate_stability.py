#!/usr/bin/env python3
"""Summarize seed-stability metrics across candidate source-mix experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.candidate import reference_metrics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize positive AUC/PRC across candidate seed experiments."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="OOF result directories (e.g. results/results_oof_with_prc/source_mix_tcga_r50_50)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for stability summary artifacts",
    )
    return parser.parse_args()


def load_oof_metrics(oof_dir: Path) -> dict:
    metrics_path = oof_dir / "oof_metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {
            "experiment": oof_dir.name,
            "path": str(oof_dir),
            "positive_auc": payload.get("positive_auc"),
            "positive_prc": payload.get("positive_prc"),
            "macro_ovr_auc": payload.get("macro_ovr_auc"),
            "weighted_ovr_auc": payload.get("weighted_ovr_auc"),
            "hard_til_mae": payload.get("hard_til_mae"),
        }

    eval_summary = oof_dir / "eval_summary.json"
    if eval_summary.is_file():
        payload = json.loads(eval_summary.read_text(encoding="utf-8"))
        patch = payload.get("patch_metrics", {})
        slide = payload.get("slide_metrics", {})
        hard = (
            slide.get("hard_til_mae", {})
            if isinstance(slide.get("hard_til_mae"), dict)
            else {}
        )

        def nested_value(*keys):
            for key in keys:
                item = patch.get(key)
                if isinstance(item, dict) and "value" in item:
                    return item["value"]
                if item is not None and not isinstance(item, dict):
                    return item
            return None

        return {
            "experiment": oof_dir.name,
            "path": str(oof_dir),
            "positive_auc": nested_value(
                "positive_vs_rest_auc_positive_binary",
                "ovr_auc_positive_none",
            ),
            "positive_prc": nested_value(
                "positive_vs_rest_average_precision_positive_binary",
                "ovr_average_precision_positive_none",
            ),
            "macro_ovr_auc": nested_value("ovr_auc_macro"),
            "weighted_ovr_auc": nested_value("ovr_auc_weighted"),
            "hard_til_mae": hard.get("value"),
        }

    summary_path = oof_dir / "patch_metric_summary.csv"
    if not summary_path.is_file():
        summary_path = oof_dir / "patch_auc_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(
            "Missing OOF metrics in {0}; expected oof_metrics.json, "
            "eval_summary.json, or patch_*summary.csv".format(oof_dir)
        )
    summary = pd.read_csv(summary_path)
    lookup = {
        (
            row["metric"],
            "" if pd.isna(row.get("class", "")) else str(row.get("class", "")),
            "" if pd.isna(row.get("average", "")) else str(row.get("average", "")),
        ): row["value"]
        for _, row in summary.iterrows()
    }
    return {
        "experiment": oof_dir.name,
        "path": str(oof_dir),
        "positive_auc": lookup.get(("positive_vs_rest_auc", "positive", "binary")),
        "positive_prc": lookup.get(
            ("positive_vs_rest_average_precision", "positive", "binary")
        ),
        "macro_ovr_auc": lookup.get(("ovr_auc", "", "macro")),
        "weighted_ovr_auc": lookup.get(("ovr_auc", "", "weighted")),
        "hard_til_mae": None,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [load_oof_metrics(Path(path)) for path in args.experiments]
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_seed_summary.csv", index=False)

    auc_values = pd.to_numeric(frame["positive_auc"], errors="coerce").dropna()
    prc_values = pd.to_numeric(frame["positive_prc"], errors="coerce").dropna()
    summary = {
        "n_experiments": int(len(frame)),
        "mean_positive_auc": float(auc_values.mean()) if len(auc_values) else None,
        "std_positive_auc": float(auc_values.std(ddof=0)) if len(auc_values) else None,
        "mean_positive_prc": float(prc_values.mean()) if len(prc_values) else None,
        "std_positive_prc": float(prc_values.std(ddof=0)) if len(prc_values) else None,
        "reference": reference_metrics(),
        "experiments": rows,
    }
    with (output_dir / "candidate_stability_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
