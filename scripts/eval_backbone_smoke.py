#!/usr/bin/env python3
"""Summarize fold 0+1 smoke metrics for backbone replacement screening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average fold 0+1 test metrics from backbone smoke training."
    )
    parser.add_argument("--pred-dir", required=True, help="Smoke training output root")
    parser.add_argument("--output", required=True, help="OOF-style summary directory")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    return parser.parse_args()


def load_fold_test_metrics(pred_dir: Path, fold: int) -> dict:
    metrics_path = pred_dir / "fold{0:02d}".format(fold) / "fold_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError("Missing smoke fold metrics: {0}".format(metrics_path))
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_metrics", {})
    test_metrics = selected.get("test")
    if not isinstance(test_metrics, dict):
        raise ValueError("fold {0} missing selected_metrics.test".format(fold))
    return test_metrics


def average_metric(rows: list[dict], key: str):
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = [load_fold_test_metrics(pred_dir, fold) for fold in args.folds]
    summary = {
        "experiment": output_dir.name,
        "path": str(output_dir),
        "smoke_folds": args.folds,
        "positive_auc": average_metric(fold_metrics, "positive_auc"),
        "positive_prc": average_metric(fold_metrics, "positive_prc"),
        "macro_ovr_auc": average_metric(fold_metrics, "macro_ovr_auc"),
        "weighted_ovr_auc": average_metric(fold_metrics, "weighted_ovr_auc"),
        "hard_til_mae": None,
        "per_fold_test_metrics": {
            str(fold): metrics for fold, metrics in zip(args.folds, fold_metrics)
        },
    }
    with (output_dir / "oof_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "smoke_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
