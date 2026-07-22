#!/usr/bin/env python3
"""Summarize fold 0+1 smoke / B5 metrics for backbone replacement screening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.backbone_decision import (  # noqa: E402
    BackboneMetrics,
    decide_backbone_status,
)
from path_til.backbone_metrics import summarize_smoke_folds  # noqa: E402
from path_til.candidate import reference_metrics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average fold 0+1 test metrics from backbone smoke / B5 training."
    )
    parser.add_argument("--pred-dir", required=True, help="Training output root")
    parser.add_argument("--output", required=True, help="Summary output directory")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--csv", default=None, help="Unused; kept for CLI compatibility")
    parser.add_argument(
        "--fold-csv", default=None, help="Unused; kept for CLI compatibility"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.experiment_name or pred_dir.name
    summary = summarize_smoke_folds(pred_dir, folds=args.folds, experiment_name=name)

    reference = reference_metrics()
    decision = decide_backbone_status(
        BackboneMetrics(
            name=name,
            positive_auc=float(summary["positive_auc"] or 0.0),
            positive_prc=float(summary["positive_prc"] or 0.0),
            macro_ovr_auc=float(summary["macro_ovr_auc"] or 0.0),
            weighted_ovr_auc=float(summary["weighted_ovr_auc"] or 0.0),
            fold_auc_gap=summary.get("fold_auc_gap"),
            fold_prc_gap=summary.get("fold_prc_gap"),
        ),
        BackboneMetrics(
            name="irv2_candidate",
            positive_auc=float(reference["positive_auc"]),
            positive_prc=float(reference["positive_prc"]),
            macro_ovr_auc=float(reference["macro_ovr_auc"]),
            weighted_ovr_auc=float(reference["weighted_ovr_auc"]),
        ),
    )
    summary["decision"] = decision.decision
    summary["reasons"] = decision.reasons

    for filename in (
        "oof_metrics.json",
        "smoke_summary.json",
        "metrics.json",
        "eval_summary.json",
    ):
        with (output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
