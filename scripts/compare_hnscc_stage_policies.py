#!/usr/bin/env python3
"""Compare stage-selection policies on existing fold predictions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.stage_selection import STAGE_POLICIES, select_stage  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(STAGE_POLICIES),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in args.policies:
        policy_dir = output / policy
        if policy_dir.exists():
            shutil.rmtree(policy_dir)
        for fold in range(5):
            fold_dir = pred_dir / "fold{0:02d}".format(fold)
            metrics_path = fold_dir / "fold_metrics.json"
            with metrics_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            stage_metrics = {
                int(key): value
                for key, value in payload["stage_metrics"].items()
            }
            keras_auc = {
                int(key): float(value)
                for key, value in payload["validation_keras_auc"].items()
            }
            selected = select_stage(policy, stage_metrics, keras_auc)
            target_fold = policy_dir / "fold{0:02d}".format(fold)
            target_fold.mkdir(parents=True, exist_ok=True)
            for split in ("val", "test"):
                source = fold_dir / "stage{0}_{1}_predictions.csv".format(
                    selected, split
                )
                shutil.copy(
                    source, target_fold / "{0}_predictions.csv".format(split)
                )
            for stage in sorted(stage_metrics):
                for split in ("train", "val", "test"):
                    source = fold_dir / "stage{0}_{1}_predictions.csv".format(
                        stage, split
                    )
                    if source.is_file():
                        shutil.copy(source, target_fold / source.name)
            summary = {
                "fold": fold,
                "policy": policy,
                "selected_stage": selected,
                "validation_keras_auc": keras_auc,
            }
            with (target_fold / "fold_metrics.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
                handle.write("\n")
            rows.append(summary)
        print("Wrote policy predictions: {0}".format(policy_dir))
    pd.DataFrame(rows).to_csv(output / "stage_policy_selection.csv", index=False)
    print("Stage policy comparison scaffold -> {0}".format(output))
    print(
        "Next: run eval_hnscc_oof.py on each policy directory under {0}".format(
            output
        )
    )


if __name__ == "__main__":
    main()
