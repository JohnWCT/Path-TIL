#!/usr/bin/env python3
"""Evaluate trained fold models on external lock-box cohorts under dataset/Testset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.external_eval import (  # noqa: E402
    EXTERNAL_DATASETS,
    SUMMARY_COLUMNS,
    build_external_dataset_manifest,
    metrics_to_summary_row,
    write_metrics_bundle,
)
from scripts._eval_patch_manifest import evaluate_manifest  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GroupCV fold models on CPTAC_LUAD, CPTAC_LUSC, and RUMC-BRCA. "
            "External results are report-only and must not be used for tuning."
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--testset-root", default="dataset/Testset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", default="selected")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--hne-norm",
        choices=("on", "off", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--path-prefix",
        default="/workspace/dataset/Testset",
        help="Manifest path prefix for dataset/Testset cohorts",
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=min(8, max(1, __import__("os").cpu_count() or 1)),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    testset_root = Path(args.testset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hne_norm = None if args.hne_norm == "auto" else args.hne_norm == "on"

    manifests = build_external_dataset_manifest(testset_root, path_prefix=args.path_prefix)
    summary_rows = []
    for dataset in EXTERNAL_DATASETS:
        manifest = manifests[dataset]
        cohort_root = testset_root / dataset
        metrics, predictions, report, model_paths = evaluate_manifest(
            Path(args.model_dir),
            manifest,
            cohort_root,
            stage=args.stage,
            batch_size=args.batch_size,
            hne_norm=hne_norm,
            image_workers=args.image_workers,
        )
        write_metrics_bundle(output_dir, dataset, metrics, predictions)
        summary_rows.append(metrics_to_summary_row(dataset, metrics, len(manifest)))
        sidecar = output_dir / "{0}_evaluation_report.json".format(dataset)
        with sidecar.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset": dataset,
                    "model_dir": str(args.model_dir),
                    "testset_root": str(cohort_root),
                    "stage": args.stage,
                    "model_paths": model_paths,
                    "preprocessing_report": report,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        print("Finished external cohort {0}".format(dataset))

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary.to_csv(output_dir / "external_summary.csv", index=False)
    print("External lock-box evaluation complete -> {0}".format(output_dir))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
