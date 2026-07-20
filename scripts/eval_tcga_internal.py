#!/usr/bin/env python3
"""Evaluate trained fold models on TCGA internal holdout (dataset/test)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.external_eval import build_patch_manifest, metrics_to_summary_row  # noqa: E402
from scripts._eval_patch_manifest import evaluate_manifest  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GroupCV fold models on TCGA internal test patches."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-root", default="dataset/test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", default="selected")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--hne-norm",
        choices=("on", "off", "auto"),
        default="auto",
        help="Use training config when auto (default)",
    )
    parser.add_argument(
        "--path-prefix",
        default="/workspace/dataset/test",
        help="Manifest path prefix for dataset/test",
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=min(8, max(1, __import__("os").cpu_count() or 1)),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    test_root = Path(args.test_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_patch_manifest(test_root, path_prefix=args.path_prefix)
    hne_norm = None if args.hne_norm == "auto" else args.hne_norm == "on"
    metrics, predictions, report, model_paths = evaluate_manifest(
        Path(args.model_dir),
        manifest,
        test_root,
        stage=args.stage,
        batch_size=args.batch_size,
        hne_norm=hne_norm,
        image_workers=args.image_workers,
    )

    summary = metrics_to_summary_row("tcga_internal", metrics, len(manifest))
    with (output_dir / "tcga_internal_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    predictions.to_csv(output_dir / "tcga_internal_predictions.csv", index=False)
    with (output_dir / "evaluation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_dir": str(args.model_dir),
                "test_root": str(test_root),
                "stage": args.stage,
                "model_paths": model_paths,
                "preprocessing_report": report,
                "summary": summary,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print("TCGA internal evaluation complete -> {0}".format(output_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
