#!/usr/bin/env python3
"""Build labeled patch CSV manifests from TCGA train/test or external cohort roots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.external_eval import build_patch_manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan a labeled patch root (dataset/train, dataset/test, etc.) and "
            "write case_id,image_path,label CSV."
        )
    )
    parser.add_argument(
        "--root",
        default="dataset/train",
        help="Root with A_positive/B_negative/C_other or A_Positive/... folders",
    )
    parser.add_argument(
        "--output",
        default="tcga_train_dataset.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--path-prefix",
        default="/workspace",
        help=(
            "Rewrite image_path prefix for Docker training "
            "(empty string keeps absolute host paths)"
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset label column value (defaults to root folder name)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError("Root not found: {0}".format(root))
    prefix = args.path_prefix.strip()
    if prefix:
        prefix = "{0}/{1}".format(prefix.rstrip("/"), root.as_posix())
    frame = build_patch_manifest(
        root,
        path_prefix=prefix,
        dataset_name=args.dataset_name or root.name,
    )
    output = Path(args.output)
    frame.loc[:, ["case_id", "image_path", "label"]].to_csv(output, index=False)
    print("Wrote {0} rows, {1} cases -> {2}".format(
        len(frame), frame["case_id"].nunique(), output
    ))
    print(frame["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
