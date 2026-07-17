#!/usr/bin/env python3
"""Build tcga_train_dataset.csv from dataset/train (pretrained TCGA source)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


LABEL_FOLDERS = {
    "A_positive": "positive",
    "B_negative": "negative",
    "C_other": "other",
}

CASE_RE = re.compile(r"(TCGA-[A-Z0-9-]+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan dataset/train (TCGA pretrained training patches) and write "
            "case_id,image_path,label CSV for source-mix training."
        )
    )
    parser.add_argument(
        "--train-dir",
        default="dataset/train",
        help="TCGA train root with A_positive/B_negative/C_other",
    )
    parser.add_argument(
        "--output",
        default="tcga_train_dataset.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--path-prefix",
        default="/workspace/dataset/train",
        help=(
            "Rewrite image_path prefix for Docker training "
            "(empty string keeps absolute host paths)"
        ),
    )
    return parser.parse_args()


def extract_case_id(path: Path) -> str:
    match = CASE_RE.search(path.name)
    if match is None:
        raise ValueError("Cannot extract TCGA case_id from {0}".format(path))
    return match.group(1)


def build_records(train_dir: Path, path_prefix: str) -> list[dict]:
    records = []
    for folder, label in LABEL_FOLDERS.items():
        sub = train_dir / folder
        if not sub.is_dir():
            raise FileNotFoundError("Missing label folder: {0}".format(sub))
        images = sorted(sub.glob("*.tif")) + sorted(sub.glob("*.tiff"))
        for path in images:
            case_id = extract_case_id(path)
            if path_prefix:
                image_path = "{0}/{1}/{2}".format(
                    path_prefix.rstrip("/"), folder, path.name
                )
            else:
                image_path = str(path.resolve())
            records.append(
                {
                    "case_id": case_id,
                    "image_path": image_path,
                    "label": label,
                }
            )
        print("{0} -> {1}: {2}".format(folder, label, len(images)))
    return records


def main():
    args = parse_args()
    train_dir = Path(args.train_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError("Train dir not found: {0}".format(train_dir))
    records = build_records(train_dir, args.path_prefix)
    frame = pd.DataFrame(records, columns=["case_id", "image_path", "label"])
    output = Path(args.output)
    frame.to_csv(output, index=False)
    print("Wrote {0} rows, {1} cases -> {2}".format(
        len(frame), frame["case_id"].nunique(), output
    ))
    print(frame["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
