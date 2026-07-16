#!/usr/bin/env python3
"""Mix TCGA/source patches into HNSCC train splits only (no test leakage)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.hnscc import LABELS  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-hnscc", required=True)
    parser.add_argument("--csv-tcga", required=True)
    parser.add_argument("--fold-csv", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hnscc-ratio", type=float, default=0.75)
    parser.add_argument("--tcga-ratio", type=float, default=0.25)
    parser.add_argument("--aug", default="heavy")
    parser.add_argument("--hne-norm", default="off")
    parser.add_argument("--class-weight", default="on")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_ratio(hnscc_ratio, tcga_ratio):
    if hnscc_ratio < 0 or tcga_ratio < 0:
        raise ValueError("ratios must be non-negative")
    total = hnscc_ratio + tcga_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            "hnscc_ratio + tcga_ratio must equal 1.0; got {0}".format(total)
        )


def load_source_csv(path, source_name):
    frame = pd.read_csv(path)
    required = {"image_path", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("{0} missing columns: {1}".format(path, missing))
    invalid = sorted(set(frame["label"].astype(str)) - set(LABELS))
    if invalid:
        raise ValueError("{0} has invalid labels: {1}".format(path, invalid))
    frame = frame.copy()
    frame["source"] = source_name
    if "case_id" not in frame.columns:
        frame["case_id"] = source_name + "_unknown"
    frame["case_id"] = frame["case_id"].astype(str)
    return frame


def build_mixed_train_manifest(hnscc, tcga, fold_csv, hnscc_ratio, tcga_ratio, seed):
    folds = pd.read_csv(fold_csv)
    rng = np.random.RandomState(seed)
    rows = []
    for fold, group in folds.groupby("fold"):
        train_cases = group.loc[group["role"] == "train", "case_id"].astype(str)
        hnscc_train = hnscc[hnscc["case_id"].isin(train_cases)].copy()
        n_hnscc = len(hnscc_train)
        if n_hnscc == 0:
            raise ValueError("Fold {0} has no HNSCC train patches".format(fold))
        n_tcga = int(round(n_hnscc * tcga_ratio / hnscc_ratio)) if hnscc_ratio else 0
        if n_tcga > 0:
            if len(tcga) == 0:
                raise ValueError("TCGA CSV is empty")
            chosen = tcga.sample(n=n_tcga, replace=True, random_state=rng.randint(1e9))
        else:
            chosen = tcga.iloc[0:0].copy()
        mixed = pd.concat([hnscc_train, chosen], ignore_index=True)
        mixed["fold"] = int(fold)
        mixed["role"] = "train_mix"
        rows.append(mixed)
        print(
            "Fold {0}: HNSCC train={1}, TCGA mix={2}, ratio target={3}/{4}".format(
                fold, n_hnscc, len(chosen), hnscc_ratio, tcga_ratio
            )
        )
    return pd.concat(rows, ignore_index=True)


def main():
    args = parse_args()
    validate_ratio(args.hnscc_ratio, args.tcga_ratio)
    tcga_path = Path(args.csv_tcga)
    if not tcga_path.is_file():
        raise FileNotFoundError(
            "TCGA/source CSV not found: {0}. Create tcga_train_dataset.csv "
            "before running source-mix training.".format(tcga_path)
        )
    hnscc = load_source_csv(args.csv_hnscc, "hnscc")
    tcga = load_source_csv(args.csv_tcga, "tcga")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mixed = build_mixed_train_manifest(
        hnscc,
        tcga,
        args.fold_csv,
        args.hnscc_ratio,
        args.tcga_ratio,
        args.seed,
    )
    mixed_path = output / "mixed_train_manifest.csv"
    mixed.to_csv(mixed_path, index=False)
    summary = {
        "hnscc_ratio": args.hnscc_ratio,
        "tcga_ratio": args.tcga_ratio,
        "n_mixed_rows": int(len(mixed)),
        "aug": args.aug,
        "hne_norm": args.hne_norm,
        "class_weight": args.class_weight,
        "pretrained": args.pretrained,
        "note": (
            "Manifest prepared for train-only mixing. Full IRV2 trainer wiring "
            "that injects mixed rows into each fold train split is the next step; "
            "held-out HNSCC test cases remain untouched."
        ),
    }
    import json

    with (output / "source_mix_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if args.dry_run:
        print("Source-mix dry run complete -> {0}".format(output))
        return
    print(
        "Prepared source-mix manifest at {0}. "
        "Launch training with train_hnscc_method.py once the trainer accepts "
        "an external train-mix CSV.".format(mixed_path)
    )


if __name__ == "__main__":
    main()
