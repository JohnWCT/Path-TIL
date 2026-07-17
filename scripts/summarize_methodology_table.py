#!/usr/bin/env python3
"""Render a compact Markdown table from methodology_comparison.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = [
    "experiment_name",
    "keep_or_drop",
    "positive_auc",
    "positive_prc",
    "macro_ovr_auc",
    "weighted_ovr_auc",
    "accuracy",
    "macro_f1",
    "hard_til_mae",
    "soft_til_mae",
    "delta_positive_auc",
    "delta_positive_prc",
    "delta_hard_til_mae",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="methodology_comparison.csv")
    parser.add_argument("--output", required=True, help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_csv(args.input)
    present = [column for column in COLUMNS if column in frame.columns]
    subset = frame[present].copy()
    lines = [
        "# HNSCC Methodology Comparison",
        "",
        "| " + " | ".join(present) + " |",
        "| " + " | ".join(["---"] * len(present)) + " |",
    ]
    for _, row in subset.iterrows():
        values = []
        for column in present:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append("{0:.4f}".format(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print("Wrote {0}".format(args.output))


if __name__ == "__main__":
    main()
