#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate TIL score text files into one CSV table.

Example:
python aggregate_til_scores.py --results_dir results_fold04_stage2_tvgh
"""

import argparse
import csv
import glob
import math
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Aggregate *_TIL_score.txt files into a CSV table."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing *_TIL_score.txt files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path. Default: <results_dir>/til_scores_summary.csv",
    )
    return parser.parse_args()


def slide_id_from_score_path(score_file):
    """由檔名推得 slide_id，可正確處理含空格或括號的檔名（如 S110-12651K (1)）。"""
    return os.path.basename(score_file).replace("_TIL_score.txt", "")


def parse_score_file(score_file):
    """
    解析 TILscout_edit.py 寫入的 *_TIL_score.txt。

    檔案內容格式為「<slide_id> <score>」，以空白分隔，例如：
      S104-10358A 0.011234847738247757
      S110-12651K (1) 0.012385877545808593

    slide_id 本身可能含空格（複製檔名如 "S110-12651K (1)"），
    因此不可假設 parts[0]=id、parts[1]=score；分數一律取最後一個 token。
    slide_id 以檔名為準，與 TILscout_edit 輸出檔命名一致。
    """
    slide_id = slide_id_from_score_path(score_file)

    with open(score_file, "r", encoding="utf-8") as f:
        line = f.readline().strip()

    if not line:
        return slide_id, math.nan

    parts = line.split()
    if len(parts) < 2:
        return slide_id, math.nan

    try:
        score = float(parts[-1])
    except ValueError:
        score = math.nan

    return slide_id, score


def main():
    args = parse_arguments()
    results_dir = os.path.abspath(args.results_dir)
    output_csv = args.output_csv or os.path.join(results_dir, "til_scores_summary.csv")

    score_files = sorted(glob.glob(os.path.join(results_dir, "*_TIL_score.txt")))
    if not score_files:
        raise FileNotFoundError(f"No *_TIL_score.txt files found in: {results_dir}")

    rows = []
    for score_file in score_files:
        slide_id, score = parse_score_file(score_file)
        rows.append(
            {
                "slide_id": slide_id,
                "til_score": score,
                "source_file": os.path.basename(score_file),
            }
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_id", "til_score", "source_file"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Aggregated {len(rows)} score files into: {output_csv}")


if __name__ == "__main__":
    main()
