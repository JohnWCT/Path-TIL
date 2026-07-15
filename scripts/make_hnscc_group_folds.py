#!/usr/bin/env python3
"""Create deterministic case-grouped folds for the HNSCC patch dataset."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.hnscc import (  # noqa: E402
    build_fold_assignments,
    build_summary,
    load_hnscc_csv,
    validate_fold_assignments,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate an HNSCC patch CSV and create deterministic grouped folds "
            "with two test cases and one validation case per fold."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Input CSV with exactly case_id,image_path,label columns",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output case-level fold CSV (fold,case_id,role)",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Summary JSON path (default: output path with a .json suffix)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds (default: 5; requires exactly twice as many cases)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic tie-breaking seed (default: 42)",
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=10,
        help="Exact number of cases required in the input CSV (default: 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_folds < 2:
        raise SystemExit("--n-folds must be at least 2")
    if args.expected_cases != 2 * args.n_folds:
        raise SystemExit(
            "--expected-cases must equal 2 * --n-folds for paired test folds"
        )

    output_path = Path(args.output)
    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else output_path.with_suffix(".json")
    )

    try:
        frame = load_hnscc_csv(args.csv, expected_cases=args.expected_cases)
        assignments, objective = build_fold_assignments(
            frame, n_folds=args.n_folds, seed=args.seed
        )
        validate_fold_assignments(frame, assignments, n_folds=args.n_folds)
        summary = build_summary(frame, assignments, args.seed, objective)
    except (OSError, ValueError) as error:
        raise SystemExit("Error: {0}".format(error))

    assignments.to_csv(str(output_path), index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("Validated {0} patches across {1} cases.".format(
        len(frame), frame["case_id"].nunique()
    ))
    print("Wrote fold assignments: {0}".format(output_path))
    print("Wrote fold summary: {0}".format(summary_path))
    print("Objective: {0:.12g}".format(objective["total"]))


if __name__ == "__main__":
    main()
