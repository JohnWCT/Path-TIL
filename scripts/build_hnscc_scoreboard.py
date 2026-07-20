#!/usr/bin/env python3
"""Build the HNSCC living scoreboard Markdown and CSV from registered OOF dirs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.paths import detect_repo_root  # noqa: E402
from path_til.scoreboard import write_scoreboard  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build docs/hnscc_living_scoreboard.md from configs/scoreboard_experiments.yaml"
        )
    )
    parser.add_argument(
        "--registry",
        default="configs/scoreboard_experiments.yaml",
        help="Experiment registry YAML",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing OOF outputs (default: results/)",
    )
    parser.add_argument(
        "--output-md",
        default="docs/hnscc_living_scoreboard.md",
        help="Markdown output path",
    )
    parser.add_argument(
        "--output-csv",
        default="results/results_methodology_comparison_scoreboard/scoreboard.csv",
        help="CSV output path",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Repo root (default: detect /workspace or PATH_TIL_ROOT)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve() if args.root else detect_repo_root()
    registry = root / args.registry
    results_root = root / args.results_root
    output_md = root / args.output_md
    output_csv = root / args.output_csv if args.output_csv else None

    if not registry.is_file():
        raise SystemExit("Registry not found: {0}".format(registry))
    if not results_root.is_dir():
        raise SystemExit("Results root not found: {0}".format(results_root))

    payload = write_scoreboard(
        registry,
        results_root,
        output_md,
        output_csv=output_csv,
    )
    print(
        "Wrote scoreboard -> {0} ({1} experiments, {2} missing)".format(
            output_md,
            len(payload["all_rows"]),
            len(payload["missing"]),
        )
    )
    if output_csv:
        print("Wrote CSV -> {0}".format(output_csv))
    if payload["missing"]:
        print("Missing OOF paths:")
        for item in payload["missing"]:
            print("  - {0}: {1}".format(item.get("id"), item.get("oof_path")))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
