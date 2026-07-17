#!/usr/bin/env python3
"""Run component-wise heavy augmentation ablations via method training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# color_jitter is historically probability-0 in the candidate heavy pipeline,
# so leave-one-out on it is a no-op; still allow it for explicit checks.
COMPONENTS = (
    "geometric",
    "hed",
    "blur_noise",
    "cutout",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Launch leave-one-component-out heavy augmentation ablations. "
            "Each run reuses train_hnscc_method.py with --disable-aug-component."
        )
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--fold-csv", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "method_heavy_aug_ablation.yaml"),
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=list(COMPONENTS),
        help="Components to disable one at a time",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fold", action="append", type=int, default=None)
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip the full_heavy control run (reuse candidate OOF instead)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    if not args.skip_full:
        jobs.append(("full_heavy", None))
    for component in args.components:
        jobs.append(("without_{0}".format(component), component))

    for name, disabled in jobs:
        output_dir = output_root / name
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_hnscc_method.py"),
            "--config",
            args.config,
            "--csv",
            args.csv,
            "--fold-csv",
            args.fold_csv,
            "--pretrained",
            args.pretrained,
            "--output-dir",
            str(output_dir),
        ]
        if disabled is not None:
            command.extend(["--disable-aug-component", disabled])
        if args.fold:
            for fold in args.fold:
                command.extend(["--fold", str(fold)])
        if args.dry_run:
            command.append("--dry-run")
        print("Running ablation job {0} (disable={1})".format(name, disabled))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "ablation_component.txt").write_text(
            "disabled_component={0}\n".format(disabled), encoding="utf-8"
        )
        subprocess.check_call(command)
    print("Heavy augmentation ablation launches complete -> {0}".format(output_root))


if __name__ == "__main__":
    main()
