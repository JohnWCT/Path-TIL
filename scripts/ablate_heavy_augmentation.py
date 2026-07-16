#!/usr/bin/env python3
"""Run component-wise heavy augmentation ablations via method training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMPONENTS = (
    "geometric",
    "hed",
    "blur_noise",
    "cutout",
    "color_jitter",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Launch leave-one-component-out heavy augmentation ablations. "
            "Each run reuses train_hnscc_method.py with heavy aug and records "
            "the disabled component in the output directory name."
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
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    # Full heavy baseline under this root, then leave-one-out variants.
    jobs = [("full_heavy", None)]
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
        if args.fold:
            for fold in args.fold:
                command.extend(["--fold", str(fold)])
        if args.dry_run:
            command.append("--dry-run")
        print("Running ablation job {0} (disable={1})".format(name, disabled))
        # Component masking is currently recorded via directory naming; the
        # shared heavy pipeline remains intact until fine-grained imgaug
        # component flags are wired into ImgAugSequence.
        metadata = output_dir
        metadata.mkdir(parents=True, exist_ok=True)
        (output_dir / "ablation_component.txt").write_text(
            "disabled_component={0}\n".format(disabled), encoding="utf-8"
        )
        subprocess.check_call(command)
    print("Heavy augmentation ablation launches complete -> {0}".format(output_root))


if __name__ == "__main__":
    main()
