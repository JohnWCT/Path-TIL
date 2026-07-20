#!/usr/bin/env python3
"""Reproducible wrapper for the current HNSCC source-mix candidate workflow."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from path_til.candidate import CANDIDATE_SETTINGS, default_method_config_path  # noqa: E402
from path_til.experiment_registry import load_method_config  # noqa: E402
from scripts import train_hnscc_source_mix as source_mix  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the locked HNSCC candidate (source mix 0.50:0.50, IRV2, heavy aug)."
        )
    )
    parser.add_argument(
        "--config",
        default=str(default_method_config_path(PROJECT_ROOT / "configs")),
    )
    parser.add_argument("--csv-hnscc", default="qupath_dataset.csv")
    parser.add_argument("--csv-tcga", default="tcga_train_dataset.csv")
    parser.add_argument("--fold-csv", default="folds_hnscc_group5.csv")
    parser.add_argument(
        "--pretrained",
        default=CANDIDATE_SETTINGS["pretrained_path"],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", action="append", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    method = load_method_config(cli.config)
    method["seed"] = int(cli.seed)
    args = source_mix.build_namespace(
        argparse.Namespace(
            config=cli.config,
            csv_hnscc=cli.csv_hnscc,
            csv_tcga=cli.csv_tcga,
            fold_csv=cli.fold_csv,
            pretrained=cli.pretrained,
            output_dir=cli.output_dir,
            hnscc_ratio=None,
            tcga_ratio=None,
            fold=cli.fold,
            dry_run=cli.dry_run,
            epochs_stage1=None,
            epochs_stage2=None,
            batch_size=None,
        ),
        method,
    )
    source_mix.base.validate_args(args)
    source_mix.install_source_mix_split()

    hnscc = source_mix.base.load_hnscc_csv(args.csv, expected_cases=10)
    assignments = source_mix.base.load_assignments(args.fold_csv)
    source_mix.base.validate_fold_assignments(hnscc, assignments, n_folds=5)
    tcga = source_mix.load_tcga_csv(args.csv_tcga)
    mixed, sampled = source_mix.build_mixed_frame(
        hnscc, tcga, args.hnscc_ratio, args.tcga_ratio, args.seed
    )
    folds = source_mix.base.selected_folds(args.fold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mixed.to_csv(output_dir / "mixed_frame.csv", index=False)
    sampled.to_csv(output_dir / "tcga_sampled.csv", index=False)
    summary = {
        "candidate": "source_mix_tcga_r50_50",
        "seed": args.seed,
        "hnscc_ratio": args.hnscc_ratio,
        "tcga_ratio": args.tcga_ratio,
    }
    source_mix.base.write_json(output_dir / "candidate_summary.json", summary)
    print("Candidate summary: {0}".format(summary))

    if args.dry_run:
        print("Candidate dry run complete -> {0}".format(output_dir))
        return

    import tensorflow as tf

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf = source_mix.base.configure_tensorflow(args)
    images, preprocessing_report = source_mix.base.load_all_images(
        mixed["image_path"].tolist(),
        source_mix.base.on_off(args.hne_norm),
        args.image_workers,
    )
    auc_class = source_mix.base.sparse_auc_class(tf)
    for fold in folds:
        config = source_mix.make_source_mix_fold_config(
            args, mixed, assignments, fold, folds, summary
        )
        config["preprocessing_report"] = preprocessing_report
        fold_dir = output_dir / "fold{0:02d}".format(fold)
        fold_dir.mkdir(parents=True, exist_ok=True)
        source_mix.base.write_json(fold_dir / "config.json", config)
        source_mix.base.train_fold(
            tf, auc_class, args, mixed, assignments, images, fold, config
        )
    print("Final candidate training complete: {0}".format(output_dir))


if __name__ == "__main__":
    main()
