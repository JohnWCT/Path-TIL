#!/usr/bin/env python3
"""Train HNSCC source-mix models with optional L2-SP anti-forgetting."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from path_til.experiment_registry import load_method_config  # noqa: E402
from path_til.l2sp import l2sp_penalty, snapshot_trainable_weights  # noqa: E402
from scripts import train_hnscc_source_mix as source_mix  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune HNSCC GroupCV with source mix and optional L2-SP penalty "
            "toward the initial pretrained weights."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv-hnscc", default="qupath_dataset.csv")
    parser.add_argument("--csv-tcga", default="tcga_train_dataset.csv")
    parser.add_argument("--fold-csv", default="folds_hnscc_group5.csv")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hnscc-ratio", type=float, default=None)
    parser.add_argument("--tcga-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fold", action="append", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs-stage1", type=int, default=None)
    parser.add_argument("--epochs-stage2", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def install_l2sp_training(lambda_l2sp: float):
    base = source_mix.base
    original_compile = base.compile_model
    original_train_fold = base.train_fold
    original_load = base.load_and_validate_model
    state = {"theta_star": None, "snapshot_taken": False}

    def patched_compile_model(tf, model, learning_rate, auc_class):
        if lambda_l2sp <= 0.0 or state["theta_star"] is None:
            return original_compile(tf, model, learning_rate, auc_class)
        base_loss = tf.keras.losses.SparseCategoricalCrossentropy()

        def loss_fn(y_true, y_pred):
            loss = base_loss(y_true, y_pred)
            return loss + lambda_l2sp * l2sp_penalty(model, state["theta_star"])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=loss_fn,
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                auc_class(base.NUM_CLASSES, name="auc"),
            ],
        )

    def patched_load_and_validate_model(tf, path, mixed_precision):
        model = original_load(tf, path, mixed_precision)
        if not state["snapshot_taken"]:
            state["theta_star"] = snapshot_trainable_weights(model)
            state["snapshot_taken"] = True
        return model

    def patched_train_fold(tf, auc_class, args, frame, assignments, images, fold, config):
        state["snapshot_taken"] = False
        state["theta_star"] = None
        return original_train_fold(
            tf, auc_class, args, frame, assignments, images, fold, config
        )

    base.compile_model = patched_compile_model
    base.load_and_validate_model = patched_load_and_validate_model
    base.train_fold = patched_train_fold


def main():
    cli = parse_args()
    method = load_method_config(cli.config)
    if cli.seed is not None:
        method["seed"] = int(cli.seed)
    lambda_l2sp = float(method.get("lambda_l2sp", 0.0))
    install_l2sp_training(lambda_l2sp)

    args = source_mix.build_namespace(cli, method)
    if cli.fold is not None:
        args.fold = cli.fold
    args.dry_run = cli.dry_run
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
        "hnscc_ratio": args.hnscc_ratio,
        "tcga_ratio": args.tcga_ratio,
        "lambda_l2sp": lambda_l2sp,
        "seed": args.seed,
        "n_hnscc": int(len(hnscc)),
        "n_tcga_pool": int(len(tcga)),
        "n_tcga_sampled": int(len(sampled)),
        "n_mixed": int(len(mixed)),
    }
    source_mix.base.write_json(output_dir / "source_mix_summary.json", summary)
    print("L2-SP source-mix summary: {0}".format(summary))

    if args.dry_run:
        print("L2-SP dry run complete -> {0}".format(output_dir))
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
        config["lambda_l2sp"] = lambda_l2sp
        fold_dir = output_dir / "fold{0:02d}".format(fold)
        fold_dir.mkdir(parents=True, exist_ok=True)
        source_mix.base.write_json(fold_dir / "config.json", config)
        source_mix.base.train_fold(
            tf, auc_class, args, mixed, assignments, images, fold, config
        )
    print("L2-SP source-mix training complete: {0}".format(output_dir))


if __name__ == "__main__":
    main()
