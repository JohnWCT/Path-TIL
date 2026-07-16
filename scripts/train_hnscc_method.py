#!/usr/bin/env python3
"""Train HNSCC GroupCV with a YAML methodology configuration."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_base_module():
    path = PROJECT_ROOT / "scripts" / "train_hnscc_groupcv_irv2.py"
    spec = importlib.util.spec_from_file_location(
        "train_hnscc_groupcv_irv2", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_base_module()

from path_til.experiment_registry import load_method_config  # noqa: E402
from path_til.hnscc import LABELS, balanced_class_weights  # noqa: E402
from path_til.losses import (  # noqa: E402
    build_keras_loss,
    class_frequencies,
    effective_number_weights,
)
from path_til.samplers import make_epoch_indices  # noqa: E402
from path_til.stage_selection import select_stage  # noqa: E402


LABEL_TO_IDX = {label: index for index, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HNSCC GroupCV using a methodology YAML config"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--fold-csv", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", action="append", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs-stage1", type=int, default=None)
    parser.add_argument("--epochs-stage2", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def build_namespace(cli, method):
    return argparse.Namespace(
        csv=cli.csv,
        fold_csv=cli.fold_csv,
        pretrained=cli.pretrained,
        output_dir=cli.output_dir,
        fold=cli.fold,
        aug=method.get("aug", "heavy"),
        hne_norm=method.get("hne_norm", "off"),
        class_weight=method.get("class_weight", "on"),
        epochs_stage1=(
            cli.epochs_stage1
            if cli.epochs_stage1 is not None
            else int(method.get("epochs_stage1", 30))
        ),
        epochs_stage2=(
            cli.epochs_stage2
            if cli.epochs_stage2 is not None
            else int(method.get("epochs_stage2", 30))
        ),
        batch_size=(
            cli.batch_size
            if cli.batch_size is not None
            else int(method.get("batch_size", 32))
        ),
        seed=int(method.get("seed", 42)),
        image_workers=min(8, max(1, os.cpu_count() or 1)),
        fit_workers=1,
        use_multiprocessing="off",
        mixed_precision="off",
        patience=10,
        dry_run=cli.dry_run,
        baseline_only=False,
        method=method,
        loss=method.get("loss", "weighted_ce"),
        sampler=method.get("sampler", "random"),
        stage_policy=method.get(
            "stage_policy", "validation_multiclass_auc"
        ),
    )


def loss_for_args(tf, args, train_labels):
    counts, frequencies = class_frequencies(train_labels, NUM_CLASSES)
    class_weight_map = None
    if args.loss in ("weighted_ce", "class_balanced_focal"):
        if args.loss == "class_balanced_focal":
            weights = effective_number_weights(counts)
        else:
            weight_dict = balanced_class_weights(
                [LABELS[index] for index in train_labels]
            )
            weights = np.array(
                [weight_dict[index] for index in range(NUM_CLASSES)],
                dtype=np.float64,
            )
        class_weight_map = {
            index: float(weights[index]) for index in range(NUM_CLASSES)
        }
    return build_keras_loss(
        tf,
        args.loss,
        NUM_CLASSES,
        class_weights=class_weight_map,
        frequencies=frequencies,
    )


def install_patches(args):
    original_compile = base.compile_model
    original_training_inputs = base.training_inputs
    original_train_fold = base.train_fold

    def compile_model(tf, model, learning_rate, auc_class):
        train_labels = getattr(args, "_current_train_labels", None)
        if train_labels is None:
            return original_compile(tf, model, learning_rate, auc_class)
        loss = loss_for_args(tf, args, train_labels)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=loss,
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                auc_class(NUM_CLASSES, name="auc"),
            ],
        )

    def training_inputs(tf, train_args, images, labels, fold, stage_seed):
        sampler = getattr(train_args, "sampler", "random")
        if sampler == "random":
            return original_training_inputs(
                tf, train_args, images, labels, fold, stage_seed
            )
        case_ids = getattr(train_args, "_current_train_case_ids")
        steps = max(1, int(np.ceil(len(images) / float(train_args.batch_size))))
        epoch_indices = make_epoch_indices(
            labels,
            case_ids,
            sampler,
            steps_per_epoch=steps,
            batch_size=train_args.batch_size,
            seed=stage_seed,
        )
        return original_training_inputs(
            tf,
            train_args,
            images[epoch_indices],
            labels[epoch_indices],
            fold,
            stage_seed,
        )

    def train_fold(tf, auc_class, train_args, frame, assignments, images, fold, config):
        indices = base.split_indices(frame, assignments, fold)
        train_args = copy.copy(train_args)
        train_args._current_train_labels = (
            frame.iloc[indices["train"]]["label"]
            .map(LABEL_TO_IDX)
            .to_numpy(dtype=np.int32)
        )
        train_args._current_train_case_ids = (
            frame.iloc[indices["train"]]["case_id"].astype(str).to_numpy()
        )
        config = copy.deepcopy(config)
        # Loss-internal weighting or sampler balancing: disable fit class_weight.
        if train_args.loss in (
            "weighted_ce",
            "class_balanced_focal",
            "logit_adjusted_ce",
            "focal_gamma1",
            "focal_gamma2",
        ) or train_args.sampler != "random":
            config["class_weights"] = None

        payload = original_train_fold(
            tf,
            auc_class,
            train_args,
            frame,
            assignments,
            images,
            fold,
            config,
        )
        fold_dir = Path(train_args.output_dir) / "fold{0:02d}".format(fold)
        selected = select_stage(
            train_args.stage_policy,
            payload["stage_metrics"],
            payload.get("validation_keras_auc"),
        )
        if selected != int(payload["selected_stage"]):
            import pandas as pd

            for split in ("val", "test"):
                source = fold_dir / "stage{0}_{1}_predictions.csv".format(
                    selected, split
                )
                target = fold_dir / "{0}_predictions.csv".format(split)
                pd.read_csv(source).to_csv(
                    target, index=False, float_format="%.8f"
                )
            payload["selected_stage"] = selected
            payload["selected_metrics"] = payload["stage_metrics"][
                selected
                if selected in payload["stage_metrics"]
                else str(selected)
            ]
        payload["stage_policy"] = train_args.stage_policy
        payload["method"] = train_args.method
        base.write_json(fold_dir / "fold_metrics.json", payload)
        print(
            "Fold {0}: selected stage {1} by policy {2}".format(
                fold, payload["selected_stage"], train_args.stage_policy
            )
        )
        return payload

    base.compile_model = compile_model
    base.training_inputs = training_inputs
    base.train_fold = train_fold


def main():
    cli = parse_args()
    method = load_method_config(cli.config)
    args = build_namespace(cli, method)
    base.validate_args(args)
    install_patches(args)

    frame = base.load_hnscc_csv(args.csv, expected_cases=10)
    assignments = base.load_assignments(args.fold_csv)
    base.validate_fold_assignments(frame, assignments, n_folds=5)
    folds = base.selected_folds(args.fold)
    base.check_image_paths(frame)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        fold_dir = output_dir / "fold{0:02d}".format(fold)
        fold_dir.mkdir(parents=True, exist_ok=True)
        config = base.make_fold_config(args, frame, assignments, fold, folds)
        config["method"] = method
        base.write_json(fold_dir / "config.json", config)
        print(
            "Fold {0}: method={1} loss={2} sampler={3} policy={4}".format(
                fold,
                method.get("name"),
                args.loss,
                args.sampler,
                args.stage_policy,
            )
        )

    if args.dry_run:
        print("Dry run complete for {0}".format(method.get("name")))
        return

    import tensorflow as tf

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    if hasattr(base, "configure_tensorflow"):
        tf = base.configure_tensorflow(args)
    else:
        try:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as error:  # noqa: BLE001
            print(
                "GPU memory growth warning: {0}".format(error), file=sys.stderr
            )

    images, preprocessing_report = base.load_all_images(
        frame["image_path"].tolist(),
        base.on_off(args.hne_norm),
        args.image_workers,
    )
    auc_class = base.sparse_auc_class(tf)
    for fold in folds:
        config = base.make_fold_config(args, frame, assignments, fold, folds)
        config["method"] = method
        config["preprocessing_report"] = preprocessing_report
        base.write_json(
            Path(args.output_dir) / "fold{0:02d}".format(fold) / "config.json",
            config,
        )
        base.train_fold(
            tf, auc_class, args, frame, assignments, images, fold, config
        )
    print("Method training complete: {0}".format(args.output_dir))


if __name__ == "__main__":
    main()
