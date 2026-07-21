#!/usr/bin/env python3
"""Source-domain pretraining for replacement backbones on dataset/train."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from path_til.external_eval import patch_evaluation_metrics  # noqa: E402
from path_til.gpu_profile import detect_gpu_profile  # noqa: E402
from path_til.hnscc import LABELS, balanced_class_weights  # noqa: E402
from path_til.model_factory import build_classifier, load_classifier_from_checkpoint  # noqa: E402
from path_til.source_pretrain import (  # noqa: E402
    load_source_pretrain_config,
    output_artifact_paths,
    validate_no_testset_leakage,
)
from scripts._eval_patch_manifest import (  # noqa: E402
    build_prediction_frame,
    load_train_base,
    resolve_disk_paths,
    setup_tensorflow,
)


LABEL_TO_IDX = {label: index for index, label in enumerate(LABELS)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain a backbone on TCGA source train and validate on dataset/test."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--finish-only",
        action="store_true",
        help="Skip training; write validation metrics from an existing best checkpoint",
    )
    return parser.parse_args()


def read_csv(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"image_path", "label", "case_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("{0} missing columns: {1}".format(path, missing))
    return frame


def load_images(base, frame: pd.DataFrame, data_root: Path, use_hne_norm: bool, workers: int):
    disk = resolve_disk_paths(frame, data_root)
    return base.load_all_images(
        disk["disk_path"].tolist(),
        use_hne_norm,
        workers,
    )


def make_dataset(tf, base, images, labels, batch_size, training, aug_level, seed):
    augmentation = base.build_tf_augmentation(tf, aug_level) if training else None
    return base.make_tf_dataset(
        tf, images, labels, batch_size, training, augmentation, seed
    )


def train_source_model(tf, base, model, train_data, val_data, epochs, learning_rate, class_weights, checkpoint_path, patience):
    auc_class = base.sparse_auc_class(tf)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            auc_class(len(LABELS), name="auc"),
        ],
    )
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=base.callbacks(tf, checkpoint_path, patience),
        class_weight=class_weights,
        verbose=1,
    )
    return history


def main():
    cli = parse_args()
    config = load_source_pretrain_config(cli.config)
    validate_no_testset_leakage(config)
    profile = detect_gpu_profile()
    batch_size = (
        cli.batch_size
        if cli.batch_size is not None
        else int(config["training"].get("batch_size", profile["batch_size_pretrain"]))
    )
    workers = (
        cli.image_workers
        if cli.image_workers is not None
        else profile["image_workers"]
    )
    train_csv = cli.train_csv or config["data"]["train_csv"]
    val_csv = cli.val_csv or config["data"]["val_csv"]
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = output_artifact_paths(output_dir, config["model"]["backbone"])
    with artifacts["config_snapshot"].open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    train = read_csv(train_csv)
    val = read_csv(val_csv)
    print("Train rows={0}, val rows={1}".format(len(train), len(val)))
    if cli.dry_run:
        print("Source pretrain dry run complete -> {0}".format(output_dir))
        return

    finish_only = cli.finish_only or (
        artifacts["best"].is_file() and not artifacts["val_metrics"].is_file()
    )
    if finish_only and not artifacts["best"].is_file():
        raise FileNotFoundError("finish-only requested but missing checkpoint: {0}".format(artifacts["best"]))

    base = load_train_base()
    tf = setup_tensorflow()
    hne_norm = bool(config.get("preprocessing", {}).get("hne_norm", False))
    aug = config.get("preprocessing", {}).get("augmentation", "heavy")
    seed = int(config.get("training", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_root = Path(config["data"].get("train_root", "dataset/train"))
    val_root = Path(config["data"].get("val_root", "dataset/test"))
    train_images, _ = load_images(base, train, train_root, hne_norm, workers)
    val_images, _ = load_images(base, val, val_root, hne_norm, workers)
    train_labels = train["label"].map(LABEL_TO_IDX).to_numpy(dtype=np.int32)
    val_labels = val["label"].map(LABEL_TO_IDX).to_numpy(dtype=np.int32)
    class_weights = (
        balanced_class_weights(train["label"].tolist())
        if config["training"].get("class_weight", True)
        else None
    )

    backbone = config["model"]["backbone"]
    dropout = float(config["model"].get("dropout", 0.3))
    weights = "imagenet" if config["model"].get("imagenet_init", True) else None
    val_data = make_dataset(
        tf, base, val_images, val_labels, batch_size, False, "none", seed
    )
    if finish_only:
        print("Source pretrain finish-only: evaluating existing checkpoint")
        best = load_classifier_from_checkpoint(
            tf, backbone, str(artifacts["best"]), num_classes=len(LABELS), dropout=dropout
        )
    else:
        model = build_classifier(
            backbone,
            num_classes=len(LABELS),
            weights=weights,
            dropout=dropout,
            train_backbone=False,
        )
        train_data = make_dataset(
            tf, base, train_images, train_labels, batch_size, True, aug, seed
        )
        history1 = train_source_model(
            tf,
            base,
            model,
            train_data,
            val_data,
            int(config["training"]["epochs_head"]),
            float(config["training"]["learning_rate_head"]),
            class_weights,
            output_dir / "head_best.h5",
            int(config["training"].get("patience", 8)),
        )
        # Phase-1 uses restore_best_weights=True, so the in-memory model already
        # holds the best head checkpoint and can continue directly to phase-2.
        model.trainable = True
        for layer in model.layers:
            if hasattr(layer, "trainable"):
                layer.trainable = True
        train_data = make_dataset(
            tf, base, train_images, train_labels, batch_size, True, aug, seed + 1
        )
        history2 = train_source_model(
            tf,
            base,
            model,
            train_data,
            val_data,
            int(config["training"]["epochs_finetune"]),
            float(config["training"]["learning_rate_finetune"]),
            class_weights,
            artifacts["best"],
            int(config["training"].get("patience", 8)),
        )
        model.save(artifacts["last"], include_optimizer=False)
        base.save_learning_curve(
            history1,
            output_dir / "source_head_learning_curve.png",
            "Source pretrain head",
        )
        base.save_learning_curve(
            history2,
            artifacts["learning_curve"],
            "Source pretrain finetune",
        )
        log_rows = []
        for name, history in (("head", history1), ("finetune", history2)):
            for epoch, loss in enumerate(history.history.get("loss", []), start=1):
                log_rows.append(
                    {
                        "phase": name,
                        "epoch": epoch,
                        "loss": loss,
                        "val_loss": history.history.get("val_loss", [None])[epoch - 1],
                        "auc": history.history.get("auc", [None])[epoch - 1],
                        "val_auc": history.history.get("val_auc", [None])[epoch - 1],
                    }
                )
        pd.DataFrame(log_rows).to_csv(artifacts["training_log"], index=False)
        # EarlyStopping(restore_best_weights=True) keeps the best weights in memory.
        best = model
    val_sequence = make_dataset(
        tf, base, val_images, val_labels, batch_size, False, "none", seed
    )
    probabilities = np.asarray(best.predict(val_sequence, verbose=1), dtype=np.float64)
    predictions = build_prediction_frame(val, probabilities)
    predictions.to_csv(artifacts["val_predictions"], index=False)
    metrics = patch_evaluation_metrics(val_labels, probabilities)
    with artifacts["val_metrics"].open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("Source pretraining complete -> {0}".format(output_dir))


if __name__ == "__main__":
    main()
