#!/usr/bin/env python3
"""Fine-tune source-pretrained backbones on HNSCC GroupCV with TCGA source mix."""

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
import yaml

from path_til.experiment_registry import load_method_config  # noqa: E402
from path_til.hnscc import LABELS, balanced_class_weights  # noqa: E402
from path_til.model_factory import build_classifier  # noqa: E402
from scripts import train_hnscc_source_mix as source_mix  # noqa: E402
from scripts._eval_patch_manifest import load_train_base, setup_tensorflow  # noqa: E402


LABEL_TO_IDX = {label: index for index, label in enumerate(LABELS)}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test or train replacement backbones with HNSCC GroupCV + source mix."
        )
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--csv-hnscc", default="qupath_dataset.csv")
    parser.add_argument("--csv-tcga", default="tcga_train_dataset.csv")
    parser.add_argument("--fold-csv", default="folds_hnscc_group5.csv")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--source-mix-ratio", type=float, default=0.50)
    parser.add_argument("--aug", default="heavy")
    parser.add_argument("--hne-norm", choices=("on", "off"), default="off")
    parser.add_argument("--class-weight", choices=("on", "off"), default="on")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_settings(cli):
    if cli.config:
        payload = load_yaml_config(Path(cli.config))
        model = payload.get("model", {})
        data = payload.get("data", {})
        training = payload.get("training", {})
        preprocessing = payload.get("preprocessing", {})
        return {
            "backbone": model.get("backbone", cli.backbone),
            "pretrained": model.get("pretrained", cli.pretrained),
            "dropout": float(model.get("dropout", 0.3)),
            "label_smoothing": float(model.get("label_smoothing", 0.0)),
            "csv_hnscc": data.get("csv_hnscc", cli.csv_hnscc),
            "csv_tcga": data.get("csv_tcga", cli.csv_tcga),
            "fold_csv": data.get("fold_csv", cli.fold_csv),
            "hnscc_ratio": float(data.get("source_mix_ratio", cli.source_mix_ratio)),
            "tcga_ratio": 1.0 - float(data.get("source_mix_ratio", cli.source_mix_ratio)),
            "aug": preprocessing.get("augmentation", cli.aug),
            "hne_norm": preprocessing.get("hne_norm", cli.hne_norm == "on"),
            "class_weight": training.get("class_weight", cli.class_weight == "on"),
            "learning_rate": float(training.get("learning_rate", 1e-4)),
            "weight_decay": float(training.get("weight_decay", 1e-4)),
            "batch_size": int(training.get("batch_size", 32)),
            "epochs_stage1": int(training.get("epochs_stage1", 20)),
            "epochs_stage2": int(training.get("epochs_stage2", 30)),
            "seed": int(training.get("seed", 42)),
        }
    if cli.backbone is None or cli.pretrained is None:
        raise ValueError("Provide --config or both --backbone and --pretrained")
    tcga_ratio = 1.0 - cli.source_mix_ratio
    return {
        "backbone": cli.backbone,
        "pretrained": cli.pretrained,
        "dropout": 0.3,
        "label_smoothing": 0.0,
        "csv_hnscc": cli.csv_hnscc,
        "csv_tcga": cli.csv_tcga,
        "fold_csv": cli.fold_csv,
        "hnscc_ratio": cli.source_mix_ratio,
        "tcga_ratio": tcga_ratio,
        "aug": cli.aug,
        "hne_norm": cli.hne_norm == "on",
        "class_weight": cli.class_weight == "on",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs_stage1": 20,
        "epochs_stage2": 30,
        "seed": 42,
    }


def train_backbone_fold(tf, base, settings, mixed, assignments, images, fold, output_dir):
    indices = source_mix.base.split_indices(mixed, assignments, fold)
    labels = mixed["label"].map(LABEL_TO_IDX).to_numpy(dtype=np.int32)
    class_weights = (
        balanced_class_weights(mixed.iloc[indices["train"]]["label"].tolist())
        if settings["class_weight"]
        else None
    )
    validation = base.nonaug_sequence(
        tf, images[indices["val"]], labels[indices["val"]], settings["batch_size"]
    )
    fold_dir = output_dir / "fold{0:02d}".format(fold)
    fold_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(settings["pretrained"])
    if model_path.is_file():
        model = tf.keras.models.load_model(str(model_path), compile=False)
    else:
        model = build_classifier(
            settings["backbone"],
            num_classes=len(LABELS),
            weights=None,
            dropout=settings["dropout"],
            train_backbone=False,
        )
        model.load_weights(settings["pretrained"])
    auc_class = base.sparse_auc_class(tf)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(settings["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            auc_class(len(LABELS), name="auc"),
        ],
    )
    augmentation = base.build_tf_augmentation(tf, settings["aug"])
    train_data = base.make_tf_dataset(
        tf,
        images[indices["train"]],
        labels[indices["train"]],
        settings["batch_size"],
        True,
        augmentation,
        settings["seed"] + fold,
    )
    stage1_path = fold_dir / "stage1_best.h5"
    history1 = model.fit(
        train_data,
        validation_data=validation,
        epochs=settings["epochs_stage1"],
        callbacks=base.callbacks(tf, stage1_path, patience=8),
        class_weight=class_weights,
        verbose=1,
    )
    base.save_learning_curve(history1, fold_dir / "stage1_learning_curve.png", "Fold {0}".format(fold))
    model = tf.keras.models.load_model(stage1_path, compile=False)
    model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(settings["learning_rate"] * 0.1),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            auc_class(len(LABELS), name="auc"),
        ],
    )
    train_data = base.make_tf_dataset(
        tf,
        images[indices["train"]],
        labels[indices["train"]],
        settings["batch_size"],
        True,
        augmentation,
        settings["seed"] + 1000 + fold,
    )
    stage2_path = fold_dir / "stage2_best.h5"
    history2 = model.fit(
        train_data,
        validation_data=validation,
        epochs=settings["epochs_stage2"],
        callbacks=base.callbacks(tf, stage2_path, patience=8),
        class_weight=class_weights,
        verbose=1,
    )
    base.save_learning_curve(history2, fold_dir / "stage2_learning_curve.png", "Fold {0} stage2".format(fold))

    stage_metrics = {}
    validation_keras_auc = {}
    for stage, path in ((1, stage1_path), (2, stage2_path)):
        stage_model = tf.keras.models.load_model(path, compile=False)
        stage_model.compile(
            optimizer=tf.keras.optimizers.Adam(settings["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=[auc_class(len(LABELS), name="auc")],
        )
        result = stage_model.evaluate(validation, verbose=0, return_dict=True)
        validation_keras_auc[stage] = float(result["auc"])
        metrics = {}
        for split_name, split_idx in (("val", indices["val"]), ("test", indices["test"])):
            split_labels = labels[split_idx]
            split_data = base.nonaug_sequence(
                tf, images[split_idx], split_labels, settings["batch_size"]
            )
            probs = np.asarray(stage_model.predict(split_data, verbose=0), dtype=np.float64)
            metrics[split_name] = source_mix.base.classification_metrics(split_labels, probs)
        stage_metrics[stage] = metrics
    selected_stage = max(validation_keras_auc, key=lambda stage: (validation_keras_auc[stage], -stage))
    payload = {
        "fold": fold,
        "selected_stage": selected_stage,
        "validation_keras_auc": validation_keras_auc,
        "stage_metrics": stage_metrics,
        "selected_metrics": stage_metrics[selected_stage],
        "backbone": settings["backbone"],
    }
    with (fold_dir / "fold_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    config = {
        "parameters": settings,
        "fold": fold,
        "preprocessing": {"hne_norm": settings["hne_norm"]},
    }
    with (fold_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tf.keras.backend.clear_session()


def main():
    cli = parse_args()
    settings = resolve_settings(cli)
    source_mix.install_source_mix_split()

    hnscc = source_mix.base.load_hnscc_csv(settings["csv_hnscc"], expected_cases=10)
    assignments = source_mix.base.load_assignments(settings["fold_csv"])
    source_mix.base.validate_fold_assignments(hnscc, assignments, n_folds=5)
    tcga = source_mix.load_tcga_csv(settings["csv_tcga"])
    mixed, sampled = source_mix.build_mixed_frame(
        hnscc,
        tcga,
        settings["hnscc_ratio"],
        settings["tcga_ratio"],
        settings["seed"],
    )
    folds = cli.folds if cli.folds is not None else list(range(5))
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_mix.base.write_json(
        output_dir / "backbone_source_mix_summary.json",
        {"settings": settings, "folds": folds},
    )
    if cli.dry_run:
        print("Backbone source-mix dry run complete -> {0}".format(output_dir))
        return

    base = load_train_base()
    tf = setup_tensorflow()
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    tf.random.set_seed(settings["seed"])
    images, _ = base.load_all_images(
        mixed["image_path"].tolist(),
        settings["hne_norm"],
        min(8, max(1, __import__("os").cpu_count() or 1)),
    )
    for fold in folds:
        train_backbone_fold(tf, base, settings, mixed, assignments, images, fold, output_dir)
    print("Backbone source-mix training complete -> {0}".format(output_dir))


if __name__ == "__main__":
    main()
