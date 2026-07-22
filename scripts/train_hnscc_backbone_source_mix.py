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
import pandas as pd
import yaml

from path_til.class_weighting import scale_class_weight  # noqa: E402
from path_til.hnscc import LABELS, balanced_class_weights  # noqa: E402
from path_til.model_factory import build_classifier, load_classifier_from_checkpoint  # noqa: E402
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
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-workers", type=int, default=None)
    parser.add_argument("--fit-workers", type=int, default=None)
    parser.add_argument("--use-multiprocessing", choices=("on", "off"), default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    inherit = payload.get("inherit_from")
    if inherit:
        parent = load_yaml_config(Path(inherit))
        merged = dict(parent)
        for key, value in payload.items():
            if key == "inherit_from":
                continue
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                child = dict(merged[key])
                child.update(value)
                merged[key] = child
            else:
                merged[key] = value
        return merged
    return payload


def resolve_source_mix(data: dict, default_ratio: float) -> tuple[float, float]:
    source_mix = data.get("source_mix")
    if isinstance(source_mix, dict):
        hnscc = float(source_mix.get("hnscc", default_ratio))
        tcga = float(source_mix.get("tcga", 1.0 - hnscc))
        return hnscc, tcga
    ratio = float(data.get("source_mix_ratio", default_ratio))
    return ratio, 1.0 - ratio


def resolve_settings(cli):
    if cli.config:
        payload = load_yaml_config(Path(cli.config))
        model = payload.get("model", {})
        data = payload.get("data", {})
        training = payload.get("training", {})
        preprocessing = payload.get("preprocessing", {})
        hnscc_ratio, tcga_ratio = resolve_source_mix(data, cli.source_mix_ratio)
        stage1_lr = float(
            training.get(
                "stage1_lr",
                training.get("learning_rate", 1e-4),
            )
        )
        stage2_lr = float(
            training.get(
                "stage2_lr",
                stage1_lr * 0.1,
            )
        )
        return {
            "backbone": model.get("backbone", cli.backbone),
            "pretrained": model.get("pretrained", cli.pretrained),
            "dropout": float(model.get("dropout", 0.3)),
            "label_smoothing": float(model.get("label_smoothing", 0.0)),
            "csv_hnscc": data.get("csv_hnscc", cli.csv_hnscc),
            "csv_tcga": data.get("csv_tcga", cli.csv_tcga),
            "fold_csv": data.get("fold_csv", cli.fold_csv),
            "hnscc_ratio": hnscc_ratio,
            "tcga_ratio": tcga_ratio,
            "aug": preprocessing.get("augmentation", cli.aug),
            "hne_norm": bool(preprocessing.get("hne_norm", cli.hne_norm == "on")),
            "class_weight": bool(training.get("class_weight", cli.class_weight == "on")),
            "positive_weight_scale": float(training.get("positive_weight_scale", 1.0)),
            "stage1_lr": stage1_lr,
            "stage2_lr": stage2_lr,
            "learning_rate": stage1_lr,
            "weight_decay": float(training.get("weight_decay", 1e-4)),
            "batch_size": int(training.get("batch_size", 32)),
            "epochs_stage1": int(training.get("epochs_stage1", 20)),
            "epochs_stage2": int(training.get("epochs_stage2", 30)),
            "stage_selection_metric": training.get(
                "stage_selection_metric", "val_multiclass_auc"
            ),
            "seed": int(training.get("seed", 42)),
            "experiment_name": payload.get("experiment", {}).get("name"),
            "phase": payload.get("experiment", {}).get("phase"),
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
        "positive_weight_scale": 1.0,
        "stage1_lr": 1e-4,
        "stage2_lr": 1e-5,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs_stage1": 20,
        "epochs_stage2": 30,
        "stage_selection_metric": "val_multiclass_auc",
        "seed": 42,
        "experiment_name": None,
        "phase": None,
    }


def build_loss(tf, label_smoothing: float):
    if label_smoothing and label_smoothing > 0.0:
        categorical = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=float(label_smoothing)
        )

        def loss_fn(y_true, y_pred):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_true = tf.one_hot(y_true, depth=len(LABELS))
            return categorical(y_true, y_pred)

        return loss_fn
    return "sparse_categorical_crossentropy"


def prediction_frame(frame, indices, fold, split, probabilities):
    subset = frame.iloc[indices].reset_index(drop=True)
    y_true = subset["label"].map(LABEL_TO_IDX).to_numpy(dtype=np.int32)
    y_pred = np.argmax(probabilities, axis=1).astype(np.int32)
    result = pd.DataFrame(
        {
            "patch_id": subset["image_path"],
            "case_id": subset["case_id"],
            "image_path": subset["image_path"],
            "fold": int(fold),
            "split": split,
            "y_true_idx": y_true,
            "y_true_label": [LABELS[index] for index in y_true],
            "y_pred_idx": y_pred,
            "y_pred_label": [LABELS[index] for index in y_pred],
            "prob_positive": probabilities[:, 0],
            "prob_negative": probabilities[:, 1],
            "prob_other": probabilities[:, 2],
            "confidence": np.max(probabilities, axis=1),
            "correct": y_true == y_pred,
        }
    )
    return result


def select_stage(stage_metrics, validation_keras_auc, metric_name):
    if metric_name == "composite_positive_macro":
        scores = {}
        for stage, metrics in stage_metrics.items():
            val = metrics.get("val", {})
            positive = float(val.get("positive_auc") or 0.0)
            macro = float(val.get("macro_ovr_auc") or 0.0)
            scores[stage] = 0.5 * positive + 0.5 * macro
        return max(scores, key=lambda stage: (scores[stage], -stage))
    return max(
        validation_keras_auc, key=lambda stage: (validation_keras_auc[stage], -stage)
    )


def train_backbone_fold(tf, base, settings, mixed, assignments, images, fold, output_dir):
    indices = source_mix.base.split_indices(mixed, assignments, fold)
    labels = mixed["label"].map(LABEL_TO_IDX).to_numpy(dtype=np.int32)
    class_weights = None
    if settings["class_weight"]:
        class_weights = balanced_class_weights(
            mixed.iloc[indices["train"]]["label"].tolist()
        )
        class_weights = scale_class_weight(
            class_weights,
            positive_class_index=LABEL_TO_IDX["positive"],
            positive_scale=float(settings.get("positive_weight_scale", 1.0)),
        )
    validation = base.nonaug_sequence(
        tf, images[indices["val"]], labels[indices["val"]], settings["batch_size"]
    )
    fold_dir = output_dir / "fold{0:02d}".format(fold)
    fold_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(settings["pretrained"])
    if model_path.is_file():
        model = load_classifier_from_checkpoint(
            tf,
            settings["backbone"],
            str(model_path),
            num_classes=len(LABELS),
            dropout=settings["dropout"],
        )
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
    loss = build_loss(tf, settings.get("label_smoothing", 0.0))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(settings["stage1_lr"]),
        loss=loss,
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
    base.save_learning_curve(
        history1, fold_dir / "stage1_learning_curve.png", "Fold {0}".format(fold)
    )
    model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(settings["stage2_lr"]),
        loss=loss,
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
    base.save_learning_curve(
        history2,
        fold_dir / "stage2_learning_curve.png",
        "Fold {0} stage2".format(fold),
    )

    stage_metrics = {}
    validation_keras_auc = {}
    stage_probabilities = {}
    for stage, path in ((1, stage1_path), (2, stage2_path)):
        if stage == 2:
            stage_model = model
        else:
            stage_model = load_classifier_from_checkpoint(
                tf,
                settings["backbone"],
                str(path),
                num_classes=len(LABELS),
                dropout=settings["dropout"],
            )
        stage_model.compile(
            optimizer=tf.keras.optimizers.Adam(settings["stage1_lr"]),
            loss=loss,
            metrics=[auc_class(len(LABELS), name="auc")],
        )
        result = stage_model.evaluate(validation, verbose=0, return_dict=True)
        validation_keras_auc[stage] = float(result["auc"])
        metrics = {}
        probs_by_split = {}
        for split_name, split_idx in (("val", indices["val"]), ("test", indices["test"])):
            split_labels = labels[split_idx]
            split_data = base.nonaug_sequence(
                tf, images[split_idx], split_labels, settings["batch_size"]
            )
            probs = np.asarray(stage_model.predict(split_data, verbose=0), dtype=np.float64)
            probs_by_split[split_name] = probs
            metrics[split_name] = source_mix.base.classification_metrics(
                split_labels, probs
            )
        stage_metrics[stage] = metrics
        stage_probabilities[stage] = probs_by_split
        prediction_frame(
            mixed, indices["test"], fold, "test", probs_by_split["test"]
        ).to_csv(
            fold_dir / "stage{0}_test_predictions.csv".format(stage),
            index=False,
            float_format="%.8f",
        )

    selected_stage = select_stage(
        stage_metrics,
        validation_keras_auc,
        settings.get("stage_selection_metric", "val_multiclass_auc"),
    )
    selected_probs = stage_probabilities[selected_stage]["test"]
    prediction_frame(mixed, indices["test"], fold, "test", selected_probs).to_csv(
        fold_dir / "test_predictions.csv",
        index=False,
        float_format="%.8f",
    )
    payload = {
        "fold": fold,
        "selected_stage": selected_stage,
        "validation_keras_auc": validation_keras_auc,
        "stage_metrics": stage_metrics,
        "selected_metrics": stage_metrics[selected_stage],
        "backbone": settings["backbone"],
        "stage_selection_metric": settings.get("stage_selection_metric"),
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
    from path_til.gpu_profile import detect_gpu_profile  # noqa: E402

    profile = detect_gpu_profile()
    workers = (
        cli.image_workers
        if cli.image_workers is not None
        else profile["image_workers"]
    )
    if cli.batch_size is not None:
        settings["batch_size"] = cli.batch_size
    else:
        settings["batch_size"] = profile["batch_size_train_backbone"]
    source_mix.base.write_json(
        output_dir / "backbone_source_mix_summary.json",
        {"settings": settings, "folds": folds, "sampled_tcga_rows": int(len(sampled))},
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
        workers,
    )
    for fold in folds:
        marker = output_dir / "fold{0:02d}".format(fold) / "fold_metrics.json"
        if marker.is_file():
            print("Skip existing fold {0}: {1}".format(fold, marker))
            continue
        train_backbone_fold(
            tf, base, settings, mixed, assignments, images, fold, output_dir
        )
    print("Backbone source-mix training complete -> {0}".format(output_dir))


if __name__ == "__main__":
    main()
