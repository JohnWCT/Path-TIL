#!/usr/bin/env python3
"""Shared inference helpers for TCGA internal and external lock-box evaluation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from path_til.external_eval import patch_evaluation_metrics, resolve_fold_model_paths
from path_til.hnscc import LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_TO_IDX = {label: index for index, label in enumerate(LABELS)}


def load_train_base():
    path = PROJECT_ROOT / "scripts" / "train_hnscc_groupcv_irv2.py"
    spec = importlib.util.spec_from_file_location("train_hnscc_groupcv_irv2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def setup_tensorflow():
    import tensorflow as tf

    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as error:  # noqa: BLE001
        print("GPU setup warning: {0}".format(error), file=sys.stderr)
    return tf


def resolve_disk_paths(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Map Docker-style manifest paths back to local filesystem paths when needed."""
    root = Path(root)
    resolved = []
    for path in frame["image_path"].tolist():
        candidate = Path(str(path))
        if candidate.is_file():
            resolved.append(str(candidate))
            continue
        parts = candidate.parts
        if len(parts) >= 2:
            local = root / parts[-2] / parts[-1]
            if local.is_file():
                resolved.append(str(local))
                continue
        local = root / candidate.name
        if local.is_file():
            resolved.append(str(local))
            continue
        raise FileNotFoundError("Unable to resolve image path: {0}".format(path))
    copy = frame.copy()
    copy["disk_path"] = resolved
    return copy


def load_manifest_images(base, frame: pd.DataFrame, use_hne_norm: bool, workers: int):
    disk_paths = frame["disk_path"].tolist()
    images, report = base.load_all_images(disk_paths, use_hne_norm, workers)
    return np.asarray(images, dtype=np.uint8), report


def load_classifier_model(tf, base, model_path: str):
    """Load IRV2 checkpoints or generic backbone classifiers."""
    try:
        return base.load_and_validate_model(tf, model_path, mixed_precision=False)
    except ValueError:
        model = tf.keras.models.load_model(model_path, compile=False)
        output_shape = model.output_shape
        if isinstance(output_shape, list) or int(output_shape[-1]) != len(LABELS):
            raise ValueError(
                "Unsupported classifier output shape for {0}: {1}".format(
                    model_path, output_shape
                )
            )
        return model


def predict_with_models(tf, base, models, images, batch_size: int) -> np.ndarray:
    labels = np.zeros(len(images), dtype=np.int32)
    probabilities = []
    for fold, model_path in models:
        print("Predicting with fold {0}: {1}".format(fold, model_path))
        model = load_classifier_model(tf, base, str(model_path))
        sequence = base.nonaug_sequence(tf, images, labels, batch_size)
        fold_probs = np.asarray(model.predict(sequence, verbose=1), dtype=np.float64)
        probabilities.append(fold_probs)
        del model
        tf.keras.backend.clear_session()
    return np.mean(np.stack(probabilities, axis=0), axis=0)


def build_prediction_frame(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    y_true = frame["label"].map(LABEL_TO_IDX).to_numpy(dtype=np.int32)
    y_pred = probabilities.argmax(axis=1).astype(np.int32)
    return pd.DataFrame(
        {
            "case_id": frame["case_id"].astype(str),
            "image_path": frame["image_path"].astype(str),
            "dataset": frame.get("dataset", frame["case_id"]).astype(str),
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


def evaluate_manifest(
    model_dir: Path,
    manifest: pd.DataFrame,
    data_root: Path,
    stage: str,
    batch_size: int,
    hne_norm: bool | None,
    image_workers: int,
):
    base = load_train_base()
    tf = setup_tensorflow()
    models = resolve_fold_model_paths(model_dir, stage=stage)
    if hne_norm is None:
        from path_til.external_eval import read_training_hne_norm

        hne_norm = read_training_hne_norm(model_dir)
    frame = resolve_disk_paths(manifest, data_root)
    images, report = load_manifest_images(base, frame, hne_norm, image_workers)
    probabilities = predict_with_models(tf, base, models, images, batch_size)
    predictions = build_prediction_frame(frame, probabilities)
    metrics = patch_evaluation_metrics(
        predictions["y_true_idx"].to_numpy(),
        probabilities,
    )
    return metrics, predictions, report, [str(path) for _, path in models]
