from pathlib import Path

import pandas as pd

from path_til.external_eval import (
    build_patch_manifest,
    metrics_to_summary_row,
    patch_evaluation_metrics,
    resolve_stage_name,
)


def test_resolve_stage_name_aliases():
    assert resolve_stage_name("selected") == "selected"
    assert resolve_stage_name("stage1") == "stage1"
    assert resolve_stage_name("1") == "stage1"


def test_build_patch_manifest_on_train_root():
    root = Path("dataset/train")
    if not root.is_dir():
        return
    frame = build_patch_manifest(root, path_prefix="/workspace/dataset/train")
    assert {"case_id", "image_path", "label"}.issubset(frame.columns)
    assert set(frame["label"]).issubset({"positive", "negative", "other"})
    assert len(frame) > 0


def test_patch_evaluation_metrics_shape():
    import numpy as np

    y_true = np.array([0, 1, 2, 0], dtype=np.int32)
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
        ],
        dtype=np.float64,
    )
    metrics = patch_evaluation_metrics(y_true, probabilities)
    assert "positive_auc" in metrics
    assert "positive_prc" in metrics
    assert len(metrics["confusion_matrix"]) == 3
    summary = metrics_to_summary_row("demo", metrics, len(y_true))
    assert summary["n_patches"] == 4
