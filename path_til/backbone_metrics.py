"""Helpers for aggregating backbone smoke / B5 fold metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_fold_test_metrics(pred_dir: Path, fold: int) -> dict:
    metrics_path = Path(pred_dir) / "fold{0:02d}".format(fold) / "fold_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError("Missing fold metrics: {0}".format(metrics_path))
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_metrics", {})
    test_metrics = selected.get("test")
    if not isinstance(test_metrics, dict):
        raise ValueError("fold {0} missing selected_metrics.test".format(fold))
    return test_metrics


def average_metric(rows: list[dict], key: str):
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def fold_gap(rows: list[dict], key: str):
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if len(values) < 2:
        return None
    return float(abs(values[0] - values[1]))


def summarize_smoke_folds(
    pred_dir: Path,
    folds: list[int] | None = None,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Average fold metrics and compute fold0/fold1 gaps for B5 selection."""
    folds = list(folds) if folds is not None else [0, 1]
    pred_dir = Path(pred_dir)
    fold_metrics = [load_fold_test_metrics(pred_dir, fold) for fold in folds]
    name = experiment_name or pred_dir.name
    summary = {
        "experiment": name,
        "path": str(pred_dir),
        "smoke_folds": folds,
        "positive_auc": average_metric(fold_metrics, "positive_auc"),
        "positive_prc": average_metric(fold_metrics, "positive_prc"),
        "macro_ovr_auc": average_metric(fold_metrics, "macro_ovr_auc"),
        "weighted_ovr_auc": average_metric(fold_metrics, "weighted_ovr_auc"),
        "negative_auc": average_metric(fold_metrics, "negative_auc"),
        "other_auc": average_metric(fold_metrics, "other_auc"),
        "fold0_positive_auc": fold_metrics[0].get("positive_auc") if fold_metrics else None,
        "fold1_positive_auc": (
            fold_metrics[1].get("positive_auc") if len(fold_metrics) > 1 else None
        ),
        "fold0_positive_prc": fold_metrics[0].get("positive_prc") if fold_metrics else None,
        "fold1_positive_prc": (
            fold_metrics[1].get("positive_prc") if len(fold_metrics) > 1 else None
        ),
        "fold_auc_gap": fold_gap(fold_metrics, "positive_auc"),
        "fold_prc_gap": fold_gap(fold_metrics, "positive_prc"),
        "hard_til_mae": None,
        "per_fold_test_metrics": {
            str(fold): metrics for fold, metrics in zip(folds, fold_metrics)
        },
    }
    return summary


def required_full5_folds() -> set[int]:
    return {0, 1, 2, 3, 4}


def has_all_full5_folds(observed_folds) -> bool:
    return set(int(fold) for fold in observed_folds) == required_full5_folds()
