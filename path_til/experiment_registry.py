"""Registry of HNSCC methodology experiment configs and success rules."""

from __future__ import annotations

from pathlib import Path

import yaml


CANDIDATE_REFERENCE = {
    "name": "candidate_hne_off_heavy_weight_on",
    "positive_auc": 0.8555116064892437,
    "hard_til_mae": 0.14275303565391204,
    "macro_ovr_auc": 0.892156606025242,
    "weighted_ovr_auc": 0.9056483820503611,
}

DEFAULT_METHOD = {
    "method_type": "baseline_candidate",
    "hne_norm": "off",
    "aug": "heavy",
    "class_weight": "on",
    "loss": "weighted_ce",
    "sampler": "random",
    "stage_policy": "validation_multiclass_auc",
    "til_estimator": "hard_raw",
    "epochs_stage1": 30,
    "epochs_stage2": 30,
    "batch_size": 32,
    "seed": 42,
}


def load_method_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Method config must be a mapping: {0}".format(path))
    config = dict(DEFAULT_METHOD)
    config.update(payload)
    config["config_path"] = str(path)
    if "name" not in config:
        config["name"] = path.stem
    return config


def keep_or_drop(metrics, reference=None):
    """Apply the methodology success criteria against the current candidate."""
    reference = reference or CANDIDATE_REFERENCE
    positive_auc = float(metrics["positive_auc"])
    hard_til_mae = float(metrics["hard_til_mae"])
    macro_ovr_auc = float(metrics["macro_ovr_auc"])
    weighted_ovr_auc = float(metrics["weighted_ovr_auc"])
    reasons = []
    if positive_auc <= reference["positive_auc"]:
        reasons.append("positive_auc_not_improved")
    if hard_til_mae >= reference["hard_til_mae"]:
        reasons.append("hard_til_mae_not_improved")
    if macro_ovr_auc < reference["macro_ovr_auc"] - 1e-8:
        reasons.append("macro_ovr_auc_decreased")
    if weighted_ovr_auc < reference["weighted_ovr_auc"] - 0.01:
        reasons.append("weighted_ovr_auc_clearly_decreased")
    decision = "keep" if not reasons else "drop"
    return {
        "decision": decision,
        "reasons": reasons,
        "delta_positive_auc": positive_auc - reference["positive_auc"],
        "delta_hard_til_mae": hard_til_mae - reference["hard_til_mae"],
    }


def list_config_paths(configs_dir):
    root = Path(configs_dir)
    return sorted(root.glob("method_*.yaml"))
