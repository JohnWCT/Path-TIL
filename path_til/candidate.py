"""Reference candidate settings and reproducibility helpers for HNSCC."""

from __future__ import annotations

from pathlib import Path

from path_til.experiment_registry import CANDIDATE_REFERENCE, keep_or_drop


CANDIDATE_SETTINGS = {
    "backbone": "irv2",
    "initial_weight": "tilscout",
    "pretrained_path": "baselines/best_InceptionResNetV2_model.h5",
    "hne_norm": "off",
    "augmentation": "heavy",
    "class_weight": "on",
    "stage_selection": "validation_multiclass_auc",
    "source_mix_hnscc_ratio": 0.50,
    "source_mix_tcga_ratio": 0.50,
    "primary_metrics": ("positive_auc", "positive_prc"),
}


def reference_metrics():
    """Return the locked OOF reference metrics for the current candidate."""
    return dict(CANDIDATE_REFERENCE)


def evaluate_against_candidate(metrics):
    """Apply keep/drop rules against the locked candidate reference."""
    return keep_or_drop(metrics, reference=CANDIDATE_REFERENCE)


def default_method_config_path(configs_dir="configs"):
    """Return the canonical YAML for the current source-mix candidate."""
    return Path(configs_dir) / "method_source_mix_tcga_r50_50.yaml"
