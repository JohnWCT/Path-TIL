"""Validation-only stage selection policies for HNSCC GroupCV."""

from __future__ import annotations

import math


STAGE_POLICIES = (
    "validation_multiclass_auc",
    "validation_positive_auc",
    "validation_macro_ovr_auc",
    "composite_positive_macro",
    "fixed_stage1",
    "fixed_stage2",
)


def _require_finite(value, name):
    if value is None or not math.isfinite(float(value)):
        raise ValueError("{0} must be a finite number".format(name))
    return float(value)


def score_stage(policy, stage, stage_metrics, keras_auc=None):
    """Return a scalar score for one stage under ``policy`` (higher is better)."""
    if policy not in STAGE_POLICIES:
        raise ValueError("Unknown stage policy: {0}".format(policy))
    stage = int(stage)
    if policy == "fixed_stage1":
        return 1.0 if stage == 1 else -float(stage)
    if policy == "fixed_stage2":
        return 1.0 if stage == 2 else -float(stage)

    validation = stage_metrics["val"]
    if policy == "validation_multiclass_auc":
        if keras_auc is None:
            raise ValueError("validation_multiclass_auc requires keras_auc")
        return _require_finite(keras_auc, "keras_auc")
    if policy == "validation_positive_auc":
        return _require_finite(validation["positive_auc"], "positive_auc")
    if policy == "validation_macro_ovr_auc":
        return _require_finite(validation["macro_ovr_auc"], "macro_ovr_auc")
    if policy == "composite_positive_macro":
        positive = _require_finite(validation["positive_auc"], "positive_auc")
        macro = _require_finite(validation["macro_ovr_auc"], "macro_ovr_auc")
        return 0.7 * positive + 0.3 * macro
    raise ValueError("Unhandled policy: {0}".format(policy))


def select_stage(policy, stage_metrics, validation_keras_auc=None):
    """Select the best stage key under a validation-only policy.

    Tie-break prefers the lower stage index to keep selection deterministic.
    """
    if not stage_metrics:
        raise ValueError("stage_metrics must not be empty")
    scores = {}
    for stage in stage_metrics:
        keras_auc = None
        if validation_keras_auc is not None:
            keras_auc = validation_keras_auc.get(stage)
            if keras_auc is None:
                keras_auc = validation_keras_auc.get(str(stage))
            if keras_auc is None:
                keras_auc = validation_keras_auc.get(int(stage))
        scores[int(stage)] = score_stage(
            policy, stage, stage_metrics[stage], keras_auc=keras_auc
        )
    return max(scores, key=lambda stage: (scores[stage], -stage))


def select_stage_payload(policy, fold_metrics):
    """Select stage from a ``fold_metrics.json`` payload."""
    return select_stage(
        policy,
        fold_metrics["stage_metrics"],
        fold_metrics.get("validation_keras_auc"),
    )
