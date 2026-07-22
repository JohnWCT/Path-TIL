"""Decision rules for backbone smoke / B5 / full5 promotion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneMetrics:
    name: str
    positive_auc: float
    positive_prc: float
    macro_ovr_auc: float
    weighted_ovr_auc: float
    fold_auc_gap: float | None = None
    fold_prc_gap: float | None = None


@dataclass(frozen=True)
class BackboneDecision:
    decision: str
    reasons: list[str]


def decide_backbone_status(
    candidate: BackboneMetrics,
    reference: BackboneMetrics,
    weighted_drop_tolerance: float = 0.01,
    macro_drop_tolerance: float = 0.0,
    fold_gap_warning: float = 0.05,
    fold_prc_gap_warning: float = 0.08,
) -> BackboneDecision:
    """Classify a backbone run relative to the locked IRV2 candidate."""
    reasons: list[str] = []

    positive_improved = (
        candidate.positive_auc > reference.positive_auc
        and candidate.positive_prc > reference.positive_prc
    )
    macro_ok = candidate.macro_ovr_auc >= reference.macro_ovr_auc - macro_drop_tolerance
    weighted_ok = (
        candidate.weighted_ovr_auc >= reference.weighted_ovr_auc - weighted_drop_tolerance
    )

    if not positive_improved:
        reasons.append("positive_auc_or_prc_not_improved")
    if not macro_ok:
        reasons.append("macro_ovr_auc_decreased")
    if not weighted_ok:
        reasons.append("weighted_ovr_auc_clearly_decreased")

    fold_gap_fail = False
    if candidate.fold_auc_gap is not None and candidate.fold_auc_gap > fold_gap_warning:
        reasons.append("fold_auc_gap_high")
        fold_gap_fail = True
    if (
        candidate.fold_prc_gap is not None
        and candidate.fold_prc_gap > fold_prc_gap_warning
    ):
        reasons.append("fold_prc_gap_high")
        fold_gap_fail = True

    if positive_improved and macro_ok and weighted_ok and not fold_gap_fail:
        return BackboneDecision("replace_candidate", reasons)
    if positive_improved:
        return BackboneDecision("positive_specialist_pending_full5", reasons)
    return BackboneDecision("drop", reasons)


def is_forbidden_smoke_promotion(phase: str, decision: str) -> bool:
    """True when a fold-0+1 smoke run tries to become the formal candidate."""
    return phase in {"B4", "smoke"} and decision == "replace_candidate"
