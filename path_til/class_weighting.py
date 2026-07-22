"""Class-weight helpers for backbone repair experiments."""

from __future__ import annotations

from typing import Dict


def scale_class_weight(
    class_weight: Dict[int, float],
    positive_class_index: int = 0,
    positive_scale: float = 1.0,
) -> Dict[int, float]:
    """Return a copied class-weight dict with optional positive-class scaling."""
    scaled = dict(class_weight)
    if positive_class_index in scaled:
        scaled[positive_class_index] = scaled[positive_class_index] * positive_scale
    return scaled
