"""Training index samplers for HNSCC methodology experiments."""

from __future__ import annotations

import numpy as np


SAMPLER_NAMES = (
    "random",
    "class_balanced",
    "case_balanced",
    "hybrid_class_case",
)


def _validate_inputs(labels, case_ids, seed):
    labels = np.asarray(labels)
    case_ids = np.asarray(case_ids)
    if labels.shape[0] != case_ids.shape[0]:
        raise ValueError("labels and case_ids must have the same length")
    if labels.size == 0:
        raise ValueError("sampler inputs must not be empty")
    return labels, case_ids, np.random.RandomState(seed)


def sample_indices(labels, case_ids, mode, size, seed=42, class_ratio=0.5):
    """Return ``size`` training indices under the requested sampling policy."""
    if mode not in SAMPLER_NAMES:
        raise ValueError("Unknown sampler: {0}".format(mode))
    if size < 1:
        raise ValueError("size must be positive")
    if not (0.0 <= class_ratio <= 1.0):
        raise ValueError("class_ratio must be in [0, 1]")
    labels, case_ids, rng = _validate_inputs(labels, case_ids, seed)
    n = labels.shape[0]
    indices = np.arange(n)

    if mode == "random":
        return rng.choice(indices, size=size, replace=True)

    if mode == "class_balanced":
        classes = np.unique(labels)
        chosen = []
        per_class = int(np.ceil(size / float(len(classes))))
        for label in classes:
            pool = indices[labels == label]
            chosen.append(rng.choice(pool, size=per_class, replace=True))
        selected = np.concatenate(chosen)
        rng.shuffle(selected)
        return selected[:size]

    if mode == "case_balanced":
        cases = np.unique(case_ids)
        chosen = []
        per_case = int(np.ceil(size / float(len(cases))))
        for case_id in cases:
            pool = indices[case_ids == case_id]
            chosen.append(rng.choice(pool, size=per_case, replace=True))
        selected = np.concatenate(chosen)
        rng.shuffle(selected)
        return selected[:size]

    # hybrid: blend class-balanced and case-balanced draws
    class_size = int(round(size * class_ratio))
    case_size = size - class_size
    class_part = sample_indices(
        labels, case_ids, "class_balanced", max(1, class_size), seed=seed
    )
    case_part = sample_indices(
        labels, case_ids, "case_balanced", max(1, case_size), seed=seed + 1
    )
    selected = np.concatenate([class_part[:class_size], case_part[:case_size]])
    rng.shuffle(selected)
    return selected


def make_epoch_indices(labels, case_ids, mode, steps_per_epoch, batch_size, seed=42):
    """Build one epoch of ordered indices for Sequence-style training."""
    size = int(steps_per_epoch) * int(batch_size)
    return sample_indices(labels, case_ids, mode, size=size, seed=seed)
