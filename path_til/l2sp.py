"""L2-SP regularization helpers for anti-forgetting during fine-tuning."""

from __future__ import annotations


def variable_match_key(name: str) -> str:
    """Normalize Keras variable names across save/load prefixes."""
    parts = str(name).split("/")
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return str(name)


def snapshot_weights_by_name(model):
    """Capture a name-keyed frozen copy of all model weights."""
    import tensorflow as tf

    return {
        variable_match_key(variable.name): tf.identity(variable)
        for variable in model.weights
    }


def snapshot_trainable_weights(model):
    """Capture a frozen list of current trainable weights (tests / simple use)."""
    import tensorflow as tf

    return [tf.identity(variable) for variable in model.trainable_variables]


def matched_l2sp_pairs(model, theta_star):
    """Return (current_var, source_tensor) pairs with matching names and shapes.

    Pair selection happens in Python so the result is safe to close over inside
    a compiled Keras loss without graph-mode shape branching.
    """
    if not isinstance(theta_star, dict):
        return list(zip(model.trainable_variables, theta_star))

    pairs = []
    for variable in model.trainable_variables:
        source = theta_star.get(variable_match_key(variable.name))
        if source is None:
            continue
        if tuple(source.shape.as_list()) != tuple(variable.shape.as_list()):
            continue
        pairs.append((variable, source))
    return pairs


def l2sp_penalty(model, theta_star):
    """Sum of squared deviations from the source-domain snapshot."""
    import tensorflow as tf

    penalty = tf.constant(0.0, dtype=tf.float32)
    for current, source in matched_l2sp_pairs(model, theta_star):
        penalty += tf.reduce_sum(tf.square(current - source))
    return penalty


def l2sp_penalty_from_pairs(pairs):
    """Penalty from pre-matched variable pairs (graph-safe)."""
    import tensorflow as tf

    penalty = tf.constant(0.0, dtype=tf.float32)
    for current, source in pairs:
        penalty += tf.reduce_sum(tf.square(current - source))
    return penalty
