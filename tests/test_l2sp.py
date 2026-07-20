import unittest

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - host without TF
    tf = None

from path_til.l2sp import l2sp_penalty, snapshot_trainable_weights


@unittest.skipIf(tf is None, "tensorflow not installed")
class L2SPTests(unittest.TestCase):
    def test_l2sp_penalty_zero_for_identical_weights(self):
        model = tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(4,))])
        theta_star = snapshot_trainable_weights(model)
        penalty = l2sp_penalty(model, theta_star)
        self.assertEqual(float(penalty.numpy()), 0.0)

    def test_l2sp_penalty_positive_after_weight_change(self):
        model = tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(4,))])
        theta_star = snapshot_trainable_weights(model)
        for variable in model.trainable_variables:
            variable.assign_add(tf.ones_like(variable) * 0.01)
        penalty = l2sp_penalty(model, theta_star)
        self.assertGreater(float(penalty.numpy()), 0.0)
