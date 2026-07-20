import unittest

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None

from path_til.model_factory import build_classifier


@unittest.skipIf(tf is None, "tensorflow not installed")
class ModelFactoryTests(unittest.TestCase):
    def test_build_efficientnetv2_classifier(self):
        model = build_classifier(
            "efficientnetv2_s",
            num_classes=3,
            weights=None,
            train_backbone=False,
        )
        output = model(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=False)
        self.assertEqual(tuple(output.shape), (1, 3))

    def test_convnext_requires_supported_keras(self):
        if not hasattr(tf.keras.applications, "ConvNeXtTiny"):
            self.skipTest("ConvNeXtTiny unavailable in this TensorFlow build")
        model = build_classifier(
            "convnext_tiny",
            num_classes=3,
            weights=None,
            train_backbone=False,
        )
        output = model(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=False)
        self.assertEqual(tuple(output.shape), (1, 3))
