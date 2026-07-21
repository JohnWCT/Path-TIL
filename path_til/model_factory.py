"""Build Keras classifiers for backbone replacement experiments."""

from __future__ import annotations


def build_classifier(
    backbone: str,
    input_shape=(224, 224, 3),
    num_classes: int = 3,
    weights="imagenet",
    dropout: float = 0.3,
    train_backbone: bool = False,
):
    """Return a softmax classifier built on a named Keras application backbone."""
    import tensorflow as tf

    if backbone == "efficientnetv2_s":
        base = tf.keras.applications.EfficientNetV2S(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling="avg",
        )
    elif backbone == "convnext_tiny":
        if not hasattr(tf.keras.applications, "ConvNeXtTiny"):
            raise ImportError(
                "ConvNeXtTiny is not available in this TensorFlow/Keras build. "
                "Use EfficientNetV2-S first, or install a compatible Keras version."
            )
        base = tf.keras.applications.ConvNeXtTiny(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling="avg",
        )
    else:
        raise ValueError("Unsupported backbone for this factory: {0}".format(backbone))

    base.trainable = train_backbone
    inputs = base.input
    features = base.output
    features = tf.keras.layers.Dropout(dropout)(features)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(features)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="{0}_classifier".format(backbone))


def load_classifier_from_checkpoint(
    tf,
    backbone: str,
    path: str,
    num_classes: int = 3,
    dropout: float = 0.3,
):
    """Load a saved classifier; rebuild + load_weights when custom layers block load_model."""
    path = str(path)
    try:
        return tf.keras.models.load_model(path, compile=False)
    except (ValueError, TypeError, OSError) as error:
        message = str(error)
        if backbone != "convnext_tiny" and "LayerScale" not in message and "Unknown layer" not in message:
            raise
        model = build_classifier(
            backbone,
            num_classes=num_classes,
            weights=None,
            dropout=dropout,
            train_backbone=True,
        )
        model.load_weights(path)
        return model
