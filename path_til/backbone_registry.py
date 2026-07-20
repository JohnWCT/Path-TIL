"""Supported backbone metadata for source pretraining and HNSCC fine-tuning."""

from __future__ import annotations

SUPPORTED_BACKBONES = {
    "irv2": {
        "family": "tf_keras",
        "input_size": 224,
        "default_pretrain": "tilscout",
    },
    "efficientnetv2_s": {
        "family": "tf_keras",
        "input_size": 224,
        "default_pretrain": "imagenet",
    },
    "convnext_tiny": {
        "family": "tf_keras_or_optional",
        "input_size": 224,
        "default_pretrain": "imagenet",
    },
    "swin_tiny": {
        "family": "optional_transformer",
        "input_size": 224,
        "default_pretrain": "imagenet",
    },
}


def get_backbone_spec(name: str) -> dict:
    if name not in SUPPORTED_BACKBONES:
        raise ValueError("Unsupported backbone: {0}".format(name))
    return dict(SUPPORTED_BACKBONES[name])
