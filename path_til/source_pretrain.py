"""Source-domain pretraining helpers for backbone replacement."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_source_pretrain_config(path: Path | str) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Source pretrain config must be a mapping: {0}".format(path))
    payload["config_path"] = str(path)
    return payload


def validate_no_testset_leakage(config: dict) -> None:
    text = yaml.safe_dump(config, sort_keys=True)
    if "dataset/Testset" in text or "Testset" in text:
        raise ValueError("Source pretrain config must not reference dataset/Testset")


def output_artifact_paths(output_dir: Path, backbone: str) -> dict[str, Path]:
    output_dir = Path(output_dir)
    prefix = "source_pretrained_{0}".format(backbone)
    return {
        "best": output_dir / "{0}_best.h5".format(prefix),
        "last": output_dir / "{0}_last.h5".format(prefix),
        "config_snapshot": output_dir / "source_pretrain_config_snapshot.yaml",
        "val_predictions": output_dir / "source_val_predictions.csv",
        "val_metrics": output_dir / "source_val_metrics.json",
        "training_log": output_dir / "source_training_log.csv",
        "learning_curve": output_dir / "source_learning_curve.png",
    }
