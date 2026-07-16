#!/usr/bin/env python3
"""Evaluate a fixed checkpoint on case-level GroupCV held-out tests (no training)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_base():
    path = PROJECT_ROOT / "scripts" / "train_hnscc_groupcv_irv2.py"
    spec = importlib.util.spec_from_file_location("train_hnscc_groupcv_irv2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_base()

from path_til.hnscc import (  # noqa: E402
    LABELS,
    load_hnscc_csv,
    validate_fold_assignments,
)


LABEL_TO_IDX = {label: index for index, label in enumerate(LABELS)}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-weight inference on each GroupCV held-out test split "
            "and write fold*/test_predictions.csv for OOF evaluation."
        )
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--fold-csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hne-norm", choices=("on", "off"), default="on")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--image-workers",
        type=int,
        default=min(8, max(1, __import__("os").cpu_count() or 1)),
    )
    parser.add_argument("--tag", default="old_base")
    return parser.parse_args()


def main():
    args = parse_args()
    frame = load_hnscc_csv(args.csv, expected_cases=10)
    assignments = base.load_assignments(args.fold_csv)
    validate_fold_assignments(frame, assignments, n_folds=5)
    base.check_image_paths(frame)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf

    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as error:  # noqa: BLE001
        print("GPU setup warning: {0}".format(error), file=sys.stderr)

    print("Loading images with hne_norm={0}...".format(args.hne_norm))
    images, preprocessing_report = base.load_all_images(
        frame["image_path"].tolist(),
        base.on_off(args.hne_norm),
        args.image_workers,
    )
    print("H&E report: {0}".format(preprocessing_report))

    model = base.load_and_validate_model(tf, args.model, mixed_precision=False)
    labels = frame["label"].map(LABEL_TO_IDX).to_numpy(dtype=np.int32)
    sequence = base.nonaug_sequence(tf, images, labels, args.batch_size)
    print("Predicting {0} patches...".format(len(frame)))
    probabilities = np.asarray(model.predict(sequence, verbose=1), dtype=np.float64)

    all_predictions, _ = base.prediction_frame(
        frame, np.arange(len(frame)), fold=-1, split="all", probabilities=probabilities
    )

    for fold in range(5):
        indices = base.split_indices(frame, assignments, fold)
        test_idx = indices["test"]
        fold_dir = output_dir / "fold{0:02d}".format(fold)
        fold_dir.mkdir(parents=True, exist_ok=True)
        subset = all_predictions.iloc[test_idx].copy()
        subset["fold"] = int(fold)
        subset["split"] = "test"
        subset.to_csv(
            fold_dir / "test_predictions.csv", index=False, float_format="%.8f"
        )
        # Also keep stage0 naming for compatibility with --stage 0 eval.
        subset.to_csv(
            fold_dir / "stage0_test_predictions.csv",
            index=False,
            float_format="%.8f",
        )
        payload = {
            "tag": args.tag,
            "fold": fold,
            "model": str(args.model),
            "hne_norm": args.hne_norm,
            "selected_stage": 0,
            "n_test": int(len(subset)),
            "test_cases": sorted(
                frame.iloc[test_idx]["case_id"].astype(str).unique().tolist()
            ),
            "preprocessing_report": preprocessing_report,
        }
        base.write_json(fold_dir / "fold_metrics.json", payload)
        print(
            "Fold {0}: wrote {1} test predictions for cases {2}".format(
                fold, len(subset), payload["test_cases"]
            )
        )

    summary = {
        "tag": args.tag,
        "model": str(args.model),
        "hne_norm": args.hne_norm,
        "n_patches": int(len(frame)),
        "preprocessing_report": preprocessing_report,
        "note": (
            "Fixed checkpoint inference on GroupCV held-out tests only; "
            "no HNSCC fine-tuning."
        ),
    }
    base.write_json(output_dir / "old_base_inference_config.json", summary)
    print("Old-base inference complete -> {0}".format(output_dir))


if __name__ == "__main__":
    main()
