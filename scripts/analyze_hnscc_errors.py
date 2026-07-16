#!/usr/bin/env python3
"""Create case-level classification, confidence, and stain error audits."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


LABELS = ("positive", "negative", "other")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze HNSCC OOF errors by case, class, confidence, and stain"
    )
    parser.add_argument("--oof-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-errors", type=int, default=100)
    parser.add_argument(
        "--image-workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
    )
    return parser.parse_args()


def validate_predictions(frame):
    required = {
        "case_id",
        "image_path",
        "y_true_label",
        "y_pred_label",
        "confidence",
        "prob_positive",
        "prob_negative",
        "prob_other",
        "correct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("OOF CSV is missing columns: {0}".format(missing))
    if frame.empty:
        raise ValueError("OOF CSV contains no rows")
    if frame["image_path"].duplicated().any():
        raise ValueError("OOF CSV contains duplicate image paths")
    for column in ("y_true_label", "y_pred_label"):
        invalid = sorted(set(frame[column]) - set(LABELS))
        if invalid:
            raise ValueError("{0} has invalid labels: {1}".format(column, invalid))


def image_statistics(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "image_path": str(path),
            "image_status": "unreadable",
            "rgb_mean": np.nan,
            "rgb_std": np.nan,
            "saturation_mean": np.nan,
            "optical_density_mean": np.nan,
        }
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    optical_density = -np.log((rgb.astype(np.float64) + 1.0) / 256.0)
    return {
        "image_path": str(path),
        "image_status": "ok",
        "rgb_mean": float(rgb.mean()),
        "rgb_std": float(rgb.std()),
        "saturation_mean": float(hsv[:, :, 1].mean()),
        "optical_density_mean": float(optical_density.mean()),
    }


def confusion_by_case(frame):
    rows = []
    for case_id, group in frame.groupby("case_id", sort=True):
        table = pd.crosstab(group["y_true_label"], group["y_pred_label"])
        table = table.reindex(index=LABELS, columns=LABELS, fill_value=0)
        for true_label in LABELS:
            for pred_label in LABELS:
                rows.append(
                    {
                        "case_id": case_id,
                        "y_true_label": true_label,
                        "y_pred_label": pred_label,
                        "count": int(table.loc[true_label, pred_label]),
                    }
                )
    return pd.DataFrame(rows)


def per_case_class_metrics(frame):
    rows = []
    for case_id, group in frame.groupby("case_id", sort=True):
        for label in LABELS:
            true_positive = int(
                (
                    (group["y_true_label"] == label)
                    & (group["y_pred_label"] == label)
                ).sum()
            )
            false_positive = int(
                (
                    (group["y_true_label"] != label)
                    & (group["y_pred_label"] == label)
                ).sum()
            )
            false_negative = int(
                (
                    (group["y_true_label"] == label)
                    & (group["y_pred_label"] != label)
                ).sum()
            )
            support = int((group["y_true_label"] == label).sum())
            precision = (
                true_positive / float(true_positive + false_positive)
                if true_positive + false_positive
                else np.nan
            )
            recall = (
                true_positive / float(true_positive + false_negative)
                if true_positive + false_negative
                else np.nan
            )
            rows.append(
                {
                    "case_id": case_id,
                    "class": label,
                    "support": support,
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "precision": precision,
                    "recall": recall,
                }
            )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    frame = pd.read_csv(args.oof_csv)
    validate_predictions(frame)
    correct_text = frame["correct"].astype(str).str.lower()
    if not correct_text.isin(("true", "false")).all():
        raise ValueError("correct column must contain true/false values")
    frame["correct"] = correct_text.eq("true")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.image_workers) as pool:
        stats = list(pool.map(image_statistics, frame["image_path"].tolist()))
    patch_audit = frame.merge(pd.DataFrame(stats), on="image_path", how="left")
    patch_audit.to_csv(
        output / "patch_error_audit.csv", index=False, float_format="%.8f"
    )

    confusion_by_case(frame).to_csv(
        output / "confusion_by_case.csv", index=False
    )
    class_metrics = per_case_class_metrics(frame)
    class_metrics.to_csv(
        output / "case_class_metrics.csv", index=False, float_format="%.8f"
    )

    confidence = (
        patch_audit.groupby(["case_id", "correct"], dropna=False)
        .agg(
            n=("image_path", "size"),
            confidence_mean=("confidence", "mean"),
            confidence_median=("confidence", "median"),
            rgb_mean=("rgb_mean", "mean"),
            saturation_mean=("saturation_mean", "mean"),
            optical_density_mean=("optical_density_mean", "mean"),
        )
        .reset_index()
    )
    confidence.to_csv(
        output / "confidence_stain_by_case.csv",
        index=False,
        float_format="%.8f",
    )

    errors = patch_audit[~patch_audit["correct"].astype(bool)].copy()
    errors = errors.sort_values(
        ["confidence", "case_id", "image_path"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    errors.head(args.top_errors).to_csv(
        output / "high_confidence_errors.csv",
        index=False,
        float_format="%.8f",
    )

    positive = class_metrics[class_metrics["class"] == "positive"].copy()
    worst_positive = positive.sort_values(
        ["false_positive", "false_negative"],
        ascending=[False, False],
        kind="mergesort",
    )
    summary = {
        "n_patches": int(len(frame)),
        "n_cases": int(frame["case_id"].nunique()),
        "accuracy": float(frame["correct"].astype(bool).mean()),
        "n_errors": int(len(errors)),
        "unreadable_images": int(
            (patch_audit["image_status"] != "ok").sum()
        ),
        "highest_positive_false_positive_cases": (
            worst_positive[
                ["case_id", "false_positive", "false_negative"]
            ]
            .head(5)
            .to_dict("records")
        ),
    }
    with (output / "error_audit_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        "Error audit complete: {0} patches, {1} errors -> {2}".format(
            len(frame), len(errors), output
        )
    )


if __name__ == "__main__":
    main()
