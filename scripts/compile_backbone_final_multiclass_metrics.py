#!/usr/bin/env python3
"""Compile final per-class / macro / micro / weighted AUROC & AUPRC tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

LABELS = ["positive", "negative", "other"]
ROOT = Path("results/results_oof_with_prc")
EXPS = {
    "IRV2 candidate (0.50:0.50)": "source_mix_tcga_r50_50",
    "EfficientNetV2-S full5": "backbone_efficientnetv2_s_full5_selected",
    "ConvNeXt-Tiny full5": "backbone_convnext_tiny_full5_selected",
}
EXT_ROOTS = {
    "IRV2 candidate (0.50:0.50)": Path("results/results_external_testset_r50_50"),
    "EfficientNetV2-S full5": Path(
        "results/results_external_testset_backbone_efficientnetv2_s_full5_selected"
    ),
    "ConvNeXt-Tiny full5": Path(
        "results/results_external_testset_backbone_convnext_tiny_full5_selected"
    ),
}


def load_patch_summary(oof_dir: Path) -> dict:
    frame = pd.read_csv(oof_dir / "patch_auc_summary.csv")
    out = {}
    for _, row in frame.iterrows():
        metric = row["metric"]
        cls = "" if pd.isna(row["class"]) else str(row["class"])
        avg = "" if pd.isna(row["average"]) else str(row["average"])
        out[(metric, cls, avg)] = float(row["value"]) if pd.notna(row["value"]) else None
    return out


def micro_from_predictions(pred_csv: Path) -> dict:
    frame = pd.read_csv(pred_csv)
    y_true = frame["y_true_idx"].to_numpy(dtype=np.int64)
    probs = frame[["prob_positive", "prob_negative", "prob_other"]].to_numpy(
        dtype=np.float64
    )
    y_bin = np.eye(len(LABELS), dtype=np.int64)[y_true]
    return {
        "micro_ovr_auc": float(roc_auc_score(y_bin.ravel(), probs.ravel())),
        "micro_ovr_ap": float(average_precision_score(y_bin, probs, average="micro")),
    }


def metrics_from_predictions(pred_csv: Path) -> dict:
    frame = pd.read_csv(pred_csv)
    y_true = frame["y_true_idx"].to_numpy(dtype=np.int64)
    probs = frame[["prob_positive", "prob_negative", "prob_other"]].to_numpy(
        dtype=np.float64
    )
    y_bin = np.eye(len(LABELS), dtype=np.int64)[y_true]
    row = {}
    for i, label in enumerate(LABELS):
        binary = (y_true == i).astype(np.int64)
        row[f"{label}_auc"] = float(roc_auc_score(binary, probs[:, i]))
        row[f"{label}_ap"] = float(average_precision_score(binary, probs[:, i]))
    row["macro_auc"] = float(
        roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=[0, 1, 2])
    )
    row["macro_ap"] = float(average_precision_score(y_bin, probs, average="macro"))
    row["micro_auc"] = float(roc_auc_score(y_bin.ravel(), probs.ravel()))
    row["micro_ap"] = float(average_precision_score(y_bin, probs, average="micro"))
    row["weighted_auc"] = float(
        roc_auc_score(
            y_true, probs, multi_class="ovr", average="weighted", labels=[0, 1, 2]
        )
    )
    row["weighted_ap"] = float(average_precision_score(y_bin, probs, average="weighted"))
    return row


def main():
    rows = []
    for name, dirname in EXPS.items():
        oof_dir = ROOT / dirname
        summary = load_patch_summary(oof_dir)
        micro = micro_from_predictions(oof_dir / "oof_predictions.csv")
        rows.append(
            {
                "model": name,
                "experiment": dirname,
                "positive_auroc": summary[("ovr_auc", "positive", "none")],
                "negative_auroc": summary[("ovr_auc", "negative", "none")],
                "other_auroc": summary[("ovr_auc", "other", "none")],
                "macro_auroc": summary[("ovr_auc", "", "macro")],
                "micro_auroc": micro["micro_ovr_auc"],
                "weighted_auroc": summary[("ovr_auc", "", "weighted")],
                "positive_auprc": summary[("ovr_average_precision", "positive", "none")],
                "negative_auprc": summary[("ovr_average_precision", "negative", "none")],
                "other_auprc": summary[("ovr_average_precision", "other", "none")],
                "macro_auprc": summary[("ovr_average_precision", "", "macro")],
                "micro_auprc": micro["micro_ovr_ap"],
                "weighted_auprc": summary[("ovr_average_precision", "", "weighted")],
            }
        )

    out_dir = Path("results/results_backbone_full5_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_csv = out_dir / "backbone_final_multiclass_metrics.csv"
    pd.DataFrame(rows).to_csv(oof_csv, index=False)
    print("Wrote", oof_csv)

    ext_rows = []
    for name, root in EXT_ROOTS.items():
        for cohort in ["CPTAC_LUAD", "CPTAC_LUSC", "RUMC-BRCA"]:
            m = metrics_from_predictions(root / f"{cohort}_predictions.csv")
            ext_rows.append({"model": name, "dataset": cohort, **m})
    ext_csv = out_dir / "backbone_final_external_multiclass_metrics.csv"
    pd.DataFrame(ext_rows).to_csv(ext_csv, index=False)
    print("Wrote", ext_csv)


if __name__ == "__main__":
    main()
