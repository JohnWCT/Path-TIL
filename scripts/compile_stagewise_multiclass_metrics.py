#!/usr/bin/env python3
"""Compile stage-wise per-class / macro / micro AUROC & AUPRC for the master report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

LABELS = ["positive", "negative", "other"]

# stage -> list of (display_name, path_to_oof_dir)
STAGES: dict[str, list[tuple[str, str]]] = {
    "baseline": [
        ("Stage0 pretrained", "results/results_groupcv_norm_heavy/oof_stage_0"),
        ("Stage1", "results/results_groupcv_norm_heavy/oof_stage_1"),
        ("Stage2", "results/results_groupcv_norm_heavy/oof_stage_2"),
        ("validation-selected", "results/results_groupcv_norm_heavy/oof_stage_selected"),
    ],
    "recipe_a1_a3": [
        (
            "H&E on / heavy / weight on",
            "results/results_groupcv_norm_heavy/oof_stage_selected",
        ),
        (
            "H&E off / heavy / weight on",
            "results/results_groupcv_nohne_heavy/oof_stage_selected",
        ),
        (
            "H&E off / medium / weight on",
            "results/results_groupcv_nohne_medium/oof_stage_selected",
        ),
        (
            "H&E off / heavy / weight off",
            "results/results_groupcv_nohne_heavy_noweight/oof_stage_selected",
        ),
    ],
    "stage_policy_b2": [
        ("fixed Stage1", "results/results_oof_with_prc/stage_fixed_stage1"),
        ("fixed Stage2", "results/results_oof_with_prc/stage_fixed_stage2"),
        (
            "validation-selected",
            "results/results_oof_with_prc/stage_validation_multiclass_auc",
        ),
        (
            "validation positive AUC",
            "results/results_oof_with_prc/stage_validation_positive_auc",
        ),
        (
            "composite positive+macro",
            "results/results_oof_with_prc/stage_composite_positive_macro",
        ),
    ],
    "methodology_loss_aug": [
        ("no-mix candidate (H&E off/heavy/wt on)", "results/results_oof_with_prc/candidate_selected"),
        ("focal γ=1", "results/results_oof_with_prc/focal_gamma1"),
        ("focal γ=2", "results/results_oof_with_prc/focal_gamma2"),
        ("logit-adjusted CE", "results/results_oof_with_prc/logit_adjusted_ce"),
        ("balanced sampler", "results/results_oof_with_prc/balanced_sampler"),
        ("aug without geometric", "results/results_oof_with_prc/aug_without_geometric"),
        ("aug without cutout", "results/results_oof_with_prc/aug_without_cutout"),
        ("aug without blur/noise", "results/results_oof_with_prc/aug_without_blur_noise"),
        ("aug without HED", "results/results_oof_with_prc/aug_without_hed"),
    ],
    "source_mix": [
        ("HNSCC:TCGA 0.90:0.10", "results/results_oof_with_prc/source_mix_tcga_r90_10"),
        ("HNSCC:TCGA 0.75:0.25", "results/results_oof_with_prc/source_mix_tcga"),
        ("HNSCC:TCGA 0.50:0.50", "results/results_oof_with_prc/source_mix_tcga_r50_50"),
        ("HNSCC:TCGA 0.25:0.75", "results/results_oof_with_prc/source_mix_tcga_r25_75"),
    ],
    "seed_stability": [
        ("seed 42 (locked)", "results/results_oof_with_prc/source_mix_tcga_r50_50"),
        ("seed 7", "results/results_oof_with_prc/source_mix_tcga_r50_50_seed7"),
        ("seed 21", "results/results_oof_with_prc/source_mix_tcga_r50_50_seed21"),
    ],
    "l2sp": [
        ("IRV2 0.50:0.50 (no L2-SP)", "results/results_oof_with_prc/source_mix_tcga_r50_50"),
        ("L2-SP λ=1e-3", "results/results_oof_with_prc/l2sp_r50_50_lambda_1e-3"),
        ("L2-SP λ=1e-4", "results/results_oof_with_prc/l2sp_r50_50_lambda_1e-4"),
        ("L2-SP λ=1e-5", "results/results_oof_with_prc/l2sp_r50_50_lambda_1e-5"),
    ],
    "backbone_b6": [
        ("IRV2 0.50:0.50", "results/results_oof_with_prc/source_mix_tcga_r50_50"),
        (
            "EfficientNetV2-S full5",
            "results/results_oof_with_prc/backbone_efficientnetv2_s_full5_selected",
        ),
        (
            "ConvNeXt-Tiny full5",
            "results/results_oof_with_prc/backbone_convnext_tiny_full5_selected",
        ),
    ],
}


def load_patch_summary(oof_dir: Path) -> dict:
    path = oof_dir / "patch_auc_summary.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    out = {}
    for _, row in frame.iterrows():
        metric = row["metric"]
        cls = "" if pd.isna(row["class"]) else str(row["class"])
        avg = "" if pd.isna(row["average"]) else str(row["average"])
        out[(metric, cls, avg)] = float(row["value"]) if pd.notna(row["value"]) else None
    return out


def metrics_from_predictions(pred_csv: Path) -> dict:
    if not pred_csv.is_file():
        return {}
    frame = pd.read_csv(pred_csv)
    y_true = frame["y_true_idx"].to_numpy(dtype=np.int64)
    probs = frame[["prob_positive", "prob_negative", "prob_other"]].to_numpy(
        dtype=np.float64
    )
    y_bin = np.eye(len(LABELS), dtype=np.int64)[y_true]
    out = {
        "micro_auroc": float(roc_auc_score(y_bin.ravel(), probs.ravel())),
        "micro_auprc": float(average_precision_score(y_bin, probs, average="micro")),
        "macro_auprc": float(average_precision_score(y_bin, probs, average="macro")),
        "weighted_auprc": float(
            average_precision_score(y_bin, probs, average="weighted")
        ),
        "macro_auroc": float(
            roc_auc_score(
                y_true, probs, multi_class="ovr", average="macro", labels=[0, 1, 2]
            )
        ),
        "weighted_auroc": float(
            roc_auc_score(
                y_true, probs, multi_class="ovr", average="weighted", labels=[0, 1, 2]
            )
        ),
    }
    for i, label in enumerate(LABELS):
        binary = (y_true == i).astype(np.int64)
        out[f"{label}_auroc"] = float(roc_auc_score(binary, probs[:, i]))
        out[f"{label}_auprc"] = float(average_precision_score(binary, probs[:, i]))
    return out


def metrics_for_dir(oof_dir: Path) -> dict:
    summary = load_patch_summary(oof_dir)
    pred = metrics_from_predictions(oof_dir / "oof_predictions.csv")
    if not summary and not pred:
        return {"status": "missing"}

    def pick(summary_key, pred_key):
        value = summary.get(summary_key) if summary else None
        if value is None:
            value = pred.get(pred_key)
        return value

    return {
        "status": "ok",
        "positive_auroc": pick(("ovr_auc", "positive", "none"), "positive_auroc"),
        "negative_auroc": pick(("ovr_auc", "negative", "none"), "negative_auroc"),
        "other_auroc": pick(("ovr_auc", "other", "none"), "other_auroc"),
        "macro_auroc": pick(("ovr_auc", "", "macro"), "macro_auroc"),
        "micro_auroc": pred.get("micro_auroc"),
        "weighted_auroc": pick(("ovr_auc", "", "weighted"), "weighted_auroc"),
        "positive_auprc": pick(
            ("ovr_average_precision", "positive", "none"), "positive_auprc"
        ),
        "negative_auprc": pick(
            ("ovr_average_precision", "negative", "none"), "negative_auprc"
        ),
        "other_auprc": pick(("ovr_average_precision", "other", "none"), "other_auprc"),
        "macro_auprc": pick(("ovr_average_precision", "", "macro"), "macro_auprc"),
        "micro_auprc": pred.get("micro_auprc"),
        "weighted_auprc": pick(
            ("ovr_average_precision", "", "weighted"), "weighted_auprc"
        ),
    }


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def markdown_tables(rows: list[dict]) -> str:
    auroc_header = (
        "| model | pos | neg | other | macro | micro |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    auprc_header = auroc_header
    auroc_lines = []
    auprc_lines = []
    for row in rows:
        auroc_lines.append(
            "| {model} | {p} | {n} | {o} | {m} | {u} |".format(
                model=row["model"],
                p=fmt(row.get("positive_auroc")),
                n=fmt(row.get("negative_auroc")),
                o=fmt(row.get("other_auroc")),
                m=fmt(row.get("macro_auroc")),
                u=fmt(row.get("micro_auroc")),
            )
        )
        auprc_lines.append(
            "| {model} | {p} | {n} | {o} | {m} | {u} |".format(
                model=row["model"],
                p=fmt(row.get("positive_auprc")),
                n=fmt(row.get("negative_auprc")),
                o=fmt(row.get("other_auprc")),
                m=fmt(row.get("macro_auprc")),
                u=fmt(row.get("micro_auprc")),
            )
        )
    return (
        "#### AUROC\n\n"
        + auroc_header
        + "\n".join(auroc_lines)
        + "\n\n#### AUPRC\n\n"
        + auprc_header
        + "\n".join(auprc_lines)
        + "\n"
    )


def main():
    all_rows = []
    stage_md = {}
    for stage, entries in STAGES.items():
        stage_rows = []
        for model, path in entries:
            metrics = metrics_for_dir(Path(path))
            row = {"stage": stage, "model": model, "path": path, **metrics}
            all_rows.append(row)
            stage_rows.append(row)
            if metrics.get("status") != "ok":
                print("WARN missing:", stage, model, path)
        stage_md[stage] = markdown_tables(stage_rows)

    out_dir = Path("results/results_master_stagewise_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stagewise_multiclass_metrics.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print("Wrote", csv_path)

    md_path = out_dir / "stagewise_tables.md"
    parts = []
    for stage in STAGES:
        parts.append(f"## {stage}\n\n{stage_md[stage]}")
    md_path.write_text("\n".join(parts), encoding="utf-8")
    print("Wrote", md_path)


if __name__ == "__main__":
    main()
