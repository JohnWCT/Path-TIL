#!/usr/bin/env python3
"""Compare backbone experiments against the locked HNSCC candidate reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.candidate import evaluate_against_candidate, reference_metrics  # noqa: E402
from scripts.summarize_candidate_stability import load_oof_metrics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a unified comparison table for candidate and backbone experiments."
    )
    parser.add_argument("--reference", required=True, help="Reference OOF directory")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--external-results", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_external_summary(path: Path) -> dict[str, dict]:
    summary_path = path / "external_summary.csv"
    if not summary_path.is_file():
        return {}
    frame = pd.read_csv(summary_path)
    return {
        str(row["dataset"]): row.to_dict()
        for _, row in frame.iterrows()
    }


def load_tcga_internal(path: Path) -> dict:
    metrics_path = path / "tcga_internal_metrics.json"
    if not metrics_path.is_file():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def load_source_pretrain_metrics(pretrained_path: str | None) -> dict:
    if not pretrained_path:
        return {}
    root = Path(pretrained_path).resolve().parent
    metrics_path = root / "source_val_metrics.json"
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {}


def build_row(
    experiment_name: str,
    oof_metrics: dict,
    external: dict,
    tcga_internal: dict,
    backbone: str = "",
    source_pretrained: str = "",
    source_pretrain_val: dict | None = None,
    hnscc_source_mix_ratio: float | None = None,
    is_reference: bool = False,
):
    source_pretrain_val = source_pretrain_val or {}
    row = {
        "experiment_name": experiment_name,
        "backbone": backbone,
        "source_pretrained": source_pretrained,
        "source_pretrain_val_positive_auc": source_pretrain_val.get("positive_auc"),
        "source_pretrain_val_positive_prc": source_pretrain_val.get("positive_prc"),
        "hnscc_source_mix_ratio": hnscc_source_mix_ratio,
        "hnscc_oof_positive_auc": oof_metrics.get("positive_auc"),
        "hnscc_oof_positive_prc": oof_metrics.get("positive_prc"),
        "hnscc_oof_macro_ovr_auc": oof_metrics.get("macro_ovr_auc"),
        "hnscc_oof_weighted_ovr_auc": oof_metrics.get("weighted_ovr_auc"),
        "hnscc_hard_til_mae_reference_only": oof_metrics.get("hard_til_mae"),
        "tcga_internal_positive_auc": tcga_internal.get("positive_auc"),
        "tcga_internal_positive_prc": tcga_internal.get("positive_prc"),
        "cptac_luad_positive_auc": external.get("CPTAC_LUAD", {}).get("positive_auc"),
        "cptac_luad_positive_prc": external.get("CPTAC_LUAD", {}).get("positive_prc"),
        "cptac_lusc_positive_auc": external.get("CPTAC_LUSC", {}).get("positive_auc"),
        "cptac_lusc_positive_prc": external.get("CPTAC_LUSC", {}).get("positive_prc"),
        "rumc_brca_positive_auc": external.get("RUMC-BRCA", {}).get("positive_auc"),
        "rumc_brca_positive_prc": external.get("RUMC-BRCA", {}).get("positive_prc"),
    }
    if is_reference:
        row["keep_or_drop"] = "locked"
        row["notes"] = "locked_candidate_reference"
        return row
    decision = evaluate_against_candidate(
        {
            "positive_auc": row["hnscc_oof_positive_auc"] or 0.0,
            "positive_prc": row["hnscc_oof_positive_prc"] or 0.0,
            "macro_ovr_auc": row["hnscc_oof_macro_ovr_auc"] or 0.0,
            "weighted_ovr_auc": row["hnscc_oof_weighted_ovr_auc"] or 0.0,
        }
    )
    row["keep_or_drop"] = decision["decision"]
    row["notes"] = ";".join(decision["reasons"]) if decision["reasons"] else "meets_criteria"
    return row


def match_side_result(experiment_name: str, side_by_name: dict) -> dict:
    """Match OOF experiment names to external/TCGA result directories by substring."""
    if experiment_name in side_by_name:
        return side_by_name[experiment_name]
    for key, value in side_by_name.items():
        if experiment_name and experiment_name in key:
            return value
        if key and key in experiment_name:
            return value
    return {}


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    external_by_experiment = {}
    tcga_by_experiment = {}
    for path in args.external_results:
        root = Path(path)
        external_by_experiment[root.name] = load_external_summary(root)
        tcga_by_experiment[root.name] = load_tcga_internal(root)

    rows = []
    reference = load_oof_metrics(Path(args.reference))
    ref_name = reference["experiment"]
    ref_external = match_side_result(
        "results_external_testset_r50_50", external_by_experiment
    )
    if not ref_external and args.external_results:
        ref_external = external_by_experiment.get(Path(args.external_results[0]).name, {})
    rows.append(
        build_row(
            ref_name,
            reference,
            ref_external,
            match_side_result(ref_name, tcga_by_experiment),
            backbone="irv2",
            hnscc_source_mix_ratio=0.50,
            is_reference=True,
        )
    )
    for experiment in args.experiments:
        oof = load_oof_metrics(Path(experiment))
        name = Path(experiment).name
        rows.append(
            build_row(
                name,
                oof,
                match_side_result(name, external_by_experiment),
                match_side_result(name, tcga_by_experiment),
            )
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "backbone_candidate_comparison.csv", index=False)
    with (output_dir / "reference.json").open("w", encoding="utf-8") as handle:
        json.dump(reference_metrics(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("Comparison table written -> {0}".format(output_dir / "backbone_candidate_comparison.csv"))


if __name__ == "__main__":
    main()
