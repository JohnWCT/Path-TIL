#!/usr/bin/env python3
"""Select B5 backbone configs for full 5-fold promotion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.backbone_decision import (  # noqa: E402
    BackboneMetrics,
    decide_backbone_status,
)
from path_til.candidate import reference_metrics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Select B5 configs for full5.")
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional reference metrics JSON; defaults to locked candidate.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Paths to metrics.json / smoke_summary dirs / glob-expanded files",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-backbone", type=int, default=1)
    return parser.parse_args()


def load_reference(path: str | None) -> BackboneMetrics:
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metrics = payload.get("patch_metrics", payload)
        def value(key, fallback_keys=()):
            item = metrics.get(key)
            if isinstance(item, dict) and "value" in item:
                return float(item["value"])
            if item is not None and not isinstance(item, dict):
                return float(item)
            for alt in fallback_keys:
                alt_item = metrics.get(alt)
                if isinstance(alt_item, dict) and "value" in alt_item:
                    return float(alt_item["value"])
                if alt_item is not None and not isinstance(alt_item, dict):
                    return float(alt_item)
            raise KeyError(key)

        return BackboneMetrics(
            name="reference",
            positive_auc=value("positive_auc"),
            positive_prc=value("positive_prc"),
            macro_ovr_auc=value("macro_ovr_auc"),
            weighted_ovr_auc=value("weighted_ovr_auc"),
        )
    ref = reference_metrics()
    return BackboneMetrics(
        name="irv2_candidate",
        positive_auc=float(ref["positive_auc"]),
        positive_prc=float(ref["positive_prc"]),
        macro_ovr_auc=float(ref["macro_ovr_auc"]),
        weighted_ovr_auc=float(ref["weighted_ovr_auc"]),
    )


def resolve_metric_path(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("metrics.json", "smoke_summary.json", "oof_metrics.json", "eval_summary.json"):
        candidate = path / name
        if candidate.is_file():
            return candidate
        nested = path / "smoke_summary" / name
        if nested.is_file():
            return nested
    raise FileNotFoundError("No metrics JSON under {0}".format(path))


def infer_backbone(name: str) -> str:
    if "efficientnetv2_s" in name:
        return "efficientnetv2_s"
    if "convnext_tiny" in name:
        return "convnext_tiny"
    return "unknown"


def load_row(path: Path, reference: BackboneMetrics) -> dict:
    metrics_path = resolve_metric_path(path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    name = payload.get("experiment") or metrics_path.parent.name
    candidate = BackboneMetrics(
        name=name,
        positive_auc=float(payload.get("positive_auc") or 0.0),
        positive_prc=float(payload.get("positive_prc") or 0.0),
        macro_ovr_auc=float(payload.get("macro_ovr_auc") or 0.0),
        weighted_ovr_auc=float(payload.get("weighted_ovr_auc") or 0.0),
        fold_auc_gap=payload.get("fold_auc_gap"),
        fold_prc_gap=payload.get("fold_prc_gap"),
    )
    decision = decide_backbone_status(candidate, reference)
    return {
        "experiment_name": name,
        "backbone": infer_backbone(name),
        "metrics_path": str(metrics_path),
        "positive_auc": candidate.positive_auc,
        "positive_prc": candidate.positive_prc,
        "macro_ovr_auc": candidate.macro_ovr_auc,
        "weighted_ovr_auc": candidate.weighted_ovr_auc,
        "negative_auc": payload.get("negative_auc"),
        "other_auc": payload.get("other_auc"),
        "fold0_positive_auc": payload.get("fold0_positive_auc"),
        "fold1_positive_auc": payload.get("fold1_positive_auc"),
        "fold_auc_gap": candidate.fold_auc_gap,
        "fold0_positive_prc": payload.get("fold0_positive_prc"),
        "fold1_positive_prc": payload.get("fold1_positive_prc"),
        "fold_prc_gap": candidate.fold_prc_gap,
        "decision": decision.decision,
        "reasons": ";".join(decision.reasons) if decision.reasons else "",
    }


def rank_key(row: dict):
    priority = {
        "replace_candidate": 0,
        "positive_specialist_pending_full5": 1,
        "drop": 2,
    }.get(row["decision"], 3)
    return (
        priority,
        -(row["positive_auc"] or 0.0),
        -(row["positive_prc"] or 0.0),
        row["fold_auc_gap"] if row["fold_auc_gap"] is not None else 9.0,
    )


def rewrite_full5_config(selected_experiment: str) -> Path | None:
    """Point full5_selected.yaml inherit_from at the winning B5 config when present."""
    if "efficientnetv2_s" in selected_experiment:
        backbone = "efficientnetv2_s"
    elif "convnext_tiny" in selected_experiment:
        backbone = "convnext_tiny"
    else:
        return None
    b5_config = PROJECT_ROOT / "configs" / "{0}.yaml".format(selected_experiment)
    full5_config = PROJECT_ROOT / "configs" / "backbone_{0}_full5_selected.yaml".format(
        backbone
    )
    if not b5_config.is_file() or not full5_config.is_file():
        return None
    payload = {
        "experiment": {
            "name": "backbone_{0}_full5_selected".format(backbone),
            "phase": "B6",
            "description": "Selected from B5: {0}".format(selected_experiment),
        },
        "inherit_from": "configs/{0}.yaml".format(selected_experiment),
        "run": {
            "folds": [0, 1, 2, 3, 4],
            "full5": True,
            "external_lockbox": False,
        },
    }
    with full5_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return full5_config


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reference = load_reference(args.reference)

    rows = [load_row(Path(path), reference) for path in args.experiments]
    frame = pd.DataFrame(rows).sort_values(
        by=["decision", "positive_auc", "positive_prc"],
        ascending=[True, False, False],
        key=None,
    )
    # Stable custom ranking.
    frame = frame.iloc[sorted(range(len(frame)), key=lambda i: rank_key(frame.iloc[i].to_dict()))]
    frame.to_csv(output / "backbone_b5_selection.csv", index=False)

    selected = []
    per_backbone = {}
    for _, row in frame.iterrows():
        if row["decision"] == "drop":
            continue
        backbone = row["backbone"]
        count = per_backbone.get(backbone, 0)
        if count >= args.max_per_backbone:
            continue
        # Prefer replace_candidate; also allow strong positive-specialist.
        if row["decision"] in {
            "replace_candidate",
            "positive_specialist_pending_full5",
        }:
            selected.append(row["experiment_name"])
            per_backbone[backbone] = count + 1
            rewrite_full5_config(row["experiment_name"])

    (output / "selected_for_full5.txt").write_text(
        "\n".join(selected) + ("\n" if selected else ""),
        encoding="utf-8",
    )

    lines = [
        "# Backbone B5 Keep / Drop",
        "",
        "| experiment | backbone | positive AUC | positive PRC | macro OVR | weighted OVR | fold AUC gap | decision |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| {0} | {1} | {2:.4f} | {3:.4f} | {4:.4f} | {5:.4f} | {6} | {7} |".format(
                row["experiment_name"],
                row["backbone"],
                row["positive_auc"],
                row["positive_prc"],
                row["macro_ovr_auc"],
                row["weighted_ovr_auc"],
                "TBD" if pd.isna(row["fold_auc_gap"]) else "{0:.4f}".format(row["fold_auc_gap"]),
                row["decision"],
            )
        )
    lines.extend(
        [
            "",
            "## Selected for full5",
            "",
        ]
    )
    if selected:
        lines.extend(["- {0}".format(name) for name in selected])
    else:
        lines.append("- none")
    (output / "backbone_b5_keep_drop.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Selection written ->", output)
    print("selected_for_full5:", selected)


if __name__ == "__main__":
    main()
