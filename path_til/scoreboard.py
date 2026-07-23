"""Build chaptered HNSCC optimization scoreboards from registered OOF summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from path_til.experiment_registry import CANDIDATE_REFERENCE, keep_or_drop
from path_til.hnscc import LABELS, patch_metric_summary

DECISION_KEYS = (
    "positive_auc",
    "positive_prc",
    "macro_ovr_auc",
    "weighted_ovr_auc",
)

TABLE_COLUMNS = (
    "experiment",
    "theme",
    "positive_auc",
    "positive_prc",
    "macro_ovr_auc",
    "weighted_ovr_auc",
    "accuracy",
    "macro_f1",
    "hard_til_mae",
    "soft_til_mae",
    "delta_auc",
    "delta_prc",
    "vs_candidate",
    "notes",
)


def load_registry(path):
    """Load scoreboard experiment registry YAML."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Registry must be a mapping: {0}".format(path))
    if "chapters" not in payload:
        raise ValueError("Registry missing chapters: {0}".format(path))
    return payload


def _metric_value(patch, *keys):
    for key in keys:
        if key in patch and patch[key].get("value") is not None:
            return patch[key]["value"]
    return None


def resolve_summary_path(oof_dir):
    """Locate eval_summary.json under an OOF directory."""
    oof_dir = Path(oof_dir)
    if oof_dir.is_file():
        return oof_dir
    for name in (
        "eval_summary.json",
        "smoke_summary.json",
        "oof_metrics.json",
        "metrics.json",
        "threshold_til_summary.json",
        "oof_summary/eval_summary.json",
    ):
        candidate = oof_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No summary JSON under {0}".format(oof_dir))


def backfill_positive_prc(oof_dir):
    """Compute positive-vs-rest AP from oof_predictions.csv when missing."""
    oof_dir = Path(oof_dir)
    predictions_path = oof_dir / "oof_predictions.csv"
    if not predictions_path.is_file():
        return None
    frame = pd.read_csv(predictions_path)
    if "y_true_idx" not in frame.columns or "prob_positive" not in frame.columns:
        return None
    y_true = frame["y_true_idx"].to_numpy(dtype=np.int64)
    probabilities = frame[
        ["prob_positive", "prob_negative", "prob_other"]
    ].to_numpy(dtype=np.float64)
    summary = patch_metric_summary(y_true, probabilities, scope="oof_backfill")
    row = summary.loc[
        (summary["metric"] == "positive_vs_rest_average_precision")
        & (summary["class"] == "positive")
        & (summary["average"] == "binary")
    ]
    if row.empty:
        return None
    value = row.iloc[0]["value"]
    if pd.isna(value):
        return None
    return float(value)


def load_oof_metrics(oof_dir, experiment_id=None, display_name=None):
    """Load one experiment row from an OOF directory."""
    oof_dir = Path(oof_dir)
    summary_path = resolve_summary_path(oof_dir)
    with summary_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    experiment_name = experiment_id or oof_dir.name
    if oof_dir.name == "oof_summary":
        experiment_name = experiment_id or oof_dir.parent.name

    if "patch_metrics" in payload:
        patch = payload["patch_metrics"]
        slide = payload.get("slide_metrics") or {}
        metrics = {
            "experiment": display_name or experiment_name,
            "experiment_id": experiment_id or experiment_name,
            "positive_auc": _metric_value(
                patch, "positive_vs_rest_auc_positive_binary"
            ),
            "positive_prc": _metric_value(
                patch,
                "positive_vs_rest_average_precision_positive_binary",
                "ovr_average_precision_positive_none",
            ),
            "macro_ovr_auc": _metric_value(patch, "ovr_auc_macro"),
            "weighted_ovr_auc": _metric_value(patch, "ovr_auc_weighted"),
            "accuracy": _metric_value(patch, "accuracy"),
            "macro_f1": _metric_value(patch, "f1_macro"),
            "hard_til_mae": slide.get("mae"),
            "soft_til_mae": slide.get("soft_mae"),
            "pearson": slide.get("pearson_r"),
            "spearman": slide.get("spearman_r"),
            "summary_path": str(summary_path),
            "oof_path": str(oof_dir),
            "status": "ok",
        }
    else:
        metrics = {
            "experiment": display_name or experiment_name,
            "experiment_id": experiment_id or experiment_name,
            "positive_auc": payload.get("positive_auc"),
            "positive_prc": payload.get("positive_prc"),
            "macro_ovr_auc": payload.get("macro_ovr_auc"),
            "weighted_ovr_auc": payload.get("weighted_ovr_auc"),
            "accuracy": payload.get("accuracy"),
            "macro_f1": payload.get("macro_f1"),
            "hard_til_mae": payload.get("hard_til_mae"),
            "soft_til_mae": payload.get("soft_til_mae"),
            "pearson": payload.get("pearson"),
            "spearman": payload.get("spearman"),
            "summary_path": str(summary_path),
            "oof_path": str(oof_dir),
            "status": "ok",
        }

    if metrics["positive_prc"] is None:
        backfilled = backfill_positive_prc(oof_dir)
        if backfilled is not None:
            metrics["positive_prc"] = backfilled
            metrics["notes"] = "positive_prc_backfilled_from_predictions"
        else:
            metrics["notes"] = ""
    else:
        metrics["notes"] = ""

    return metrics


def apply_candidate_comparison(row, candidate=None, candidate_id=None):
    """Attach delta_* and vs_candidate using keep_or_drop."""
    candidate = candidate or CANDIDATE_REFERENCE
    candidate_id = candidate_id or candidate.get("name")

    if row.get("experiment_id") == candidate_id:
        row["delta_auc"] = 0.0
        row["delta_prc"] = 0.0
        row["vs_candidate"] = "current_candidate"
        if not row.get("notes"):
            row["notes"] = "locked candidate reference"
        return row

    usable = {
        key: row[key]
        for key in DECISION_KEYS
        if row.get(key) is not None
    }
    if len(usable) < len(DECISION_KEYS):
        row["delta_auc"] = None
        row["delta_prc"] = None
        row["vs_candidate"] = "incomplete"
        row["notes"] = (row.get("notes") or "") + ";missing_metrics"
        return row

    if row.get("hard_til_mae") is not None:
        usable["hard_til_mae"] = row["hard_til_mae"]

    decision = keep_or_drop(usable, candidate)
    row["delta_auc"] = decision["delta_positive_auc"]
    row["delta_prc"] = decision["delta_positive_prc"]
    row["vs_candidate"] = decision["decision"]
    reasons = ";".join(decision["reasons"])
    if reasons:
        extra = reasons if not row.get("notes") else row["notes"] + ";" + reasons
        row["notes"] = extra
    elif row.get("notes") == "positive_prc_backfilled_from_predictions":
        row["notes"] = "meets_criteria;positive_prc_backfilled"
    elif not row.get("notes"):
        row["notes"] = "meets_criteria"
    return row


def load_chapter_rows(chapter, results_root):
    """Load all experiment rows for one registry chapter."""
    results_root = Path(results_root)
    theme = chapter.get("title", chapter.get("id", ""))
    rows = []
    missing = []
    for entry in chapter.get("experiments", []):
        oof_rel = entry["oof_path"]
        oof_dir = results_root / oof_rel
        try:
            row = load_oof_metrics(
                oof_dir,
                experiment_id=entry.get("id"),
                display_name=entry.get("display_name"),
            )
            row["theme"] = theme
            row["chapter_id"] = chapter.get("id")
            rows.append(row)
        except FileNotFoundError:
            missing.append(
                {
                    "id": entry.get("id"),
                    "oof_path": str(oof_dir),
                    "chapter": chapter.get("id"),
                }
            )
    return rows, missing


def build_scoreboard(registry_path, results_root, candidate=None):
    """Load registry and return rows grouped by chapter plus metadata."""
    registry = load_registry(registry_path)
    results_root = Path(results_root)
    candidate = candidate or CANDIDATE_REFERENCE
    candidate_id = registry.get("candidate_id") or candidate.get("name")

    chapters_out = []
    all_rows = []
    all_missing = []

    for chapter in registry["chapters"]:
        if chapter.get("include_all"):
            continue
        rows, missing = load_chapter_rows(chapter, results_root)
        all_missing.extend(missing)
        for row in rows:
            apply_candidate_comparison(row, candidate, candidate_id)
        chapters_out.append(
            {
                "id": chapter.get("id"),
                "title": chapter.get("title"),
                "rows": rows,
            }
        )
        all_rows.extend(rows)

    # Deduplicate global leaderboard by experiment_id (first wins).
    seen = set()
    leaderboard = []
    for row in sorted(
        all_rows,
        key=lambda item: (
            -(item.get("positive_auc") or -1.0),
            item.get("experiment_id", ""),
        ),
    ):
        key = row.get("experiment_id")
        if key in seen:
            continue
        seen.add(key)
        leaderboard.append(row)

    return {
        "registry_path": str(registry_path),
        "results_root": str(results_root),
        "candidate": candidate,
        "candidate_id": candidate_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chapters": chapters_out,
        "leaderboard": leaderboard,
        "missing": all_missing,
        "all_rows": all_rows,
    }


def _fmt(value, digits=4):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return "{0:.{1}f}".format(value, digits)
    return str(value)


def render_markdown_table(rows, columns=TABLE_COLUMNS):
    """Render a Markdown table for scoreboard rows."""
    present = [column for column in columns if column in (rows[0] if rows else {})]
    if not rows:
        return "_（本章尚無可用 OOF 結果）_\n"
    lines = [
        "| " + " | ".join(present) + " |",
        "| " + " | ".join(["---"] * len(present)) + " |",
    ]
    for row in rows:
        values = [_fmt(row.get(column)) for column in present]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def chapter_best_line(rows):
    """One-line summary of best positive AUC in a chapter."""
    valid = [row for row in rows if row.get("positive_auc") is not None]
    if not valid:
        return ""
    best = max(valid, key=lambda item: item["positive_auc"])
    return (
        "本章最佳（positive AUC）：**{0}** = {1:.4f}（PRC {2}）\n".format(
            best.get("experiment"),
            best["positive_auc"],
            _fmt(best.get("positive_prc")),
        )
    )


def render_scoreboard_markdown(payload):
    """Render full living scoreboard Markdown."""
    candidate = payload["candidate"]
    lines = [
        "# HNSCC Living Scoreboard",
        "",
        "自動產生；請勿手改分數。再生指令見文末。",
        "",
        "- 產生時間（UTC）：`{0}`".format(payload["generated_at"]),
        "- 候選：`{0}`".format(payload["candidate_id"]),
        "- Positive AUC：**{0:.4f}**".format(candidate["positive_auc"]),
        "- Positive PRC：**{0:.4f}**".format(candidate["positive_prc"]),
        "- 主要判斷：positive AUC / PRC；hard/soft TIL MAE 僅參考。",
        "",
        "敘事報告：[`hnscc_master_report.md`](hnscc_master_report.md)",
        "",
    ]

    lines.extend(
        [
            "## 1. 目前候選與全域排行（依 positive AUC 降序）",
            "",
            render_markdown_table(payload["leaderboard"]),
            "",
        ]
    )

    section_no = 2
    for chapter in payload["chapters"]:
        lines.append("## {0}. {1}".format(section_no, chapter["title"]))
        lines.append("")
        lines.append(render_markdown_table(chapter["rows"]))
        best = chapter_best_line(chapter["rows"])
        if best:
            lines.append(best)
        lines.append("")
        section_no += 1

    # Appendix
    lines.extend(
        [
            "## 附錄",
            "",
            "### 缺漏實驗（路徑不存在）",
            "",
        ]
    )
    if payload["missing"]:
        for item in payload["missing"]:
            lines.append(
                "- `{0}` → `{1}`（chapter `{2}`）".format(
                    item.get("id"), item.get("oof_path"), item.get("chapter")
                )
            )
    else:
        lines.append("- （無）")
    lines.extend(
        [
            "",
            "### 再生指令（Docker TIL）",
            "",
            "```bash",
            "docker exec -w /workspace TIL python3 scripts/build_hnscc_scoreboard.py \\",
            "  --registry configs/scoreboard_experiments.yaml \\",
            "  --results-root results \\",
            "  --output-md docs/hnscc_living_scoreboard.md \\",
            "  --output-csv results/results_methodology_comparison_scoreboard/scoreboard.csv",
            "```",
            "",
            "CSV：`results/results_methodology_comparison_scoreboard/scoreboard.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def scoreboard_to_dataframe(payload):
    """Flatten scoreboard rows for CSV export."""
    rows = []
    for chapter in payload["chapters"]:
        for row in chapter["rows"]:
            export = {key: row.get(key) for key in TABLE_COLUMNS}
            export["chapter_id"] = chapter.get("id")
            export["experiment_id"] = row.get("experiment_id")
            export["oof_path"] = row.get("oof_path")
            export["summary_path"] = row.get("summary_path")
            rows.append(export)
    return pd.DataFrame(rows)


def write_scoreboard(
    registry_path,
    results_root,
    output_md,
    output_csv=None,
    candidate=None,
):
    """Build and write Markdown + optional CSV scoreboard."""
    payload = build_scoreboard(registry_path, results_root, candidate=candidate)
    output_md = Path(output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_scoreboard_markdown(payload), encoding="utf-8")

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame = scoreboard_to_dataframe(payload)
        frame.to_csv(output_csv, index=False, float_format="%.8f")

    return payload
