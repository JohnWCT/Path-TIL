"""TCGA internal and external lock-box evaluation helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from path_til.hnscc import LABELS, classification_metrics, patch_metric_summary


LABEL_TO_IDX = {label: index for index, label in enumerate(LABELS)}

TCGA_LABEL_FOLDERS = {
    "A_positive": "positive",
    "B_negative": "negative",
    "C_other": "other",
}

EXTERNAL_LABEL_FOLDERS = {
    "A_Positive": "positive",
    "B_Negative": "negative",
    "C_Other": "other",
}

CASE_RE = re.compile(r"(TCGA-[A-Z0-9-]+)")
EXTERNAL_DATASETS = ("CPTAC_LUAD", "CPTAC_LUSC", "RUMC-BRCA")

SUMMARY_COLUMNS = [
    "dataset",
    "n_patches",
    "positive_auc",
    "positive_prc",
    "macro_ovr_auc",
    "weighted_ovr_auc",
    "accuracy",
    "macro_f1",
    "positive_precision",
    "positive_recall",
]


def discover_label_folders(root: Path) -> dict[str, str]:
    """Detect TCGA-style or external Testset label folder naming."""
    for mapping in (TCGA_LABEL_FOLDERS, EXTERNAL_LABEL_FOLDERS):
        if all((root / folder).is_dir() for folder in mapping):
            return dict(mapping)
    raise FileNotFoundError(
        "Could not find label folders under {0}; expected TCGA or Testset layout".format(
            root
        )
    )


def extract_case_id(path: Path) -> str:
    match = CASE_RE.search(path.name)
    if match is not None:
        return match.group(1)
    return path.stem


def build_patch_manifest(
    root: Path,
    path_prefix: str = "",
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """Scan a labeled patch root and return case_id,image_path,label rows."""
    root = Path(root)
    mapping = discover_label_folders(root)
    records = []
    for folder, label in mapping.items():
        sub = root / folder
        images = sorted(sub.glob("*.tif")) + sorted(sub.glob("*.tiff"))
        images += sorted(sub.glob("*.png"))
        for path in images:
            if path_prefix:
                image_path = "{0}/{1}/{2}".format(
                    path_prefix.rstrip("/"), folder, path.name
                )
            else:
                image_path = str(path.resolve())
            records.append(
                {
                    "case_id": extract_case_id(path),
                    "image_path": image_path,
                    "label": label,
                    "dataset": dataset_name or root.name,
                }
            )
    return pd.DataFrame(records, columns=["case_id", "image_path", "label", "dataset"])


def build_external_dataset_manifest(
    testset_root: Path,
    path_prefix: str = "",
) -> dict[str, pd.DataFrame]:
    """Build manifests for each external lock-box cohort."""
    testset_root = Path(testset_root)
    manifests = {}
    for name in EXTERNAL_DATASETS:
        cohort_root = testset_root / name
        if not cohort_root.is_dir():
            raise FileNotFoundError("Missing external dataset: {0}".format(cohort_root))
        prefix = path_prefix.rstrip("/")
        if prefix:
            prefix = "{0}/{1}".format(prefix, name)
        manifests[name] = build_patch_manifest(
            cohort_root,
            path_prefix=prefix,
            dataset_name=name,
        )
    return manifests


def resolve_stage_name(stage: str) -> str:
    normalized = str(stage).strip().lower()
    aliases = {
        "selected": "selected",
        "0": "stage0",
        "1": "stage1",
        "2": "stage2",
        "stage0": "stage0",
        "stage1": "stage1",
        "stage2": "stage2",
    }
    if normalized not in aliases:
        raise ValueError("Unsupported stage selector: {0}".format(stage))
    return aliases[normalized]


def resolve_fold_model_paths(model_dir: Path, stage: str = "selected") -> list[tuple[int, Path]]:
    """Resolve per-fold checkpoint paths from a GroupCV training directory."""
    model_dir = Path(model_dir)
    stage_name = resolve_stage_name(stage)
    resolved = []
    for fold_dir in sorted(model_dir.glob("fold*")):
        if not fold_dir.is_dir():
            continue
        fold_token = fold_dir.name.replace("fold", "")
        fold = int(fold_token)
        config_path = fold_dir / "config.json"
        metrics_path = fold_dir / "fold_metrics.json"
        if stage_name == "selected":
            if not metrics_path.is_file():
                raise FileNotFoundError(
                    "Missing fold metrics for selected stage: {0}".format(metrics_path)
                )
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            selected = int(payload["selected_stage"])
            model_path = _checkpoint_for_stage(fold_dir, config_path, selected)
        else:
            stage_index = {"stage0": 0, "stage1": 1, "stage2": 2}[stage_name]
            model_path = _checkpoint_for_stage(fold_dir, config_path, stage_index)
        resolved.append((fold, Path(model_path)))
    if not resolved:
        raise FileNotFoundError("No fold checkpoints found under {0}".format(model_dir))
    return resolved


def _checkpoint_for_stage(fold_dir: Path, config_path: Path, stage_index: int) -> Path:
    if stage_index == 0:
        if not config_path.is_file():
            raise FileNotFoundError("Missing fold config for stage 0: {0}".format(config_path))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        pretrained = config["parameters"]["pretrained"]
        return Path(pretrained)
    checkpoint = fold_dir / "stage{0}_best.h5".format(stage_index)
    if not checkpoint.is_file():
        raise FileNotFoundError("Missing checkpoint: {0}".format(checkpoint))
    return checkpoint


def read_training_hne_norm(model_dir: Path) -> bool:
    """Read hne_norm flag from the first fold config, defaulting to False."""
    model_dir = Path(model_dir)
    for fold_dir in sorted(model_dir.glob("fold*")):
        config_path = fold_dir / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            preprocessing = config.get("preprocessing", {})
            return bool(preprocessing.get("hne_norm", False))
    return False


def patch_evaluation_metrics(y_true, probabilities) -> dict:
    """Return flat evaluation metrics including positive PRC and confusion matrix."""
    from sklearn.metrics import average_precision_score, confusion_matrix

    y_true = np.asarray(y_true, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    base = classification_metrics(y_true, probabilities)
    y_pred = probabilities.argmax(axis=1)
    positive_binary = (y_true == 0).astype(np.int32)
    try:
        positive_prc = float(
            average_precision_score(positive_binary, probabilities[:, 0])
        )
    except ValueError:
        positive_prc = None
    from sklearn.metrics import f1_score, precision_score, recall_score

    positive_precision = float(
        precision_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)
    )
    positive_recall = float(
        recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    return {
        "positive_auc": base.get("positive_auc"),
        "positive_prc": positive_prc,
        "macro_ovr_auc": base.get("macro_ovr_auc"),
        "weighted_ovr_auc": base.get("weighted_ovr_auc"),
        "accuracy": base.get("accuracy"),
        "macro_f1": base.get("f1_macro"),
        "positive_precision": positive_precision,
        "positive_recall": positive_recall,
        "confusion_matrix": matrix.tolist(),
        "per_class_auc": base.get("per_class_auc"),
    }


def metrics_to_summary_row(dataset: str, metrics: dict, n_patches: int) -> dict:
    return {
        "dataset": dataset,
        "n_patches": int(n_patches),
        "positive_auc": metrics.get("positive_auc"),
        "positive_prc": metrics.get("positive_prc"),
        "macro_ovr_auc": metrics.get("macro_ovr_auc"),
        "weighted_ovr_auc": metrics.get("weighted_ovr_auc"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "positive_precision": metrics.get("positive_precision"),
        "positive_recall": metrics.get("positive_recall"),
    }


def write_metrics_bundle(output_dir: Path, dataset: str, metrics: dict, predictions: pd.DataFrame):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "{0}_metrics.json".format(dataset)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    predictions.to_csv(
        output_dir / "{0}_predictions.csv".format(dataset),
        index=False,
        float_format="%.8f",
    )
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    matrix_dir = output_dir / "external_confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_frame = pd.DataFrame(
        matrix,
        index=["true_{0}".format(label) for label in LABELS],
        columns=["pred_{0}".format(label) for label in LABELS],
    )
    matrix_frame.to_csv(matrix_dir / "{0}_confusion_matrix.csv".format(dataset))


def summarize_long_form_metrics(y_true, probabilities, scope: str) -> pd.DataFrame:
    return patch_metric_summary(y_true, probabilities, scope=scope)
