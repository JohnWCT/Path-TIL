"""HNSCC dataset validation, grouped folds, and distribution metrics.

This module deliberately has no TensorFlow dependency so fold generation and
dataset checks can run in lightweight environments.
"""

import hashlib
from pathlib import PurePath

import numpy as np
import pandas as pd


LABELS = ("positive", "negative", "other")
CSV_COLUMNS = ("case_id", "image_path", "label")
ASSIGNMENT_COLUMNS = ("fold", "case_id", "role")
PREDICTION_COLUMNS = (
    "patch_id",
    "case_id",
    "image_path",
    "fold",
    "split",
    "y_true_idx",
    "y_true_label",
    "y_pred_idx",
    "y_pred_label",
    "prob_positive",
    "prob_negative",
    "prob_other",
    "confidence",
    "correct",
)


def _path_parts(path_value):
    # PurePath on POSIX does not split Windows separators.
    return PurePath(str(path_value).replace("\\", "/")).parts


def load_hnscc_csv(csv_path, expected_cases=10):
    """Read and validate a case-level HNSCC patch manifest."""
    frame = pd.read_csv(csv_path)
    missing_columns = [column for column in CSV_COLUMNS if column not in frame.columns]
    extra_columns = [column for column in frame.columns if column not in CSV_COLUMNS]
    if missing_columns or extra_columns:
        raise ValueError(
            "CSV schema must be exactly case_id,image_path,label; "
            "missing={0}, extra={1}".format(missing_columns, extra_columns)
        )
    if frame.empty:
        raise ValueError("CSV contains no rows")
    if frame.loc[:, list(CSV_COLUMNS)].isnull().any().any():
        null_counts = frame.loc[:, list(CSV_COLUMNS)].isnull().sum()
        raise ValueError(
            "CSV contains null values: {0}".format(
                {key: int(value) for key, value in null_counts.items() if value}
            )
        )

    frame = frame.loc[:, list(CSV_COLUMNS)].copy()
    for column in CSV_COLUMNS:
        frame[column] = frame[column].astype(str)
    empty_counts = {
        column: int(frame[column].str.strip().eq("").sum()) for column in CSV_COLUMNS
    }
    empty_counts = {key: value for key, value in empty_counts.items() if value}
    if empty_counts:
        raise ValueError("CSV contains empty values: {0}".format(empty_counts))

    invalid_labels = sorted(set(frame["label"]) - set(LABELS))
    if invalid_labels:
        raise ValueError(
            "label must be one of {0}; found {1}".format(LABELS, invalid_labels)
        )

    duplicated = frame.loc[frame["image_path"].duplicated(keep=False), "image_path"]
    if not duplicated.empty:
        examples = sorted(duplicated.unique())[:5]
        raise ValueError("Duplicate image_path values found: {0}".format(examples))

    case_ids = sorted(frame["case_id"].unique())
    case_id_set = set(case_ids)
    mismatched = frame[
        [
            set(_path_parts(image_path)).intersection(case_id_set) != {str(case_id)}
            for case_id, image_path in zip(frame["case_id"], frame["image_path"])
        ]
    ]
    if not mismatched.empty:
        examples = mismatched.loc[:, ["case_id", "image_path"]].head(5)
        raise ValueError(
            "case_id must occur as a complete image_path component; examples: {0}".format(
                examples.to_dict("records")
            )
        )

    if len(case_ids) != expected_cases:
        raise ValueError(
            "Expected exactly {0} cases, found {1}: {2}".format(
                expected_cases, len(case_ids), case_ids
            )
        )

    counts = case_label_counts(frame)
    missing_per_case = []
    for case_id in case_ids:
        absent = [label for label in LABELS if int(counts.loc[case_id, label]) < 1]
        if absent:
            missing_per_case.append({"case_id": case_id, "labels": absent})
    if missing_per_case:
        raise ValueError(
            "Every case must contain at least one patch of every label: {0}".format(
                missing_per_case
            )
        )
    return frame


def case_label_counts(frame):
    """Return case-by-label patch counts in the canonical label order."""
    counts = pd.crosstab(frame["case_id"], frame["label"])
    return counts.reindex(columns=LABELS, fill_value=0).sort_index()


def normalized_distribution_error(observed, target):
    """Squared relative error for class counts plus total patch count."""
    observed_array = np.asarray(observed, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if np.any(target_array <= 0):
        raise ValueError("Distribution targets must be positive")
    class_error = np.square((observed_array - target_array) / target_array).sum()
    observed_total = float(observed_array.sum())
    target_total = float(target_array.sum())
    size_error = ((observed_total - target_total) / target_total) ** 2
    return float(class_error + size_error)


def _perfect_matchings(items):
    """Yield every perfect matching exactly once."""
    if not items:
        yield ()
        return
    first = items[0]
    for index in range(1, len(items)):
        second = items[index]
        remaining = items[1:index] + items[index + 1 :]
        for rest in _perfect_matchings(remaining):
            yield ((first, second),) + rest


def _stable_key(seed, value):
    text = "{0}|{1}".format(seed, repr(value)).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def _select_test_pairs(counts, n_folds, seed):
    cases = tuple(counts.index.tolist())
    target = counts.sum(axis=0).to_numpy(dtype=np.float64) / float(n_folds)
    vectors = {
        case_id: counts.loc[case_id].to_numpy(dtype=np.float64) for case_id in cases
    }
    best_matching = None
    best_objective = None
    best_rank = None
    for matching in _perfect_matchings(cases):
        objective = sum(
            normalized_distribution_error(vectors[left] + vectors[right], target)
            for left, right in matching
        )
        canonical = tuple(sorted(tuple(sorted(pair)) for pair in matching))
        rank = _stable_key(seed, canonical)
        if (
            best_objective is None
            or objective < best_objective - 1e-12
            or (abs(objective - best_objective) <= 1e-12 and rank < best_rank)
        ):
            best_matching = canonical
            best_objective = objective
            best_rank = rank
    ordered_pairs = sorted(
        best_matching, key=lambda pair: (_stable_key(seed, pair), pair)
    )
    return ordered_pairs, float(best_objective)


def _select_validation_cases(counts, test_pairs, seed):
    cases = tuple(counts.index.tolist())
    target = counts.mean(axis=0).to_numpy(dtype=np.float64)
    vectors = {
        case_id: counts.loc[case_id].to_numpy(dtype=np.float64) for case_id in cases
    }
    best_selection = None
    best_objective = None
    best_rank = None

    def visit(fold, selected, used, objective):
        nonlocal best_selection, best_objective, best_rank
        if best_objective is not None and objective > best_objective + 1e-12:
            return
        if fold == len(test_pairs):
            selection = tuple(selected)
            rank = _stable_key(seed, selection)
            if (
                best_objective is None
                or objective < best_objective - 1e-12
                or (abs(objective - best_objective) <= 1e-12 and rank < best_rank)
            ):
                best_selection = selection
                best_objective = objective
                best_rank = rank
            return
        blocked = set(test_pairs[fold]) | used
        candidates = [case_id for case_id in cases if case_id not in blocked]
        candidates.sort(key=lambda case_id: (_stable_key(seed + fold, case_id), case_id))
        for case_id in candidates:
            case_error = normalized_distribution_error(vectors[case_id], target)
            visit(
                fold + 1,
                selected + [case_id],
                used | {case_id},
                objective + case_error,
            )

    visit(0, [], set(), 0.0)
    if best_selection is None:
        raise ValueError("No feasible set of distinct validation cases exists")
    return list(best_selection), float(best_objective)


def build_fold_assignments(frame, n_folds=5, seed=42):
    """Build deterministic grouped test pairs and distinct validation cases."""
    counts = case_label_counts(frame)
    case_ids = counts.index.tolist()
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(case_ids) != 2 * n_folds:
        raise ValueError(
            "Pair-based folds require exactly 2 * n_folds cases; found {0} for {1} folds".format(
                len(case_ids), n_folds
            )
        )

    test_pairs, test_objective = _select_test_pairs(counts, n_folds, seed)
    validation_cases, validation_objective = _select_validation_cases(
        counts, test_pairs, seed
    )
    rows = []
    for fold, pair in enumerate(test_pairs):
        test_cases = set(pair)
        validation_case = validation_cases[fold]
        for case_id in case_ids:
            if case_id in test_cases:
                role = "test"
            elif case_id == validation_case:
                role = "val"
            else:
                role = "train"
            rows.append({"fold": fold, "case_id": case_id, "role": role})
    assignments = pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)
    objective = {
        "test": test_objective,
        "val": validation_objective,
        "total": test_objective + validation_objective,
    }
    validate_fold_assignments(frame, assignments, n_folds=n_folds)
    return assignments, objective


def validate_fold_assignments(frame, assignments, n_folds=5):
    """Validate grouped fold coverage, exclusivity, and label availability."""
    if list(assignments.columns) != list(ASSIGNMENT_COLUMNS):
        raise ValueError(
            "Assignment columns must be exactly {0}".format(ASSIGNMENT_COLUMNS)
        )
    if assignments.duplicated(["fold", "case_id"]).any():
        raise ValueError("A case has multiple roles within the same fold")

    cases = sorted(frame["case_id"].unique())
    expected_folds = list(range(n_folds))
    actual_folds = sorted(assignments["fold"].unique().tolist())
    if actual_folds != expected_folds:
        raise ValueError(
            "fold values must be 0..{0}; found {1}".format(n_folds - 1, actual_folds)
        )
    valid_roles = {"train", "val", "test"}
    invalid_roles = sorted(set(assignments["role"]) - valid_roles)
    if invalid_roles:
        raise ValueError("Invalid assignment roles: {0}".format(invalid_roles))

    for fold in expected_folds:
        fold_rows = assignments[assignments["fold"] == fold]
        if sorted(fold_rows["case_id"].tolist()) != cases:
            raise ValueError("Fold {0} does not contain every case exactly once".format(fold))
        role_counts = fold_rows["role"].value_counts().to_dict()
        expected_counts = {"test": 2, "val": 1, "train": len(cases) - 3}
        if role_counts != expected_counts:
            raise ValueError(
                "Fold {0} role counts are {1}, expected {2}".format(
                    fold, role_counts, expected_counts
                )
            )

    test_counts = (
        assignments[assignments["role"] == "test"]["case_id"].value_counts().to_dict()
    )
    if any(test_counts.get(case_id, 0) != 1 for case_id in cases):
        raise ValueError("Every case must be assigned to test exactly once")
    validation_cases = assignments[assignments["role"] == "val"]["case_id"]
    if validation_cases.duplicated().any():
        raise ValueError("Validation cases must be distinct across folds")

    labels_by_case = {
        case_id: set(group["label"]) for case_id, group in frame.groupby("case_id")
    }
    for fold in expected_folds:
        fold_rows = assignments[assignments["fold"] == fold]
        for role in ("train", "val", "test"):
            role_cases = fold_rows.loc[fold_rows["role"] == role, "case_id"]
            present = set()
            for case_id in role_cases:
                present.update(labels_by_case[case_id])
            if present != set(LABELS):
                raise ValueError(
                    "Fold {0} role {1} does not contain all three labels".format(
                        fold, role
                    )
                )
    return True


def _distribution_for_cases(counts, case_ids):
    values = counts.loc[list(case_ids)].sum(axis=0)
    distribution = {label: int(values[label]) for label in LABELS}
    distribution["total"] = int(values.sum())
    distribution["cases"] = sorted(case_ids)
    return distribution


def build_summary(frame, assignments, seed, objective):
    """Build a JSON-serializable fold summary."""
    counts = case_label_counts(frame)
    folds = []
    for fold in sorted(assignments["fold"].unique()):
        fold_rows = assignments[assignments["fold"] == fold]
        distributions = {}
        for role in ("train", "val", "test"):
            role_cases = fold_rows.loc[fold_rows["role"] == role, "case_id"].tolist()
            distributions[role] = _distribution_for_cases(counts, role_cases)
        folds.append({"fold": int(fold), "distributions": distributions})
    return {
        "seed": int(seed),
        "objective": {key: float(value) for key, value in objective.items()},
        "folds": folds,
    }


def fold_split_details(frame, assignments, fold):
    """Return case lists and patch counts for one validated grouped fold."""
    fold_rows = assignments[assignments["fold"] == int(fold)]
    if fold_rows.empty:
        raise ValueError("Fold {0} is not present in assignments".format(fold))
    details = {}
    for role in ("train", "val", "test"):
        cases = sorted(
            fold_rows.loc[fold_rows["role"] == role, "case_id"].astype(str).tolist()
        )
        split = frame[frame["case_id"].isin(cases)]
        counts = split["label"].value_counts().reindex(LABELS, fill_value=0)
        details[role] = {
            "cases": cases,
            "class_counts": {
                label: int(counts[label]) for label in LABELS
            },
            "total": int(len(split)),
        }
    return details


def balanced_class_weights(labels):
    """Compute sklearn-style balanced weights for all canonical classes."""
    values = pd.Series(labels)
    counts = values.value_counts()
    missing = [label for label in LABELS if int(counts.get(label, 0)) == 0]
    if missing:
        raise ValueError(
            "Balanced class weights require every class; missing {0}".format(missing)
        )
    total = float(len(values))
    n_classes = float(len(LABELS))
    return {
        index: total / (n_classes * float(counts[label]))
        for index, label in enumerate(LABELS)
    }


def classification_metrics(y_true, probabilities):
    """Compute multiclass metrics with safe one-vs-rest AUC handling."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        cohen_kappa_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(LABELS):
        raise ValueError("probabilities must have shape (n, {0})".format(len(LABELS)))
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities lengths differ")
    y_pred = probabilities.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    per_class_auc = {}
    for index, label in enumerate(LABELS):
        binary = (y_true == index).astype(np.int32)
        try:
            value = float(roc_auc_score(binary, probabilities[:, index]))
        except ValueError:
            value = None
        per_class_auc[label] = value
        result["{0}_auc".format(label)] = value
    result["positive_auc"] = per_class_auc["positive"]
    positive_idx = LABELS.index("positive")
    positive_binary = (y_true == positive_idx).astype(np.int32)
    try:
        result["positive_prc"] = float(
            average_precision_score(positive_binary, probabilities[:, positive_idx])
        )
    except ValueError:
        result["positive_prc"] = None
    try:
        result["macro_ovr_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=list(range(len(LABELS))),
                average="macro",
                multi_class="ovr",
            )
        )
    except ValueError:
        result["macro_ovr_auc"] = None
    try:
        result["weighted_ovr_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=list(range(len(LABELS))),
                average="weighted",
                multi_class="ovr",
            )
        )
    except ValueError:
        result["weighted_ovr_auc"] = None
    result["per_class_auc"] = per_class_auc
    return result


def validate_oof_predictions(frame, assignments, predictions, n_folds=5):
    """Strictly validate one out-of-fold test prediction per manifest patch."""
    if predictions.empty:
        raise ValueError("OOF predictions contain no rows")
    missing = [column for column in PREDICTION_COLUMNS if column not in predictions]
    extra = [column for column in predictions if column not in PREDICTION_COLUMNS]
    if missing or extra:
        raise ValueError(
            "Prediction schema must be exactly {0}; missing={1}, extra={2}".format(
                PREDICTION_COLUMNS, missing, extra
            )
        )
    values = predictions.loc[:, list(PREDICTION_COLUMNS)].copy()
    if values.isnull().any().any():
        counts = values.isnull().sum()
        raise ValueError(
            "Predictions contain null values: {0}".format(
                {key: int(value) for key, value in counts.items() if value}
            )
        )

    text_columns = (
        "patch_id",
        "case_id",
        "image_path",
        "split",
        "y_true_label",
        "y_pred_label",
    )
    for column in text_columns:
        values[column] = values[column].astype(str)
        if values[column].str.strip().eq("").any():
            raise ValueError("Prediction column {0} contains empty values".format(column))
    if set(values["split"]) != {"test"}:
        raise ValueError(
            "All OOF prediction split values must be test; found {0}".format(
                sorted(values["split"].unique().tolist())
            )
        )

    for column in ("fold", "y_true_idx", "y_pred_idx"):
        numeric = pd.to_numeric(values[column], errors="coerce")
        if numeric.isnull().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("Prediction column {0} must contain integers".format(column))
        values[column] = numeric.astype(np.int64)
    invalid_folds = sorted(set(values["fold"]) - set(range(n_folds)))
    if invalid_folds:
        raise ValueError(
            "Prediction fold values must be in 0..{0}; found {1}".format(
                n_folds - 1, invalid_folds
            )
        )
    for column in ("y_true_idx", "y_pred_idx"):
        invalid = sorted(set(values[column]) - set(range(len(LABELS))))
        if invalid:
            raise ValueError(
                "{0} values must be in 0..{1}; found {2}".format(
                    column, len(LABELS) - 1, invalid
                )
            )

    for column in ("patch_id", "image_path"):
        duplicated = values.loc[values[column].duplicated(keep=False), column]
        if not duplicated.empty:
            raise ValueError(
                "Duplicate prediction {0} values found: {1}".format(
                    column, sorted(duplicated.unique())[:5]
                )
            )
    if not values["patch_id"].equals(values["image_path"]):
        raise ValueError("Prediction patch_id must equal image_path for every row")

    manifest = frame.loc[:, list(CSV_COLUMNS)].copy()
    manifest["case_id"] = manifest["case_id"].astype(str)
    manifest["image_path"] = manifest["image_path"].astype(str)
    manifest["label"] = manifest["label"].astype(str)
    expected_paths = set(manifest["image_path"])
    actual_paths = set(values["image_path"])
    missing_paths = sorted(expected_paths - actual_paths)
    unknown_paths = sorted(actual_paths - expected_paths)
    if missing_paths or unknown_paths or len(values) != len(manifest):
        raise ValueError(
            "OOF patch coverage must match the manifest exactly; "
            "expected={0}, actual={1}, missing={2}, unknown={3}".format(
                len(manifest), len(values), missing_paths[:5], unknown_paths[:5]
            )
        )
    expected = manifest.set_index("image_path")
    aligned = expected.loc[values["image_path"]]
    expected_cases = aligned["case_id"].to_numpy()
    expected_labels = aligned["label"].to_numpy()
    if not np.array_equal(values["case_id"].to_numpy(), expected_cases):
        raise ValueError("Prediction case_id does not match the manifest image_path")
    if not np.array_equal(values["y_true_label"].to_numpy(), expected_labels):
        raise ValueError("Prediction y_true_label does not match the manifest label")

    label_to_idx = {label: index for index, label in enumerate(LABELS)}
    expected_true_idx = np.asarray(
        [label_to_idx[label] for label in expected_labels], dtype=np.int64
    )
    if not np.array_equal(values["y_true_idx"].to_numpy(), expected_true_idx):
        raise ValueError("Prediction y_true_idx does not match y_true_label")
    expected_true_labels = np.asarray(
        [LABELS[index] for index in values["y_true_idx"]]
    )
    if not np.array_equal(values["y_true_label"].to_numpy(), expected_true_labels):
        raise ValueError("Prediction y_true_label does not match y_true_idx")

    role_rows = assignments.loc[
        assignments["role"] == "test", ["fold", "case_id"]
    ].copy()
    role_rows["case_id"] = role_rows["case_id"].astype(str)
    if role_rows["case_id"].duplicated().any():
        raise ValueError("Every case must have only one test fold")
    test_fold_by_case = role_rows.set_index("case_id")["fold"].to_dict()
    unknown_cases = sorted(set(values["case_id"]) - set(test_fold_by_case))
    if unknown_cases:
        raise ValueError(
            "Prediction cases have no test assignment: {0}".format(unknown_cases)
        )
    expected_folds = values["case_id"].map(test_fold_by_case).astype(np.int64)
    if not np.array_equal(values["fold"].to_numpy(), expected_folds.to_numpy()):
        raise ValueError("Prediction case is not role=test in its stated fold")
    if values.groupby("case_id")["fold"].nunique().gt(1).any():
        raise ValueError("A prediction case occurs in more than one fold")

    probability_columns = ["prob_{0}".format(label) for label in LABELS]
    probabilities = values[probability_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("Prediction probabilities must all be finite numbers")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Prediction probabilities must be within [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Prediction probability rows must sum to 1")
    predicted_idx = probabilities.argmax(axis=1).astype(np.int64)
    if not np.array_equal(values["y_pred_idx"].to_numpy(), predicted_idx):
        raise ValueError("Prediction y_pred_idx does not match probability argmax")
    predicted_labels = np.asarray([LABELS[index] for index in predicted_idx])
    if not np.array_equal(values["y_pred_label"].to_numpy(), predicted_labels):
        raise ValueError("Prediction y_pred_label does not match y_pred_idx")

    confidence = pd.to_numeric(values["confidence"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(confidence).all() or not np.allclose(
        confidence, probabilities.max(axis=1), rtol=1e-6, atol=1e-6
    ):
        raise ValueError("Prediction confidence must equal maximum probability")
    correct_text = values["correct"].astype(str).str.lower()
    if not correct_text.isin(("true", "false")).all():
        raise ValueError("Prediction correct values must be true or false")
    correct = correct_text.eq("true").to_numpy()
    expected_correct = values["y_true_idx"].to_numpy() == predicted_idx
    if not np.array_equal(correct, expected_correct):
        raise ValueError("Prediction correct does not match true/predicted indices")

    values.loc[:, probability_columns] = probabilities
    values["confidence"] = confidence
    values["correct"] = correct
    return values.sort_values(
        ["fold", "case_id", "image_path"], kind="mergesort"
    ).reset_index(drop=True)


def patch_metric_summary(y_true, probabilities, scope="oof_patch"):
    """Return long-form patch classification and safe OVR AUC/PRC metrics."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        cohen_kappa_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or len(y_true) == 0:
        raise ValueError("y_true must be a non-empty one-dimensional array")
    if probabilities.shape != (len(y_true), len(LABELS)):
        raise ValueError(
            "probabilities must have shape ({0}, {1})".format(
                len(y_true), len(LABELS)
            )
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    y_pred = probabilities.argmax(axis=1)
    n_samples = int(len(y_true))
    rows = []

    def add(metric, value, class_name="", average="", status="ok",
            n_positive=np.nan, n_negative=np.nan, n_defined_classes=np.nan):
        rows.append(
            {
                "scope": scope,
                "metric": metric,
                "class": class_name,
                "average": average,
                "value": float(value) if value is not None else np.nan,
                "n_samples": n_samples,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "n_defined_classes": n_defined_classes,
                "status": status,
            }
        )

    add("accuracy", accuracy_score(y_true, y_pred))
    for metric, function in (
        ("precision", precision_score),
        ("recall", recall_score),
        ("f1", f1_score),
    ):
        for average in ("macro", "weighted"):
            add(
                metric,
                function(y_true, y_pred, average=average, zero_division=0),
                average=average,
            )
    add("kappa", cohen_kappa_score(y_true, y_pred))

    defined_values = []
    defined_prc_values = []
    class_counts = []
    for index, label in enumerate(LABELS):
        binary = (y_true == index).astype(np.int64)
        n_positive = int(binary.sum())
        n_negative = n_samples - n_positive
        if n_positive == 0 or n_negative == 0:
            value = None
            prc_value = None
            status = "single_class"
        else:
            value = roc_auc_score(binary, probabilities[:, index])
            prc_value = average_precision_score(binary, probabilities[:, index])
            status = "ok"
            defined_values.append(float(value))
            defined_prc_values.append(float(prc_value))
            class_counts.append(n_positive)
        add(
            "ovr_auc",
            value,
            class_name=label,
            average="none",
            status=status,
            n_positive=n_positive,
            n_negative=n_negative,
            n_defined_classes=1 if status == "ok" else 0,
        )
        add(
            "ovr_average_precision",
            prc_value,
            class_name=label,
            average="none",
            status=status,
            n_positive=n_positive,
            n_negative=n_negative,
            n_defined_classes=1 if status == "ok" else 0,
        )
        if label == "positive":
            add(
                "positive_vs_rest_auc",
                value,
                class_name=label,
                average="binary",
                status=status,
                n_positive=n_positive,
                n_negative=n_negative,
                n_defined_classes=1 if status == "ok" else 0,
            )
            add(
                "positive_vs_rest_average_precision",
                prc_value,
                class_name=label,
                average="binary",
                status=status,
                n_positive=n_positive,
                n_negative=n_negative,
                n_defined_classes=1 if status == "ok" else 0,
            )

    all_defined = len(defined_values) == len(LABELS)
    aggregate_status = "ok" if all_defined else "single_class"
    macro_value = float(np.mean(defined_values)) if all_defined else None
    weighted_value = (
        float(np.average(defined_values, weights=class_counts))
        if all_defined
        else None
    )
    macro_prc = float(np.mean(defined_prc_values)) if all_defined else None
    weighted_prc = (
        float(np.average(defined_prc_values, weights=class_counts))
        if all_defined
        else None
    )
    add(
        "ovr_auc",
        macro_value,
        average="macro",
        status=aggregate_status,
        n_defined_classes=len(defined_values),
    )
    add(
        "ovr_auc",
        weighted_value,
        average="weighted",
        status=aggregate_status,
        n_defined_classes=len(defined_values),
    )
    add(
        "ovr_average_precision",
        macro_prc,
        average="macro",
        status=aggregate_status,
        n_defined_classes=len(defined_prc_values),
    )
    add(
        "ovr_average_precision",
        weighted_prc,
        average="weighted",
        status=aggregate_status,
        n_defined_classes=len(defined_prc_values),
    )
    return pd.DataFrame(
        rows,
        columns=(
            "scope",
            "metric",
            "class",
            "average",
            "value",
            "n_samples",
            "n_positive",
            "n_negative",
            "n_defined_classes",
            "status",
        ),
    )


def _til_correlation_summary(frame, prediction_column):
    valid = frame[
        np.isfinite(frame["gt_til_score"])
        & np.isfinite(frame[prediction_column])
    ]
    n_valid = int(len(valid))
    status = "ok"
    spearman = np.nan
    pearson = np.nan
    if n_valid == 0:
        status = "no_valid_slides"
    elif n_valid < 2:
        status = "insufficient_pairs"
    else:
        gt_values = valid["gt_til_score"].to_numpy(dtype=np.float64)
        pred_values = valid[prediction_column].to_numpy(dtype=np.float64)
        if np.ptp(gt_values) == 0.0 or np.ptp(pred_values) == 0.0:
            status = "constant_input"
        else:
            pearson = float(np.corrcoef(gt_values, pred_values)[0, 1])
            gt_rank = pd.Series(gt_values).rank(method="average").to_numpy()
            pred_rank = pd.Series(pred_values).rank(method="average").to_numpy()
            spearman = float(np.corrcoef(gt_rank, pred_rank)[0, 1])
    return {
        "valid": valid,
        "n_valid": n_valid,
        "status": status,
        "spearman": spearman,
        "pearson": pearson,
    }


def slide_til_score_summary(predictions):
    """Compute per-case hard and probability-weighted TIL scores."""
    required = {"case_id", "fold", "y_true_label", "y_pred_label"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError("Slide TIL input is missing columns: {0}".format(missing))
    if predictions.empty:
        raise ValueError("Slide TIL input contains no rows")
    probability_columns = {"prob_positive", "prob_negative"}
    has_probabilities = probability_columns.issubset(predictions.columns)

    rows = []
    for case_id, group in predictions.groupby("case_id", sort=True):
        folds = group["fold"].unique()
        if len(folds) != 1:
            raise ValueError("Case {0} occurs in multiple folds".format(case_id))
        gt = group["y_true_label"].value_counts()
        pred = group["y_pred_label"].value_counts()
        gt_positive = int(gt.get("positive", 0))
        gt_negative = int(gt.get("negative", 0))
        pred_positive = int(pred.get("positive", 0))
        pred_negative = int(pred.get("negative", 0))
        gt_denominator = gt_positive + gt_negative
        pred_denominator = pred_positive + pred_negative
        gt_score = (
            gt_positive / float(gt_denominator)
            if gt_denominator
            else np.nan
        )
        pred_score = (
            pred_positive / float(pred_denominator)
            if pred_denominator
            else np.nan
        )
        if has_probabilities:
            soft_positive = float(
                pd.to_numeric(group["prob_positive"], errors="coerce").sum()
            )
            soft_negative = float(
                pd.to_numeric(group["prob_negative"], errors="coerce").sum()
            )
            soft_denominator = soft_positive + soft_negative
            soft_score = (
                soft_positive / soft_denominator
                if soft_denominator > 0 and np.isfinite(soft_denominator)
                else np.nan
            )
        else:
            soft_positive = np.nan
            soft_negative = np.nan
            soft_score = np.nan
        valid = np.isfinite(gt_score) and np.isfinite(pred_score)
        soft_valid = np.isfinite(gt_score) and np.isfinite(soft_score)
        rows.append(
            {
                "row_type": "case",
                "case_id": str(case_id),
                "fold": int(folds[0]),
                "n_patches": int(len(group)),
                "gt_positive": gt_positive,
                "gt_negative": gt_negative,
                "gt_other": int(gt.get("other", 0)),
                "pred_positive": pred_positive,
                "pred_negative": pred_negative,
                "pred_other": int(pred.get("other", 0)),
                "gt_til_score": gt_score,
                "pred_til_score": pred_score,
                "abs_error": abs(gt_score - pred_score) if valid else np.nan,
                "soft_positive_sum": soft_positive,
                "soft_negative_sum": soft_negative,
                "soft_pred_til_score": soft_score,
                "soft_abs_error": (
                    abs(gt_score - soft_score) if soft_valid else np.nan
                ),
                "n_valid_slides": np.nan,
                "mae": np.nan,
                "median_ae": np.nan,
                "spearman_r": np.nan,
                "pearson_r": np.nan,
                "soft_mae": np.nan,
                "soft_median_ae": np.nan,
                "soft_spearman_r": np.nan,
                "soft_pearson_r": np.nan,
                "soft_status": "ok" if soft_valid else "unavailable",
                "status": "ok" if valid else "zero_denominator",
            }
        )

    case_frame = pd.DataFrame(rows)
    hard = _til_correlation_summary(case_frame, "pred_til_score")
    soft = _til_correlation_summary(case_frame, "soft_pred_til_score")
    valid = hard["valid"]
    soft_valid = soft["valid"]
    overall = {
        "row_type": "overall",
        "case_id": "",
        "fold": np.nan,
        "n_patches": int(case_frame["n_patches"].sum()),
        "gt_positive": int(case_frame["gt_positive"].sum()),
        "gt_negative": int(case_frame["gt_negative"].sum()),
        "gt_other": int(case_frame["gt_other"].sum()),
        "pred_positive": int(case_frame["pred_positive"].sum()),
        "pred_negative": int(case_frame["pred_negative"].sum()),
        "pred_other": int(case_frame["pred_other"].sum()),
        "gt_til_score": np.nan,
        "pred_til_score": np.nan,
        "abs_error": np.nan,
        "soft_positive_sum": float(case_frame["soft_positive_sum"].sum()),
        "soft_negative_sum": float(case_frame["soft_negative_sum"].sum()),
        "soft_pred_til_score": np.nan,
        "soft_abs_error": np.nan,
        "n_valid_slides": hard["n_valid"],
        "mae": (
            float(valid["abs_error"].mean())
            if hard["n_valid"]
            else np.nan
        ),
        "median_ae": (
            float(valid["abs_error"].median())
            if hard["n_valid"]
            else np.nan
        ),
        "spearman_r": hard["spearman"],
        "pearson_r": hard["pearson"],
        "soft_mae": (
            float(soft_valid["soft_abs_error"].mean())
            if soft["n_valid"]
            else np.nan
        ),
        "soft_median_ae": (
            float(soft_valid["soft_abs_error"].median())
            if soft["n_valid"]
            else np.nan
        ),
        "soft_spearman_r": soft["spearman"],
        "soft_pearson_r": soft["pearson"],
        "soft_status": soft["status"],
        "status": hard["status"],
    }
    return pd.concat(
        [case_frame, pd.DataFrame([overall], columns=case_frame.columns)],
        ignore_index=True,
    )


def cross_fitted_linear_til_calibration(slide_summary):
    """Leave-one-case-out linear calibration for hard and soft TIL scores."""
    cases = slide_summary[slide_summary["row_type"] == "case"].copy()
    required = {
        "case_id",
        "gt_til_score",
        "pred_til_score",
        "soft_pred_til_score",
    }
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError("Calibration input is missing columns: {0}".format(missing))
    if len(cases) < 3:
        raise ValueError("Calibration requires at least three cases")

    rows = []
    for index, target in cases.iterrows():
        train = cases.drop(index=index)
        row = {
            "case_id": str(target["case_id"]),
            "gt_til_score": float(target["gt_til_score"]),
        }
        for prefix, column in (
            ("hard", "pred_til_score"),
            ("soft", "soft_pred_til_score"),
        ):
            fit = train[
                np.isfinite(train["gt_til_score"])
                & np.isfinite(train[column])
            ]
            raw = float(target[column])
            if (
                len(fit) < 2
                or not np.isfinite(raw)
                or np.ptp(fit[column].to_numpy(dtype=np.float64)) == 0.0
            ):
                calibrated = np.nan
                slope = np.nan
                intercept = np.nan
                status = "insufficient_calibration_data"
            else:
                slope, intercept = np.polyfit(
                    fit[column].to_numpy(dtype=np.float64),
                    fit["gt_til_score"].to_numpy(dtype=np.float64),
                    1,
                )
                calibrated = float(np.clip(slope * raw + intercept, 0.0, 1.0))
                status = "ok"
            row["{0}_raw".format(prefix)] = raw
            row["{0}_calibrated".format(prefix)] = calibrated
            row["{0}_slope".format(prefix)] = slope
            row["{0}_intercept".format(prefix)] = intercept
            row["{0}_abs_error".format(prefix)] = (
                abs(float(target["gt_til_score"]) - calibrated)
                if np.isfinite(calibrated)
                else np.nan
            )
            row["{0}_status".format(prefix)] = status
        rows.append(row)
    return pd.DataFrame(rows)
