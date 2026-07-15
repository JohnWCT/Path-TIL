#!/usr/bin/env python3
"""Train the HNSCC IRV2 classifier with fixed case-grouped cross-validation."""

import argparse
import gc
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.hnscc import (  # noqa: E402
    ASSIGNMENT_COLUMNS,
    LABELS,
    balanced_class_weights,
    classification_metrics,
    fold_split_details,
    load_hnscc_csv,
    validate_fold_assignments,
)


LABEL_TO_IDX = {label: index for index, label in enumerate(LABELS)}
IMG_SIZE = 224
NUM_CLASSES = len(LABELS)


def on_off(value):
    return value == "on"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage InceptionResNetV2 training with fixed HNSCC case-grouped folds"
        )
    )
    parser.add_argument("--csv", required=True, help="HNSCC patch manifest CSV")
    parser.add_argument(
        "--fold-csv", required=True, help="Case-level fold,case_id,role CSV"
    )
    parser.add_argument(
        "--pretrained", required=True, help="Pretrained four-layer Keras .h5 model"
    )
    parser.add_argument("--output-dir", required=True, help="Output root directory")
    parser.add_argument(
        "--fold",
        action="append",
        type=int,
        default=None,
        help="Fold to run (0..4); repeat to select multiple (default: all)",
    )
    parser.add_argument(
        "--aug",
        choices=("none", "light", "medium", "heavy"),
        default="heavy",
        help="Training-only augmentation level (default: heavy)",
    )
    parser.add_argument(
        "--hne-norm",
        choices=("on", "off"),
        default="on",
        help="Conditional H&E normalization before resize (default: on)",
    )
    parser.add_argument(
        "--class-weight",
        choices=("on", "off"),
        default="on",
        help="Balanced train-split class weights (default: on)",
    )
    parser.add_argument("--epochs-stage1", type=int, default=30)
    parser.add_argument("--epochs-stage2", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
        help="CPU threads used to load images",
    )
    parser.add_argument(
        "--fit-workers",
        type=int,
        default=1,
        help="Keras Sequence workers for heavy augmentation (default: 1)",
    )
    parser.add_argument(
        "--use-multiprocessing",
        choices=("on", "off"),
        default="off",
        help="Use multiprocessing for heavy Sequence workers (default: off)",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("on", "off"),
        default="off",
        help="Enable mixed_float16 only if loaded model output remains float32",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/splits and write configs without importing TensorFlow",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Evaluate pretrained Stage 0 on train/val/test without fine-tuning",
    )
    return parser.parse_args()


def validate_args(args):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.epochs_stage1 < 1 or args.epochs_stage2 < 1:
        raise ValueError("Stage epoch counts must be positive")
    if args.image_workers < 1 or args.fit_workers < 1:
        raise ValueError("Worker counts must be positive")
    if args.patience < 1:
        raise ValueError("--patience must be positive")
    for name in ("csv", "fold_csv", "pretrained"):
        path = Path(getattr(args, name))
        if not path.is_file():
            raise FileNotFoundError("{0} does not exist: {1}".format(name, path))


def load_assignments(path):
    assignments = pd.read_csv(path)
    if list(assignments.columns) != list(ASSIGNMENT_COLUMNS):
        raise ValueError(
            "Fold CSV columns must be exactly {0}; found {1}".format(
                ASSIGNMENT_COLUMNS, assignments.columns.tolist()
            )
        )
    if assignments.isnull().any().any():
        raise ValueError("Fold CSV contains null values")
    try:
        assignments["fold"] = assignments["fold"].astype(int)
    except (TypeError, ValueError):
        raise ValueError("Fold CSV fold values must be integers")
    assignments["case_id"] = assignments["case_id"].astype(str)
    assignments["role"] = assignments["role"].astype(str)
    return assignments


def selected_folds(values):
    folds = list(range(5)) if values is None else sorted(set(values))
    invalid = [fold for fold in folds if fold not in range(5)]
    if invalid:
        raise ValueError("--fold must be in 0..4; found {0}".format(invalid))
    if not folds:
        raise ValueError("At least one fold must be selected")
    return folds


def check_image_paths(frame):
    missing = [
        str(path)
        for path in frame["image_path"].tolist()
        if not Path(str(path)).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "{0} image paths do not exist; examples: {1}".format(
                len(missing), missing[:10]
            )
        )


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def split_indices(frame, assignments, fold):
    fold_rows = assignments[assignments["fold"] == fold]
    role_by_case = dict(zip(fold_rows["case_id"], fold_rows["role"]))
    roles = frame["case_id"].map(role_by_case)
    if roles.isnull().any():
        raise ValueError("Fold {0} has unmapped cases".format(fold))
    return {
        role: np.flatnonzero(roles.to_numpy() == role)
        for role in ("train", "val", "test")
    }


def make_fold_config(args, frame, assignments, fold, folds):
    details = fold_split_details(frame, assignments, fold)
    weights = (
        balanced_class_weights(
            frame[
                frame["case_id"].isin(details["train"]["cases"])
            ]["label"].tolist()
        )
        if on_off(args.class_weight)
        else None
    )
    return {
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in ("fold",)
        },
        "selected_folds": folds,
        "fold": fold,
        "label_to_index": LABEL_TO_IDX,
        "splits": details,
        "class_weights": weights,
        "image_count": int(len(frame)),
        "case_count": int(frame["case_id"].nunique()),
        "preprocessing": {
            "color": "cv2 BGR to RGB",
            "hne_norm": on_off(args.hne_norm),
            "hne_condition": "mean < 230 and std > 15",
            "hne_parameters": {"Io": 240, "alpha": 1, "beta": 0.15},
            "resize": [IMG_SIZE, IMG_SIZE],
            "model_scale": "float32 / 255",
        },
    }


def load_all_images(paths, use_hne_norm, workers):
    import cv2
    from normalize_HnE import norm_HnE

    def load_one(path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError("Unable to read image: {0}".format(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (
            use_hne_norm
            and float(image.mean()) < 230.0
            and float(image.std()) > 15.0
        ):
            try:
                with np.errstate(divide="raise", invalid="raise", over="raise"):
                    normalized, _, _ = norm_HnE(
                        image, Io=240, alpha=1, beta=0.15
                    )
                if normalized.shape != image.shape:
                    raise ValueError(
                        "normalized shape {0} != input shape {1}".format(
                            normalized.shape, image.shape
                        )
                    )
                image = normalized
                norm_status = "applied"
                norm_error = None
            except Exception as error:
                norm_status = "failed"
                norm_error = "{0}: {1}".format(type(error).__name__, error)
        else:
            norm_status = "skipped"
            norm_error = None
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        return np.asarray(image, dtype=np.uint8), norm_status, norm_error, str(path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        loaded = list(executor.map(load_one, paths))
    images = [item[0] for item in loaded]
    statuses = [item[1] for item in loaded]
    failures = [
        {"image_path": item[3], "reason": item[2]}
        for item in loaded
        if item[1] == "failed"
    ]
    report = {
        "total": len(loaded),
        "applied": statuses.count("applied"),
        "skipped": statuses.count("skipped"),
        "failed": len(failures),
        "failure_examples": failures[:20],
    }
    print("H&E normalization report: {0}".format(report))
    return np.asarray(images, dtype=np.uint8), report


def configure_tensorflow(args):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    if not tf.__version__.startswith("2.10"):
        print(
            "WARNING: Designed for TensorFlow 2.10; running {0}.".format(
                tf.__version__
            ),
            file=sys.stderr,
        )
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError) as error:
            print(
                "WARNING: Could not enable GPU memory growth: {0}".format(error),
                file=sys.stderr,
            )
    if on_off(args.mixed_precision):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print(
            "WARNING: mixed_float16 enabled; training will stop if the loaded "
            "classifier output is not float32.",
            file=sys.stderr,
        )
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    return tf


def sparse_auc_class(tf):
    class SparseMulticlassAUC(tf.keras.metrics.Metric):
        def __init__(self, num_classes=NUM_CLASSES, name="auc", **kwargs):
            super().__init__(name=name, **kwargs)
            self.num_classes = int(num_classes)
            self.inner_auc = tf.keras.metrics.AUC(
                multi_label=True,
                num_labels=self.num_classes,
                from_logits=False,
                name=name + "_inner",
            )

        def update_state(self, y_true, y_pred, sample_weight=None):
            values = tf.reshape(tf.cast(y_true, tf.int32), [-1])
            one_hot = tf.one_hot(values, depth=self.num_classes, dtype=tf.float32)
            self.inner_auc.update_state(
                one_hot, tf.cast(y_pred, tf.float32), sample_weight
            )

        def result(self):
            return self.inner_auc.result()

        def reset_state(self):
            if hasattr(self.inner_auc, "reset_state"):
                self.inner_auc.reset_state()
            else:
                self.inner_auc.reset_states()

        def reset_states(self):
            self.reset_state()

        def get_config(self):
            config = super().get_config()
            config.update({"num_classes": self.num_classes})
            return config

    return SparseMulticlassAUC


def load_and_validate_model(tf, path, mixed_precision):
    model = tf.keras.models.load_model(path, compile=False)
    if len(model.layers) != 4:
        raise ValueError(
            "Expected exactly 4 model layers, found {0}".format(len(model.layers))
        )
    output_shape = model.output_shape
    if isinstance(output_shape, list) or int(output_shape[-1]) != NUM_CLASSES:
        raise ValueError("Model must have one final output with 3 classes")
    output_dtype = tf.as_dtype(model.output.dtype)
    if mixed_precision and output_dtype != tf.float32:
        raise ValueError(
            "mixed_float16 made the loaded model output {0}; refusing to alter "
            "the checkpoint architecture. Use --mixed-precision off.".format(
                output_dtype.name
            )
        )
    if mixed_precision:
        compute_dtype = getattr(model.layers[0].dtype_policy, "compute_dtype", None)
        if compute_dtype != "float16":
            print(
                "WARNING: The loaded H5 backbone retains compute dtype {0}; "
                "checkpoint architecture is preserved, so mixed precision may "
                "provide no acceleration.".format(compute_dtype),
                file=sys.stderr,
            )
    return model


def compile_model(tf, model, learning_rate, auc_class):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            auc_class(NUM_CLASSES, name="auc"),
        ],
    )


def build_tf_augmentation(tf, level):
    layers = tf.keras.layers
    if level == "none":
        return None
    values = [
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
    ]
    if level == "medium":
        values.extend(
            [
                layers.RandomRotation(0.15),
                layers.RandomZoom(0.12),
                layers.RandomTranslation(0.12, 0.12),
                layers.RandomContrast(0.12),
            ]
        )
    return tf.keras.Sequential(values, name="train_only_augmentation")


def make_tf_dataset(tf, images, labels, batch_size, training, augmentation, seed):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(
            max(1, len(images)), seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.batch(batch_size)

    def prepare(batch_images, batch_labels):
        values = tf.cast(batch_images, tf.float32) / 255.0
        if training and augmentation is not None:
            values = augmentation(values, training=True)
        return values, batch_labels

    dataset = dataset.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_imgaug_sequence_class(tf):
    class ImgAugSequence(tf.keras.utils.Sequence):
        def __init__(
            self, images, labels, batch_size, shuffle=False, augment=False, seed=42
        ):
            self.images = images
            self.labels = labels
            self.batch_size = int(batch_size)
            self.shuffle = bool(shuffle)
            self.augment = bool(augment)
            self.random = np.random.RandomState(seed)
            self.indices = np.arange(len(images))
            self.pipeline = self._build_pipeline() if augment else None
            self.on_epoch_end()

        @staticmethod
        def _build_pipeline():
            import imgaug.augmenters as iaa
            from skimage.color import hed2rgb, rgb2hed

            def rgb2hed_func(images, random_state, parents, hooks):
                return [rgb2hed(image).astype("float32") for image in images]

            def hed2rgb_func(images, random_state, parents, hooks):
                return [
                    (hed2rgb(image) * 255).astype("uint8") for image in images
                ]

            return iaa.Sequential(
                [
                    iaa.Sometimes(
                        0.5,
                        iaa.SomeOf(
                            2,
                            [
                                iaa.Fliplr(1),
                                iaa.Flipud(1),
                                iaa.Rot90((1, 3)),
                            ],
                        ),
                    ),
                    iaa.Sometimes(
                        0.5,
                        iaa.Sequential(
                            [
                                iaa.Lambda(rgb2hed_func),
                                iaa.WithChannels(0, iaa.Multiply((0.45, 1.90))),
                                iaa.WithChannels(1, iaa.Multiply((0.5, 2.45))),
                                iaa.Lambda(hed2rgb_func),
                            ]
                        ),
                    ),
                    iaa.Sometimes(
                        0.5,
                        iaa.OneOf(
                            [
                                iaa.SaltAndPepper((0.0, 0.2)),
                                iaa.Salt((0.0, 0.2)),
                                iaa.Pepper((0.0, 0.12)),
                                iaa.GaussianBlur(sigma=(0.0, 1.9)),
                                iaa.AdditiveGaussianNoise(
                                    scale=(0.0, 0.14 * 255)
                                ),
                            ]
                        ),
                    ),
                    iaa.Sometimes(
                        0,
                        iaa.OneOf(
                            [
                                iaa.Add((-50, 50)),
                                iaa.Multiply((0.90, 1.34)),
                                iaa.AddToHue((-16, 8)),
                                iaa.AddToSaturation((-40, 35)),
                                iaa.MultiplyHue((0.88, 1.12)),
                                iaa.MultiplySaturation((0.75, 1.35)),
                                iaa.LinearContrast((0.80, 1.70)),
                                iaa.GammaContrast((0.80, 2.00)),
                                iaa.CLAHE(clip_limit=(0.10, 4.90)),
                                iaa.JpegCompression(compression=(0.0, 90)),
                            ]
                        ),
                    ),
                    iaa.Sometimes(
                        0.5,
                        iaa.Cutout(
                            nb_iterations=(1, 5),
                            size=0.2,
                            fill_mode="constant",
                            cval=255,
                        ),
                    ),
                ],
                random_order=True,
            )

        def __len__(self):
            return int(np.ceil(len(self.images) / float(self.batch_size)))

        def __getitem__(self, index):
            selection = self.indices[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            batch = self.images[selection].copy()
            if self.pipeline is not None:
                batch = np.asarray(self.pipeline(images=batch), dtype=np.uint8)
            return batch.astype(np.float32) / 255.0, self.labels[selection]

        def on_epoch_end(self):
            if self.shuffle:
                self.random.shuffle(self.indices)

    return ImgAugSequence


def training_inputs(tf, args, images, labels, fold, stage_seed):
    if args.aug == "heavy" and not args.baseline_only:
        sequence_class = build_imgaug_sequence_class(tf)
        train_data = sequence_class(
            images,
            labels,
            args.batch_size,
            shuffle=True,
            augment=True,
            seed=stage_seed,
        )
        fit_kwargs = {
            "workers": args.fit_workers,
            "use_multiprocessing": on_off(args.use_multiprocessing),
            "max_queue_size": max(10, args.fit_workers * 2),
        }
    else:
        augmentation = build_tf_augmentation(tf, args.aug)
        train_data = make_tf_dataset(
            tf,
            images,
            labels,
            args.batch_size,
            True,
            augmentation,
            stage_seed,
        )
        fit_kwargs = {}
    return train_data, fit_kwargs


def nonaug_sequence(tf, images, labels, batch_size):
    sequence_class = build_imgaug_sequence_class(tf)
    return sequence_class(
        images, labels, batch_size, shuffle=False, augment=False
    )


def callbacks(tf, checkpoint_path, patience):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.2,
            patience=max(1, patience // 3),
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def save_learning_curve(history, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = history.history
    epochs = np.arange(1, len(values.get("loss", [])) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    figure.suptitle(title)
    axes[0].plot(epochs, values.get("loss", []), label="train")
    axes[0].plot(epochs, values.get("val_loss", []), label="val")
    axes[0].set_ylabel("loss")
    axes[1].plot(epochs, values.get("auc", []), label="train")
    axes[1].plot(epochs, values.get("val_auc", []), label="val")
    axes[1].set_ylabel("AUC")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(figure)


def prediction_frame(frame, indices, fold, split, probabilities):
    subset = frame.iloc[indices].reset_index(drop=True)
    y_true = subset["label"].map(LABEL_TO_IDX).to_numpy(dtype=np.int32)
    y_pred = np.argmax(probabilities, axis=1).astype(np.int32)
    result = pd.DataFrame(
        {
            "patch_id": subset["image_path"],
            "case_id": subset["case_id"],
            "image_path": subset["image_path"],
            "fold": int(fold),
            "split": split,
            "y_true_idx": y_true,
            "y_true_label": [LABELS[index] for index in y_true],
            "y_pred_idx": y_pred,
            "y_pred_label": [LABELS[index] for index in y_pred],
            "prob_positive": probabilities[:, 0],
            "prob_negative": probabilities[:, 1],
            "prob_other": probabilities[:, 2],
            "confidence": np.max(probabilities, axis=1),
            "correct": y_true == y_pred,
        }
    )
    return result, y_true


def predict_split(tf, model, frame, images, indices, fold, split, args, output):
    labels = frame.iloc[indices]["label"].map(LABEL_TO_IDX).to_numpy(dtype=np.int32)
    data = nonaug_sequence(tf, images[indices], labels, args.batch_size)
    probabilities = np.asarray(model.predict(data, verbose=1), dtype=np.float64)
    predictions, y_true = prediction_frame(
        frame, indices, fold, split, probabilities
    )
    predictions.to_csv(output, index=False, float_format="%.8f")
    return classification_metrics(y_true, probabilities), predictions


def evaluate_stage_model(
    tf,
    auc_class,
    args,
    frame,
    images,
    indices,
    fold,
    stage,
    model_path,
    learning_rate,
    validation,
    fold_dir,
):
    model = load_and_validate_model(
        tf, model_path, on_off(args.mixed_precision)
    )
    compile_model(tf, model, learning_rate, auc_class)
    validation_result = model.evaluate(
        validation, verbose=0, return_dict=True
    )
    keras_val_auc = float(validation_result["auc"])
    metrics = {}
    predictions = {}
    for split in ("train", "val", "test"):
        metrics[split], predictions[split] = predict_split(
            tf,
            model,
            frame,
            images,
            indices[split],
            fold,
            split,
            args,
            fold_dir
            / "stage{0}_{1}_predictions.csv".format(stage, split),
        )
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return keras_val_auc, metrics, predictions


def train_fold(tf, auc_class, args, frame, assignments, images, fold, config):
    fold_dir = Path(args.output_dir) / "fold{0:02d}".format(fold)
    indices = split_indices(frame, assignments, fold)
    labels = frame["label"].map(LABEL_TO_IDX).to_numpy(dtype=np.int32)
    class_weights = (
        {int(key): float(value) for key, value in config["class_weights"].items()}
        if config["class_weights"] is not None
        else None
    )
    validation = nonaug_sequence(
        tf, images[indices["val"]], labels[indices["val"]], args.batch_size
    )
    stage_paths = {0: (Path(args.pretrained), 1e-3)}
    if not args.baseline_only:
        model = load_and_validate_model(
            tf, args.pretrained, on_off(args.mixed_precision)
        )
        model.layers[0].trainable = False
        compile_model(tf, model, 1e-3, auc_class)
        train_stage1, fit_stage1_kwargs = training_inputs(
            tf,
            args,
            images[indices["train"]],
            labels[indices["train"]],
            fold,
            args.seed + fold,
        )
        stage1_path = fold_dir / "stage1_best.h5"
        history1 = model.fit(
            train_stage1,
            validation_data=validation,
            epochs=args.epochs_stage1,
            callbacks=callbacks(tf, stage1_path, args.patience),
            class_weight=class_weights,
            verbose=1,
            **fit_stage1_kwargs
        )
        save_learning_curve(
            history1,
            fold_dir / "stage1_learning_curve.png",
            "Fold {0:02d} stage 1".format(fold),
        )
        del model, train_stage1
        gc.collect()

        model = load_and_validate_model(
            tf, stage1_path, on_off(args.mixed_precision)
        )
        model.layers[0].trainable = True
        compile_model(tf, model, 1e-5, auc_class)
        train_stage2, fit_stage2_kwargs = training_inputs(
            tf,
            args,
            images[indices["train"]],
            labels[indices["train"]],
            fold,
            args.seed + 1000 + fold,
        )
        stage2_path = fold_dir / "stage2_best.h5"
        history2 = model.fit(
            train_stage2,
            validation_data=validation,
            epochs=args.epochs_stage2,
            callbacks=callbacks(tf, stage2_path, args.patience),
            class_weight=class_weights,
            verbose=1,
            **fit_stage2_kwargs
        )
        save_learning_curve(
            history2,
            fold_dir / "stage2_learning_curve.png",
            "Fold {0:02d} stage 2".format(fold),
        )
        del model, train_stage2
        gc.collect()
        stage_paths.update(
            {
                1: (stage1_path, 1e-3),
                2: (stage2_path, 1e-5),
            }
        )

    stage_metrics = {}
    stage_predictions = {}
    validation_keras_auc = {}
    for stage in sorted(stage_paths):
        model_path, learning_rate = stage_paths[stage]
        (
            validation_keras_auc[stage],
            stage_metrics[stage],
            stage_predictions[stage],
        ) = evaluate_stage_model(
            tf,
            auc_class,
            args,
            frame,
            images,
            indices,
            fold,
            stage,
            model_path,
            learning_rate,
            validation,
            fold_dir,
        )

    selected_stage = max(
        validation_keras_auc,
        key=lambda stage: (validation_keras_auc[stage], -stage),
    )
    for split in ("val", "test"):
        stage_predictions[selected_stage][split].to_csv(
            fold_dir / "{0}_predictions.csv".format(split),
            index=False,
            float_format="%.8f",
        )
    print(
        "Fold {0}: selected stage {1} by validation Keras AUC: {2}".format(
            fold,
            selected_stage,
            {
                stage: round(value, 6)
                for stage, value in validation_keras_auc.items()
            },
        )
    )
    payload = {
        "fold": fold,
        "selected_stage": selected_stage,
        "validation_keras_auc": validation_keras_auc,
        "stage_metrics": stage_metrics,
        "selected_metrics": stage_metrics[selected_stage],
    }
    write_json(fold_dir / "fold_metrics.json", payload)
    del validation
    tf.keras.backend.clear_session()
    gc.collect()
    return payload


def main():
    args = parse_args()
    validate_args(args)
    frame = load_hnscc_csv(args.csv, expected_cases=10)
    assignments = load_assignments(args.fold_csv)
    validate_fold_assignments(frame, assignments, n_folds=5)
    folds = selected_folds(args.fold)
    check_image_paths(frame)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = {}
    for fold in folds:
        fold_dir = output_dir / "fold{0:02d}".format(fold)
        fold_dir.mkdir(parents=True, exist_ok=True)
        config = make_fold_config(args, frame, assignments, fold, folds)
        configs[fold] = config
        write_json(fold_dir / "config.json", config)
        print(
            "Fold {0}: train={1}, val={2}, test={3}".format(
                fold,
                config["splits"]["train"]["class_counts"],
                config["splits"]["val"]["class_counts"],
                config["splits"]["test"]["class_counts"],
            )
        )
    write_json(
        output_dir / "config.json",
        {
            "parameters": vars(args),
            "selected_folds": folds,
            "fold_configs": {
                "fold{0:02d}".format(fold): "fold{0:02d}/config.json".format(fold)
                for fold in folds
            },
        },
    )

    if args.dry_run:
        print(
            "Dry run complete: validated {0} images, 10 cases, folds {1}; "
            "TensorFlow was not imported.".format(len(frame), folds)
        )
        return

    if args.aug == "heavy":
        try:
            import imgaug  # noqa: F401
            import skimage  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "--aug heavy requires imgaug and scikit-image"
            ) from error

    tf = configure_tensorflow(args)
    auc_class = sparse_auc_class(tf)
    print("Loading {0} images once with {1} workers...".format(
        len(frame), args.image_workers
    ))
    images, preprocessing_report = load_all_images(
        frame["image_path"].tolist(), on_off(args.hne_norm), args.image_workers
    )
    for fold in folds:
        configs[fold]["preprocessing"]["runtime_report"] = preprocessing_report
        write_json(
            output_dir / "fold{0:02d}".format(fold) / "config.json",
            configs[fold],
        )
    scores = []
    for fold in folds:
        try:
            result = train_fold(
                tf, auc_class, args, frame, assignments, images, fold, configs[fold]
            )
            for stage, split_metrics in result["stage_metrics"].items():
                for split in ("train", "val", "test"):
                    row = {
                        "fold": fold,
                        "stage": int(stage),
                        "split": split,
                        "selected": int(stage) == int(result["selected_stage"]),
                    }
                    row.update(
                        {
                            key: value
                            for key, value in split_metrics[split].items()
                            if key != "per_class_auc"
                        }
                    )
                    scores.append(row)
            pd.DataFrame(scores).to_csv(
                output_dir / "fold_scores.csv",
                index=False,
                float_format="%.8f",
            )
        finally:
            tf.keras.backend.clear_session()
            gc.collect()
    print("Training complete. Results: {0}".format(output_dir))


if __name__ == "__main__":
    main()
