# -*- coding: utf-8 -*-
"""
InceptionResNetV2_QuPath_2stage.py
==================================
使用 prepare_dataset_csv.py 產生的 qupath_dataset.csv，以 best_InceptionResNetV2_model.h5
為起點做 2 階段訓練（與 TILscout.py 推論流程對齊）。

使用範例::

    python3 InceptionResNetV2_QuPath_2stage.py
    python3 InceptionResNetV2_QuPath_2stage.py --aug none
    python3 InceptionResNetV2_QuPath_2stage.py --aug medium
    python3 InceptionResNetV2_QuPath_2stage.py \\
        --csv /workspace/Path_TIL/qupath_dataset.csv \\
        --pretrained /workspace/Path_TIL/best_InceptionResNetV2_model.h5 \\
        --output-dir /workspace/Path_TIL/qupath_results \\
        --folds 5 --batch-size 32
    python3 InceptionResNetV2_QuPath_2stage.py --no-finetune
    python3 InceptionResNetV2_QuPath_2stage.py --only-finetune
    python3 InceptionResNetV2_QuPath_2stage.py --image-check
    python3 InceptionResNetV2_QuPath_2stage.py --aug heavy
    python3 InceptionResNetV2_QuPath_2stage.py --aug medium --epochs-stage1 30 --epochs-stage2 30

輸入與 TILscout.py 對齊::
    預設對每張 patch 套用 normalize_HnE.norm_HnE（Io=240, alpha=1, beta=0.15），條件與 TILscout
    process_tile 相同：僅當 mean<230 且 std>15 才標準化，否則保留原圖；**不需 whole slide**。
    之後 resize 224 → float32 /255 → model（與 TILscout 推論段之 X/255 一致）。
    若需關閉：--no-hne-norm。

標籤順序（與 TILscout 之 ['Positive','Negative','Other'] 一致）::
    索引 0 = positive, 1 = negative, 2 = other

模型架構（與原始訓練程式相同，未改圖結構）::
    Sequential: InceptionResNetV2(include_top=False, pooling='avg', input_shape=(224,224,3))
              → Flatten → Dense(512, relu) → Dense(3, softmax)
    輸入為 float32 RGB、數值 [0,1]（與 TILscout 之 X/255 一致）。

    --aug 僅作用於訓練資料管線（tf.data 或 imgaug），不插入上述 Sequential。
    預設 light 對齊 ImageDataGenerator(rescale=1/255 已由輸入完成，horizontal_flip、vertical_flip)。

輸出（--output-dir）::
    qupath_2stage_scores.csv      — 逐 fold、stage、split 指標（stage=0 為預訓練權重基線）
    qupath_2stage_cv_summary.csv  — 跨 fold 之 mean、std；主指標欄位順序以 AUC 為先
    fold{NN}_stage{1|2}_learning_curve.png — loss／AUC（Keras val_auc）學習曲線

訓練監控::
    ModelCheckpoint / EarlyStopping / ReduceLROnPlateau 皆以 **val_auc**（max）為準。
    CSV 與列印之 AUC 仍為 **sklearn roc_auc_score weighted OVR**（與 Keras val_auc 定義不同，數值可能略異）。

訓練階段（--no-finetune 與 --only-finetune 互斥；皆未指定時為 Stage1+Stage2）::
    --no-finetune   — 僅 Stage 1
    --only-finetune — 僅 Stage 2（自預訓練權重直接微調，不經凍結 backbone 階段）

資料切分（類別不平衡時）::
    train/test 使用 stratify=y；K-fold 使用 StratifiedKFold，使各 fold 驗證集約維持與整體相同的類別比例。
    要求 trainval 內每個類別至少有 --folds 筆，否則請減少 folds。
"""

# ========= 0. 匯入 =========
import os

# 須在 import tensorflow 之前：略過 C++ 層 INFO/WARNING
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import gc
import logging

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# 略過 Python 層可忽略 WARNING（如 while_loop / RngReadAndSkip / Bitcast 等 converter 提示）
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
try:
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from tensorflow import keras
from tensorflow.keras.layers import (
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from normalize_HnE import norm_HnE

# ========= 1. 命令列參數 =========
parser = argparse.ArgumentParser(
    description="InceptionResNetV2 QuPath 2-Stage Training（QuPath CSV + TILscout 相容權重）",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
使用範例:
  python3 InceptionResNetV2_QuPath_2stage.py
  python3 InceptionResNetV2_QuPath_2stage.py --aug none
  python3 InceptionResNetV2_QuPath_2stage.py --aug light
  python3 InceptionResNetV2_QuPath_2stage.py --aug medium
  python3 InceptionResNetV2_QuPath_2stage.py \\
      --csv /workspace/Path_TIL/qupath_dataset.csv --batch-size 32 --folds 5
  python3 InceptionResNetV2_QuPath_2stage.py --no-finetune
  python3 InceptionResNetV2_QuPath_2stage.py --only-finetune
  python3 InceptionResNetV2_QuPath_2stage.py --image-check
  python3 InceptionResNetV2_QuPath_2stage.py --aug heavy
  python3 InceptionResNetV2_QuPath_2stage.py --aug medium --epochs-stage1 30 --epochs-stage2 30
""",
)
parser.add_argument(
    "--csv",
    type=str,
    default="/workspace/Path_TIL/qupath_dataset.csv",
    help="資料集 CSV 路徑",
)
parser.add_argument(
    "--pretrained",
    type=str,
    default="/workspace/Path_TIL/best_InceptionResNetV2_model.h5",
    help="預訓練 Sequential .h5（TILscout 使用之模型）",
)
parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
parser.add_argument("--epochs-stage1", type=int, default=100, help="Stage 1 最大 epoch")
parser.add_argument("--epochs-stage2", type=int, default=100, help="Stage 2 最大 epoch")
parser.add_argument(
    "--folds",
    type=int,
    default=5,
    help="Stratified K-fold 折數；trainval 內每類樣本數須 ≥ folds",
)
parser.add_argument(
    "--test-ratio", type=float, default=0.2, help="測試集比例（影像層級 stratify）"
)
parser.add_argument("--seed", type=int, default=42, help="隨機種子")
parser.add_argument(
    "--output-dir",
    type=str,
    default="/workspace/Path_TIL/qupath_results",
    help="輸出目錄",
)
_stage_mode = parser.add_mutually_exclusive_group()
_stage_mode.add_argument(
    "--no-finetune",
    action="store_true",
    help="僅 Stage 1（凍結 backbone），不進行 Stage 2 微調",
)
_stage_mode.add_argument(
    "--only-finetune",
    action="store_true",
    help="僅 Stage 2：跳過 Stage 1，直接從預訓練權重微調全模型（backbone 可訓練）",
)
parser.add_argument(
    "--aug",
    type=str,
    default="light",
    choices=("none", "light", "medium", "heavy"),
    help="訓練資料增強（不屬於 Sequential 本體）。預設 light=等同 ImageDataGenerator 之"
    " horizontal_flip+vertical_flip（rescale 已由 [0,1] 輸入對齊）；none=無增強；"
    "medium=較強幾何/對比；heavy=imgaug",
)
parser.add_argument(
    "--image-check",
    action="store_true",
    help="啟用逐張 cv2 讀取檢查（預設略過以加快啟動；路徑是否存在仍會檢查）",
)
parser.add_argument(
    "--no-hne-norm",
    action="store_true",
    help="停用 H&E 標準化（norm_HnE）；預設啟用，與 TILscout patch 生成段一致",
)
args = parser.parse_args()

# ========= 2. 常數 =========
IMG_SIZE = 224
IMG_CH = 3
NUM_CLASS = 3
SEED = args.seed
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(args.output_dir, exist_ok=True)

if args.no_hne_norm:
    print("▶  H&E 標準化：已停用（--no-hne-norm）")
else:
    print(
        "▶  H&E 標準化：啟用 norm_HnE（Io=240, α=1, β=0.15；"
        "僅 mean<230 且 std>15 之 patch 才標準化，與 TILscout process_tile 一致；不需 whole slide）"
    )

# TILscout: columns=['Positive','Negative','Other'] → 與小寫 CSV 對應
LABEL_TO_IDX = {"positive": 0, "negative": 1, "other": 2}
CLASS_NAMES_DISPLAY = ["Positive", "Negative", "Other"]


class SparseMulticlassAUC(keras.metrics.Metric):
    """
    sparse 整數標籤 + softmax 機率 → one-hot 後以 Keras AUC(multi_label) 近似驗證曲線，
    供 ModelCheckpoint / EarlyStopping 監控 val_auc（與 sklearn OVR 數值可能略有差異）。
    """

    def __init__(self, num_classes: int = NUM_CLASS, name: str = "auc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self._auc = keras.metrics.AUC(
            multi_label=True,
            num_labels=self.num_classes,
            from_logits=False,
            name=name + "_inner",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y = tf.clip_by_value(y, 0, self.num_classes - 1)
        y_oh = tf.cast(tf.one_hot(y, depth=self.num_classes), tf.float32)
        self._auc.update_state(y_oh, y_pred, sample_weight)

    def result(self):
        return self._auc.result()

    def reset_state(self):
        _m = self._auc
        if hasattr(_m, "reset_state"):
            _m.reset_state()
        else:
            _m.reset_states()

    def reset_states(self):
        # Keras 新版偏好 reset_state；保留別名以相容舊呼叫點
        self.reset_state()


def train_metrics_list():
    return [
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        SparseMulticlassAUC(NUM_CLASS, name="auc"),
    ]


def to_tilscout_input(images_rgb):
    """
    與 TILscout.py 相同：RGB uint8 → X_pred = np.array(pred_images) / 255.0
    """
    return np.asarray(images_rgb, dtype=np.float32) / 255.0


def normalize_label(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().lower()
    return t if t in LABEL_TO_IDX else None


# ========= 3. optional imgaug（heavy） =========
def _build_imgaug_seq():
    """與 InceptionResNetV2_cross_val_aug.py 相同理念的 imgaug 管線。"""
    from skimage.color import hed2rgb, rgb2hed
    import imgaug.augmenters as iaa

    def rgb2hed_func(images, random_state, parents, hooks):
        return [rgb2hed(img).astype("float32") for img in images]

    def hed2rgb_func(images, random_state, parents, hooks):
        return [(hed2rgb(img) * 255).astype("uint8") for img in images]

    return iaa.Sequential(
        [
            iaa.Sometimes(
                0.5,
                iaa.SomeOf(2, [iaa.Fliplr(1), iaa.Flipud(1), iaa.Rot90((1, 3))]),
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
                        iaa.AdditiveGaussianNoise(scale=(0.0, 0.14 * 255)),
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


class ImgAugTrainSequence(keras.utils.Sequence):
    """訓練用：batch 內做 imgaug；驗證請用 augment=False。"""

    def __init__(self, x, y, batch_size, shuffle=True, augment=True):
        self.x = x
        self.y = y
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.augment = augment
        self._seq = _build_imgaug_seq() if augment else None
        self.indices = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        sl = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.x[sl].copy()
        batch_y = self.y[sl]
        if self.augment and self._seq is not None:
            batch_x = self._seq(images=batch_x)
            batch_x = np.asarray(batch_x, dtype=np.uint8)
        # 與 TILscout：增強後仍為 RGB，再 /255 送入模型
        batch_x = to_tilscout_input(batch_x)
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ========= 4. 讀取 CSV =========
print("▶  讀取 CSV ...")
df = pd.read_csv(args.csv)
required = {"case_id", "image_path", "label"}
if not required.issubset(df.columns):
    raise ValueError(f"CSV 需含欄位 {required}，目前: {df.columns.tolist()}")

df["label_norm"] = df["label"].map(normalize_label)
bad = df["label_norm"].isna()
if bad.any():
    examples = df.loc[bad, "label"].unique()[:10]
    raise ValueError(
        f"發現無法對應的 label（須為 positive/negative/other，不分大小寫）: {list(examples)}"
    )
df["label"] = df["label_norm"]
df = df.drop(columns=["label_norm"])

print(f"  ➜ 共 {len(df)} 筆, {df['case_id'].nunique()} 病例")
print(f"  ➜ 類別分佈:\n{df['label'].value_counts().to_string()}")
print(f"  ➜ 與 TILscout 神經元順序: {CLASS_NAMES_DISPLAY} ↔ 索引 {list(range(NUM_CLASS))}")

# ========= 5. 影像檢查 =========
image_paths = df["image_path"].tolist()
num_total = len(image_paths)
missing = [fp for fp in image_paths if not os.path.exists(fp)]
if missing:
    for f in missing[:10]:
        print(f"    - {f}")
    raise FileNotFoundError(
        f"{len(missing)} 個路徑不存在，請確認容器內掛載路徑（預期 /workspace/Path_TIL）。"
    )

if args.image_check:
    print(f"▶  cv2 讀取檢查 {num_total} 張 ...")
    broken = []
    for i, fp in enumerate(image_paths):
        if (i + 1) % 500 == 0 or (i + 1) == num_total:
            print(
                f"    {i+1}/{num_total} ({(i+1)/num_total*100:.1f}%)",
                end="\r",
                flush=True,
            )
        if cv2.imread(fp) is None:
            broken.append(fp)
    print()
    if broken:
        for f in broken[:10]:
            print(f"    - {f}")
        raise IOError(f"{len(broken)} 張影像無法讀取。")
    print("  ✓ 影像可讀")
else:
    print("▶  已略過逐張 cv2 檢查（預設；需要時請加 --image-check）")


def apply_hne_norm_tilscout(rgb_u8):
    """
    與 TILscout process_tile 相同：僅對「mean < 230 且 std > 15」的 RGB patch 做 norm_HnE。
    只需 numpy patch，不需原始 slide。失敗或維度異常時回傳原圖。
    """
    if rgb_u8 is None or rgb_u8.size == 0 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        return rgb_u8
    if float(rgb_u8.mean()) >= 230.0 or float(rgb_u8.std()) <= 15.0:
        return rgb_u8
    try:
        inorm, _, _ = norm_HnE(rgb_u8, Io=240, alpha=1, beta=0.15)
        return inorm
    except Exception:
        return rgb_u8


def load_images(paths, use_hne_norm=True):
    imgs = []
    for fp in paths:
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"無法讀取: {fp}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if use_hne_norm:
            img = apply_hne_norm_tilscout(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        imgs.append(img)
    return np.asarray(imgs, dtype=np.uint8)


# ========= 6. 載入全部影像並切分 =========
print("\n▶  載入全部影像至記憶體 ...")
X_all = load_images(df["image_path"].tolist(), use_hne_norm=not args.no_hne_norm)
y_all = np.array([LABEL_TO_IDX[l] for l in df["label"].values], dtype=np.int32)
print(f"  ➜ {X_all.shape}, labels {y_all.shape}")
_vc_all = pd.Series(y_all).value_counts().sort_index()
print(
    "  ➜ 全資料各類別筆數 (0=positive,1=negative,2=other):\n"
    + _vc_all.to_string()
)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_all,
    y_all,
    test_size=args.test_ratio,
    random_state=SEED,
    stratify=y_all,
)
del X_all
gc.collect()
print(f"  ➜ trainval={len(X_trainval)}, test={len(X_test)}（train/test 已 stratify 類別）")
_vc_tv = pd.Series(y_trainval).value_counts().sort_index()
_vc_te = pd.Series(y_test).value_counts().sort_index()
print("  ➜ trainval 各類別:\n" + _vc_tv.to_string())
print("  ➜ test 各類別:\n" + _vc_te.to_string())

# heavy：imgaug 需要 uint8；light/medium：直接保留 TILscout 用 float [0,1]
X_trainval_uint8 = X_trainval
X_test_float = to_tilscout_input(X_test)
if args.aug == "heavy":
    X_trainval = X_trainval_uint8
    X_test = X_test_float
else:
    X_trainval = to_tilscout_input(X_trainval_uint8)
    X_test = X_test_float
    del X_trainval_uint8
gc.collect()
print(
    "  ➜ 輸入已對齊 TILscout：float32 RGB ∈ [0,1]（heavy 之 trainval 仍存 uint8 供 imgaug）"
)


def build_train_time_augmentation():
    """
    僅用於訓練資料管線，不併入 Sequential 分類模型（與原始架構分離）。
    light：對齊 flow_from_directory 前使用之 ImageDataGenerator(
        rescale=1./255 → 本腳本改以 float [0,1] 輸入；horizontal_flip；vertical_flip)。
    """
    if args.aug == "none":
        return None
    if args.aug == "light":
        return keras.Sequential(
            [
                RandomFlip("horizontal"),
                RandomFlip("vertical"),
            ],
            name="train_only_aug",
        )
    if args.aug == "medium":
        return keras.Sequential(
            [
                RandomFlip("horizontal"),
                RandomFlip("vertical"),
                RandomRotation(0.15),
                RandomZoom(0.12),
                RandomTranslation(0.12, 0.12),
                RandomContrast(0.12),
            ],
            name="train_only_aug",
        )
    return None


def make_tf_train_dataset(X_f, y, aug_model, batch_size, shuffle_seed):
    """X_f: float32 [0,1] RGB，與 TILscout 一致。"""
    ds = tf.data.Dataset.from_tensor_slices((X_f, y))
    n = len(X_f)
    if n > 0:
        ds = ds.shuffle(min(n, max(1000, n)), seed=shuffle_seed, reshuffle_each_iteration=True)
    if aug_model is not None:

        def _aug(x, yy):
            return aug_model(x, training=True), yy

        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_tf_val_dataset(X_f, y, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((X_f, y))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(pretrained_path):
    """
    與 inference_cptac_luad / TILscout 相同：整份 Sequential 以 load_model 載入。

    先前版本曾「新建」InceptionResNetV2 再依 **頂層** layer.name 從 .h5 複製權重；
    應用程式內多數 Conv/BN 在巢狀子層，名稱對不到時會被靜默略過，導致 backbone 多為隨機初始化、
    基線 Acc≈多數類比例，與直接 load_model 推論之高準確率矛盾。
    """
    print("  ➜ 載入預訓練權重（整份 Sequential，與推論腳本一致）...")
    model = keras.models.load_model(pretrained_path, compile=False)
    ly = model.layers
    if len(ly) < 4:
        raise ValueError(
            f"預期 Sequential 為 4 層：InceptionResNetV2, Flatten, Dense(512), Dense(3)；"
            f"實際 {len(ly)} 層。請確認 {pretrained_path}。"
        )
    gc.collect()
    print(
        f"  ✓ Sequential 就緒（IRV2→Flatten→Dense512→Dense3）| 輸入=[0,1] float RGB | "
        f"訓練 aug={args.aug}（僅資料管線）| 類別: {CLASS_NAMES_DISPLAY}"
    )
    return model


def save_learning_curve(history, path_png, title):
    """將 Keras fit 回傳的 History 存成 loss／AUC 雙子圖 PNG（主指標為 AUC）。"""
    if history is None:
        return
    hist = getattr(history, "history", None)
    if not hist or not hist.get("loss"):
        return
    n = len(hist["loss"])
    ep_t = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)

    axes[0].plot(ep_t, hist["loss"], label="train", color="C0")
    vl = hist.get("val_loss")
    if vl:
        ep_v = np.arange(1, len(vl) + 1)
        axes[0].plot(ep_v, vl, label="val", color="C1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (sparse_categorical_crossentropy)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    if hist.get("auc") is not None:
        axes[1].plot(ep_t, hist["auc"], label="train", color="C0")
        va = hist.get("val_auc")
        if va:
            ep_v = np.arange(1, len(va) + 1)
            axes[1].plot(ep_v, va, label="val", color="C1")
        axes[1].set_ylabel("AUC (Keras multi-label)")
    else:
        ak = "accuracy" if "accuracy" in hist else ("acc" if "acc" in hist else None)
        vak = ("val_" + ak) if ak and ("val_" + ak) in hist else None
        if ak:
            axes[1].plot(ep_t, hist[ak], label="train", color="C0")
            if vak and hist.get(vak):
                vv = hist[vak]
                ep_v = np.arange(1, len(vv) + 1)
                axes[1].plot(ep_v, vv, label="val", color="C1")
        axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 學習曲線: {path_png}")


def evaluate_model(model, X, y, split_name=""):
    if len(X) == 0:
        return {}
    if X.dtype == np.uint8:
        X = to_tilscout_input(X)
    _ev = model.evaluate(
        X, y, verbose=0, batch_size=args.batch_size, return_dict=True
    )
    loss = float(_ev["loss"])
    acc = float(_ev.get("accuracy", _ev.get("sparse_categorical_accuracy", 0.0)))
    y_prob = model.predict(X, verbose=0, batch_size=args.batch_size)
    y_pred = np.argmax(y_prob, axis=1)
    metrics = {
        "split": split_name,
        "loss": loss,
        "accuracy": acc,
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y, y_pred),
    }
    try:
        y_bin = label_binarize(y, classes=list(range(NUM_CLASS)))
        metrics["auc"] = roc_auc_score(
            y_bin, y_prob, average="weighted", multi_class="ovr"
        )
    except Exception:
        metrics["auc"] = float("nan")
    return metrics


# ========= 7. K-Fold =========
_run_title = (
    f"{args.folds}-Fold（僅 Stage 2 微調）"
    if args.only_finetune
    else (
        f"{args.folds}-Fold（僅 Stage 1）"
        if args.no_finetune
        else f"{args.folds}-Fold 2-stage 訓練"
    )
)
print(f"\n{'='*70}\n▶  {_run_title}\n{'='*70}")

if args.aug == "heavy":
    try:
        import imgaug  # noqa: F401
        from skimage.color import rgb2hed  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "aug=heavy 需要 imgaug 與 scikit-image：pip install imgaug scikit-image"
        ) from e

# StratifiedKFold：各 fold 驗證集類別比例與 trainval 相近（sklearn 要求每類在 trainval 至少 n_splits 筆）
_class_counts = np.bincount(y_trainval, minlength=NUM_CLASS)
if np.any(_class_counts < args.folds):
    raise ValueError(
        f"Stratified {args.folds}-fold 要求 trainval 內每個類別至少 {args.folds} 筆；"
        f"目前各類計數 {_class_counts.tolist()}（順序: positive, negative, other）。"
        f"請減少 --folds 或增加少數類樣本。"
    )

kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
all_scores = []

for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_trainval, y_trainval), start=1):
    print(f"\n{'-'*60}\n▶  Fold {fold}/{args.folds}")
    X_tr, y_tr = X_trainval[tr_idx], y_trainval[tr_idx]
    X_va, y_va = X_trainval[val_idx], y_trainval[val_idx]
    print(f"  train={len(X_tr)}, val={len(X_va)}")

    model = build_model(args.pretrained)
    backbone = model.layers[0]

    print("\n  ── 預訓練權重基線 stage=0（尚未本 fold 訓練；AUC=sklearn weighted OVR）──")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=train_metrics_list(),
    )
    for _bn, _Xe, _ye in [
        ("train", X_tr, y_tr),
        ("val", X_va, y_va),
        ("test", X_test, y_test),
    ]:
        _bm = evaluate_model(model, _Xe, _ye, _bn)
        _bm.update({"fold": fold, "stage": 0})
        all_scores.append(_bm)
        print(
            f"    {_bn:5s} AUC={_bm.get('auc', float('nan')):.4f}  "
            f"Acc={_bm['accuracy']:.4f}  F1={_bm['f1']:.4f}"
        )

    def get_callbacks(stage_name):
        ckpt = os.path.join(args.output_dir, f"fold{fold:02d}_{stage_name}_best.h5")
        return [
            keras.callbacks.ModelCheckpoint(
                ckpt,
                save_best_only=True,
                monitor="val_auc",
                mode="max",
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                patience=10,
                monitor="val_auc",
                mode="max",
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.2,
                patience=4,
                min_lr=1e-6,
                monitor="val_auc",
                mode="max",
                verbose=1,
            ),
        ], ckpt

    use_seq = args.aug == "heavy"
    aug_pipe = build_train_time_augmentation() if not use_seq else None

    def fit_stages():
        if not args.only_finetune:
            # -------- Stage 1 --------
            print("\n  ── Stage 1: 凍結 backbone ──")
            backbone.trainable = False
            model.compile(
                optimizer=keras.optimizers.Adam(1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=train_metrics_list(),
            )
            cbs1, ckpt1 = get_callbacks("stage1")
            if use_seq:
                tr = ImgAugTrainSequence(
                    X_tr,
                    y_tr,
                    args.batch_size,
                    shuffle=True,
                    augment=True,
                )
                va = ImgAugTrainSequence(
                    X_va, y_va, args.batch_size, shuffle=False, augment=False
                )
                h1 = model.fit(
                    tr,
                    validation_data=va,
                    epochs=args.epochs_stage1,
                    callbacks=cbs1,
                    verbose=1,
                )
            else:
                ds_tr = make_tf_train_dataset(
                    X_tr, y_tr, aug_pipe, args.batch_size, SEED + fold
                )
                ds_va = make_tf_val_dataset(X_va, y_va, args.batch_size)
                h1 = model.fit(
                    ds_tr,
                    validation_data=ds_va,
                    epochs=args.epochs_stage1,
                    callbacks=cbs1,
                    verbose=1,
                )
            save_learning_curve(
                h1,
                os.path.join(
                    args.output_dir, f"fold{fold:02d}_stage1_learning_curve.png"
                ),
                f"Fold {fold}/{args.folds} — Stage 1 (frozen backbone)",
            )

            best_s1 = keras.models.load_model(ckpt1, compile=False)
            best_s1.compile(
                optimizer=keras.optimizers.Adam(1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=train_metrics_list(),
            )
            print("\n  ── Stage 1 評估 ──")
            for name, Xe, ye in [
                ("train", X_tr, y_tr),
                ("val", X_va, y_va),
                ("test", X_test, y_test),
            ]:
                m = evaluate_model(best_s1, Xe, ye, name)
                m.update({"fold": fold, "stage": 1})
                all_scores.append(m)
                print(
                    f"    {name:5s} AUC={m.get('auc', float('nan')):.4f}  "
                    f"Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}"
                )
            del best_s1
            gc.collect()

            if args.no_finetune:
                return None

        # -------- Stage 2 --------
        if args.only_finetune:
            print("\n  ── 僅 Stage 2：跳過 Stage 1，從預訓練權重微調 ──")
        else:
            print("\n  ── Stage 2: 微調 backbone ──")
        backbone.trainable = True
        model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=train_metrics_list(),
        )
        cbs2, ckpt2 = get_callbacks("stage2")
        if use_seq:
            tr = ImgAugTrainSequence(
                X_tr,
                y_tr,
                args.batch_size,
                shuffle=True,
                augment=True,
            )
            va = ImgAugTrainSequence(
                X_va, y_va, args.batch_size, shuffle=False, augment=False
            )
            h2 = model.fit(
                tr,
                validation_data=va,
                epochs=args.epochs_stage2,
                callbacks=cbs2,
                verbose=1,
            )
        else:
            ds_tr = make_tf_train_dataset(
                X_tr, y_tr, aug_pipe, args.batch_size, SEED + fold + 1000
            )
            ds_va = make_tf_val_dataset(X_va, y_va, args.batch_size)
            h2 = model.fit(
                ds_tr,
                validation_data=ds_va,
                epochs=args.epochs_stage2,
                callbacks=cbs2,
                verbose=1,
            )
        save_learning_curve(
            h2,
            os.path.join(args.output_dir, f"fold{fold:02d}_stage2_learning_curve.png"),
            f"Fold {fold}/{args.folds} — Stage 2 (fine-tune)",
        )
        return ckpt2

    ckpt2 = fit_stages()

    if ckpt2 is not None:
        best_s2 = keras.models.load_model(ckpt2, compile=False)
        best_s2.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=train_metrics_list(),
        )
        print("\n  ── Stage 2 評估 ──")
        for name, Xe, ye in [
            ("train", X_tr, y_tr),
            ("val", X_va, y_va),
            ("test", X_test, y_test),
        ]:
            m = evaluate_model(best_s2, Xe, ye, name)
            m.update({"fold": fold, "stage": 2})
            all_scores.append(m)
            print(
                f"    {name:5s} AUC={m.get('auc', float('nan')):.4f}  "
                f"Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}"
            )
        del best_s2
        gc.collect()

    del model
    gc.collect()
    keras.backend.clear_session()

# ========= 8. 輸出 CSV =========
print(f"\n{'='*70}\n▶  寫入 qupath_2stage_scores.csv ...")
scores_df = pd.DataFrame(all_scores)
col_order = [
    "fold",
    "stage",
    "split",
    "auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "kappa",
    "loss",
]
scores_df = scores_df[[c for c in col_order if c in scores_df.columns]]
out_csv = os.path.join(args.output_dir, "qupath_2stage_scores.csv")
scores_df.to_csv(out_csv, index=False, float_format="%.6f")
print(f"  ✓ {out_csv}")

# CV 整合：各 stage × split 跨 fold 的 mean / std（std 為樣本標準差 ddof=1）
_metric_cols = [
    c
    for c in (
        "auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "kappa",
        "loss",
    )
    if c in scores_df.columns
]
_summary_rows = []
for _stage in sorted(scores_df["stage"].unique()):
    for _m in _metric_cols:
        for _split in ["train", "val", "test"]:
            _sub = scores_df[
                (scores_df["stage"] == _stage) & (scores_df["split"] == _split)
            ]
            if _sub.empty:
                continue
            _vals = _sub[_m].dropna().to_numpy(dtype=np.float64)
            if _vals.size == 0:
                continue
            _mean = float(np.mean(_vals))
            _std = float(np.std(_vals, ddof=1)) if _vals.size > 1 else 0.0
            _summary_rows.append(
                {
                    "stage": int(_stage),
                    "metric": _m,
                    "split": _split,
                    "mean": _mean,
                    "std": _std,
                    "n_folds": int(_vals.size),
                }
            )
summary_df = pd.DataFrame(_summary_rows)
if not summary_df.empty:
    # 列順序：stage → metric（依 _metric_cols）→ split 固定為 train → val → test（整數鍵排序）
    _metric_rank = {m: i for i, m in enumerate(_metric_cols)}
    _split_rank = {"train": 0, "val": 1, "test": 2}
    summary_df = (
        summary_df.assign(
            __mr=summary_df["metric"].map(_metric_rank),
            __sr=summary_df["split"].map(_split_rank),
        )
        .sort_values(by=["stage", "__mr", "__sr"], kind="stable")
        .drop(columns=["__mr", "__sr"])
        .reset_index(drop=True)
    )
    _sum_cols = ["stage", "metric", "split", "mean", "std", "n_folds"]
    summary_df = summary_df[[c for c in _sum_cols if c in summary_df.columns]]
summary_csv = os.path.join(args.output_dir, "qupath_2stage_cv_summary.csv")
if not summary_df.empty:
    summary_df.to_csv(summary_csv, index=False, float_format="%.6f")
    print(f"  ✓ CV 整合（mean±std）: {summary_csv}")
else:
    print("  ⚠ 無法產生 CV 整合 CSV（scores 為空）")

print(f"\n{'='*70}\n▶  彙總（依 stage / split）\n{'='*70}")
for stage in sorted(scores_df["stage"].unique()):
    print(f"\n  === Stage {int(stage)} ===")
    for split in ["train", "val", "test"]:
        sub = scores_df[(scores_df["stage"] == stage) & (scores_df["split"] == split)]
        if sub.empty:
            continue
        print(f"\n  [{split.upper()}]")
        for metric in ["auc", "accuracy", "precision", "recall", "f1", "kappa"]:
            if metric in sub.columns:
                v = sub[metric].dropna().values
                if len(v):
                    print(f"    {metric:12s}: {np.mean(v):.4f} ± {np.std(v):.4f}")

test_scores = scores_df[scores_df["split"] == "test"]
if not test_scores.empty:
    print(f"\n{'='*70}\n▶  測試集最佳 fold（依 AUC）\n{'='*70}")
    for stage in sorted(test_scores["stage"].unique()):
        st = test_scores[test_scores["stage"] == stage]
        j = st["auc"].idxmax()
        row = st.loc[j]
        print(
            f"\n  Stage {int(stage)} best fold {int(row['fold'])}: "
            f"AUC={row['auc']:.4f}  Acc={row['accuracy']:.4f}"
        )

print("\n✓ 完成")
_stages_mode = (
    "only-finetune"
    if args.only_finetune
    else ("no-finetune" if args.no_finetune else "stage1+stage2")
)
print(f"  設定摘要: stages={_stages_mode}, aug={args.aug}, pretrained={args.pretrained}")
