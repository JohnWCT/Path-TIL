# -*- coding: utf-8 -*-
"""
InceptionResNetV2
10-fold 交叉驗證 + 指定資料夾測試
"""

# ========= 0. 共用匯入 =========
import numpy as np
import glob, os, cv2, tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, cohen_kappa_score
)

# ========= 共用常數 =========
IMG_SIZE  = 224
IMG_CH    = 3
NUM_CLASS = 3             # ← 請依資料集調整
TRAIN_DIR = "Train_new2/train"
TEST_DIR  = "Train_new2/test"   # ← 指定測試資料夾
SEED      = 42
FOLDS     = 10

# --- 自訂 H&E stain 相關工具 ---
from skimage.color import rgb2hed, hed2rgb

def rgb2hed_func(images, random_state, parents, hooks):
    return [rgb2hed(img).astype('float32') for img in images]

def hed2rgb_func(images, random_state, parents, hooks):
    return [(hed2rgb(img) * 255).astype('uint8') for img in images]

# --- imgaug 增強管線 ---
import imgaug.augmenters as iaa
seq = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.SomeOf(2, [
        iaa.Fliplr(1), iaa.Flipud(1), iaa.Rot90((1, 3))
    ])),
    iaa.Sometimes(0.5, iaa.Sequential([
        iaa.Lambda(rgb2hed_func),
        iaa.WithChannels(0, iaa.Multiply((0.45, 1.90))),  # Hematoxylin
        iaa.WithChannels(1, iaa.Multiply((0.5, 2.45))),   # Eosin
        iaa.Lambda(hed2rgb_func)
    ])),
    iaa.Sometimes(0.5, iaa.OneOf([
        iaa.SaltAndPepper((0.0, 0.2)), iaa.Salt((0.0, 0.2)),
        iaa.Pepper((0.0, 0.12)), iaa.GaussianBlur(sigma=(0.0, 1.9)),
        iaa.AdditiveGaussianNoise(scale=(0.0, 0.14*255))
    ])),
    iaa.Sometimes(0, iaa.OneOf([
        iaa.Add((-50, 50)), iaa.Multiply((0.90, 1.34)),
        iaa.AddToHue((-16, 8)), iaa.AddToSaturation((-40, 35)),
        iaa.MultiplyHue((0.88, 1.12)), iaa.MultiplySaturation((0.75, 1.35)),
        iaa.LinearContrast((0.80, 1.70)), iaa.GammaContrast((0.80, 2.00)),
        iaa.CLAHE(clip_limit=(0.10, 4.90)),
        iaa.JpegCompression(compression=(0.0, 90))
    ])),
    iaa.Sometimes(0.5,
        iaa.Cutout(nb_iterations=(1, 5), size=0.2,
                   fill_mode="constant", cval=255)),
], random_order=True)




# ------------------------------------------------------------
# ❶ 訓練集載入（含標籤）與編碼
# ------------------------------------------------------------
class ImgAugSequence(keras.utils.Sequence):
    """
    將 numpy 影像陣列 + 標籤轉成 batch，並以 imgaug 做即時增強。
    """
    def __init__(self, x, y, batch_size=32, shuffle=True, augment=True):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_ids = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x = self.x[batch_ids]
        batch_y = self.y[batch_ids]

        if self.augment:
            # imgaug 期望 uint8；若傳 tensor 需轉 numpy
            batch_x = seq(images=batch_x)              # <── 套用增強
            batch_x = np.array(batch_x, dtype=np.uint8)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

print("▶  載入訓練資料 ...")
BATCH = 32
train_gen = ImgAugSequence(X_train[tr_idx], y_train[tr_idx],
                           batch_size=BATCH, augment=True)
val_gen   = ImgAugSequence(X_train[val_idx], y_train[val_idx],
                           batch_size=BATCH, augment=False)

# ------------------------------------------------------------
# ❷ 10-fold 交叉驗證 (每個 fold 另存最佳模型)
# ------------------------------------------------------------
acc_fold, prec_fold, rec_fold, f1_fold, auc_fold, kappa_fold, loss_fold = ([] for _ in range(7))
kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

def build_model():
    aug = keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1)
    ], name="data_aug")

    inp = Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CH), dtype=tf.uint8)
    x = tf.cast(inp, tf.float32)
    x = aug(x)
    x = preprocess_input(x)

    backbone = keras.applications.InceptionResNetV2(
        include_top=False, input_tensor=x, pooling="avg", weights="imagenet"
    )
    x = Dense(512, activation='relu')(backbone.output)
    out = Dense(NUM_CLASS, activation='softmax')(x)
    model = keras.Model(inp, out)
    return model, backbone

for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_train, y_train), start=1):
    print(f"\n{'-'*60}\n▶  Fold {fold}/{FOLDS}")
    model, backbone = build_model()

    ckpt_path = f"fold_{fold:02d}_best.h5"
    cbs = [
        keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True,
                                        monitor="val_accuracy", verbose=1),
        keras.callbacks.EarlyStopping(patience=10, monitor="val_loss", restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=4, min_lr=1e-6, verbose=1)
    ]

    # -- 階段一: 凍結 backbone
    backbone.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=50, callbacks=cbs, verbose=1)

    # -- 階段二: 微調
    backbone.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=50, callbacks=cbs, verbose=1)

    # -- 評估 (以儲存之最佳權重)
    best = keras.models.load_model(ckpt_path)
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    val_loss, val_acc = best.evaluate(X_val, y_val, verbose=0)
    y_prob = best.predict(X_val, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # 指標累積
    loss_fold.append(val_loss)
    acc_fold.append(val_acc)
    prec_fold.append(precision_score(y_val, y_pred, average='weighted'))
    rec_fold.append(recall_score(y_val, y_pred, average='weighted'))
    f1_fold.append(f1_score(y_val, y_pred, average='weighted'))
    kappa_fold.append(cohen_kappa_score(y_val, y_pred))
    auc_fold.append(roc_auc_score(
        label_binarize(y_val, classes=range(NUM_CLASS)),
        y_prob, average='weighted', multi_class='ovr')
    )

# ---- 交叉驗證結果 ----
def show_metric(name, vals):
    print(f"{name:<20}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

print("\n======== K-fold 平均結果 ========")
for n, v in [("Acc", acc_fold), ("Prec", prec_fold), ("Recall", rec_fold),
             ("F1", f1_fold), ("AUC", auc_fold), ("Kappa", kappa_fold), ("Loss", loss_fold)]:
    show_metric(n, v)

# ------------------------------------------------------------
# ❸ 測試集載入與推論 (使用 10 個 fold 模型平均投票)
# ------------------------------------------------------------
print("\n▶  載入測試資料 ...")
X_test, y_test_str = load_images_from_folder(TEST_DIR)
y_test = le.transform(y_test_str)          # 同訓練標籤編碼
print(f"  ➜ 測試集 {len(X_test)} 張")

# -- 收集各 fold 模型預測 (機率平均 / 多數票 皆可)
probs_all = np.zeros((len(X_test), NUM_CLASS))

for fold in range(1, FOLDS + 1):
    model = keras.models.load_model(f"fold_{fold:02d}_best.h5")
    probs_all += model.predict(X_test, verbose=0)

probs_all /= FOLDS
y_pred = np.argmax(probs_all, axis=1)

# ---- 測試集指標 ----
print("\n======== Test Set Results ========")
print(f"Accuracy            : {accuracy_score(y_test, y_pred):.4f}")
print(f"Weighted Precision  : {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Weighted Recall     : {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Weighted F1-score   : {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Cohen Kappa         : {cohen_kappa_score(y_test, y_pred):.4f}")

y_test_bin = label_binarize(y_test, classes=range(NUM_CLASS))
print(f"Weighted AUC        : {roc_auc_score(y_test_bin, probs_all, average='weighted', multi_class='ovr'):.4f}")
