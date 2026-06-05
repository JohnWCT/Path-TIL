# -*- coding: utf-8 -*-
"""
inference_cptac_luad.py
=======================
對 cptac_luad_dataset.csv（或相同欄位之 CSV）做批次推論，**預處理與 TILscout.py 推論迴圈一致**，
用於確認 `best_InceptionResNetV2_model.h5` 與 TILscout 之 `model.predict` 行為相同（非訓練腳本之差異）。

TILscout 對應程式（無 norm_HnE，僅幾何與縮放）::

    pred_images = [
        cv2.cvtColor(
            cv2.resize(cv2.imread(p, cv2.IMREAD_COLOR), (224, 224)),
            cv2.COLOR_BGR2RGB,
        )
        for p in batch_img_paths
    ]
    X_pred = np.array(pred_images) / 255.0
    y_proba = prediction_model.predict(X_pred)

標籤：CSV 為小寫 positive / negative / other，對應模型輸出維度 0,1,2
（與 TILscout 之 DataFrame 欄位 Positive, Negative, Other 順序一致）。

報表：除 classification_report 外，會輸出 **各類別 one-vs-rest AUC**（sklearn 二元 ROC AUC），
並附 macro 平均（略過 nan）。

若您先前在「已 norm_HnE 之 patch」上得到約 Accuracy≈0.96、AUC≈0.99，請**勿**加 --hne-norm。
若 CPTAC 圖為未染色正規化之 raw patch，可嘗試 --hne-norm（與 TILscout **產 patch** 段之條件式 norm 對齊）。

使用範例::

    python3 inference_cptac_luad.py
    python3 inference_cptac_luad.py --csv /workspace/Path_TIL/cptac_luad_dataset.csv \\
        --model /workspace/Path_TIL/best_InceptionResNetV2_model.h5 --batch-size 256
    python3 inference_cptac_luad.py --hne-norm
"""

from __future__ import annotations

import argparse
import gc
import logging
import os

# 須在 import tensorflow 之前，否則無法抑制 oneDNN / cpu_feature_guard 等 C++ INFO
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from tensorflow import keras

# 與 InceptionResNetV2_QuPath_2stage / TILscout 欄位順序一致
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
try:
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

IMG_SIZE = 224
LABEL_TO_IDX = {"positive": 0, "negative": 1, "other": 2}
IDX_TO_TILSCOUT_COL = {0: "Positive", 1: "Negative", 2: "Other"}
CLASS_NAMES_REPORT = ["Positive", "Negative", "Other"]
CLASS_INDICES = [0, 1, 2]


def per_class_auc_ovr(
    y_true: np.ndarray, y_prob: np.ndarray
) -> pd.DataFrame:
    """
    各類別 one-vs-rest AUC：以該類為正、其餘為負，使用該類對應之預測機率欄。
    與整體 weighted OVR 同為 sklearn roc_auc_score 之二元 AUC 組合邏輯。
    若該類在真實標籤中未出現或僅單一類別，則 AUC 為 nan。
    """
    rows = []
    for cls_idx, name in zip(CLASS_INDICES, CLASS_NAMES_REPORT):
        y_bin = (y_true == cls_idx).astype(np.int32)
        n_pos = int(y_bin.sum())
        n_neg = int(len(y_bin) - n_pos)
        if n_pos == 0 or n_neg == 0:
            auc_val = np.nan
        else:
            try:
                auc_val = roc_auc_score(y_bin, y_prob[:, cls_idx])
            except ValueError:
                auc_val = np.nan
        rows.append(
            {
                "class": name,
                "auc_ovr": auc_val,
                "n_true_class": n_pos,
                "n_other": n_neg,
            }
        )
    return pd.DataFrame(rows)


def _normalize_label(s) -> str | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().lower()
    return t if t in LABEL_TO_IDX else None


def _apply_hne_norm_tilscout(rgb_u8: np.ndarray) -> np.ndarray:
    """與 TILscout process_tile + normalize_HnE 相同條件（選用）。"""
    from normalize_HnE import norm_HnE

    if rgb_u8 is None or rgb_u8.size == 0 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        return rgb_u8
    if float(rgb_u8.mean()) >= 230.0 or float(rgb_u8.std()) <= 15.0:
        return rgb_u8
    try:
        inorm, _, _ = norm_HnE(rgb_u8, Io=240, alpha=1, beta=0.15)
        return inorm
    except Exception:
        return rgb_u8


def load_batch_tilscout_style(
    paths: list[str], use_hne_norm: bool
) -> np.ndarray:
    """
    回傳 float32 (N,224,224,3)，數值 [0,1]。

    預設（與 TILscout 逐字一致）::
        imread BGR → resize(BGR, 224) → cvtColor(BGR2RGB) → /255

    --hne-norm：全解析度 RGB → 條件式 norm_HnE → resize(RGB, 224) → /255
    （對齊「產 patch 時先 norm 再存檔」之語意；非 TILscout 推論迴圈原始寫法）
    """
    rgb_list = []
    for p in paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"無法讀取: {p}")
        if use_hne_norm:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = _apply_hne_norm_tilscout(rgb)
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        else:
            resized_bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
            rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        rgb_list.append(rgb)
    return np.asarray(rgb_list, dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPTAC CSV 推論（預處理對齊 TILscout predict）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default="/workspace/Path_TIL/cptac_luad_dataset.csv",
        help="含 case_id, image_path, label 之 CSV",
    )
    parser.add_argument(
        "--model",
        default="/workspace/Path_TIL/best_InceptionResNetV2_model.h5",
        help="TILscout 相同之 Sequential .h5",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="predict batch size")
    parser.add_argument(
        "--hne-norm",
        action="store_true",
        help="讀檔後、resize 前做條件式 norm_HnE（預設關閉＝與 TILscout 推論段一致）",
    )
    parser.add_argument(
        "--save-preds",
        type=str,
        default="",
        help="若指定路徑，寫出每筆路徑、真實標籤、預測與三類機率之 CSV",
    )
    args = parser.parse_args()

    print("▶  載入 CSV:", args.csv)
    df = pd.read_csv(args.csv)
    need = {"image_path", "label"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV 需含欄位 {need}，目前: {df.columns.tolist()}")

    df = df.copy()
    df["label_norm"] = df["label"].map(_normalize_label)
    bad = df["label_norm"].isna()
    if bad.any():
        ex = df.loc[bad, "label"].unique()[:10]
        raise ValueError(f"無法解析之 label: {list(ex)}")
    df["y"] = df["label_norm"].map(LABEL_TO_IDX)

    paths = df["image_path"].tolist()
    y_true = df["y"].to_numpy(dtype=np.int32)
    n = len(paths)
    print(f"  ➜ {n} 筆")
    print(f"  ➜ 類別分佈:\n{df['label_norm'].value_counts().to_string()}")

    if args.hne_norm:
        print("▶  預處理：resize 前條件式 norm_HnE（**與 TILscout 推論段不同**，僅在需對齊「產 patch」流程時使用）")
    else:
        print(
            "▶  預處理：**與 TILscout.py 推論相同** — imread → resize(224) → BGR2RGB → /255，無 norm_HnE"
        )

    print("▶  載入模型:", args.model)
    model = keras.models.load_model(args.model, compile=False)

    probs_all = np.zeros((n, 3), dtype=np.float32)
    bs = max(1, int(args.batch_size))
    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch_p = paths[start:end]
        xb = load_batch_tilscout_style(batch_p, use_hne_norm=args.hne_norm)
        probs_all[start:end] = model.predict(xb, verbose=0)
        del xb
        gc.collect()

    y_pred = np.argmax(probs_all, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    try:
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        auc = roc_auc_score(
            y_bin, probs_all, average="weighted", multi_class="ovr"
        )
    except Exception:
        auc = float("nan")

    print(f"\n{'='*60}")
    print("▶  整體指標（與訓練腳本 evaluate 相同定義）")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.6f}")
    print(f"  Precision (weighted): {prec:.6f}")
    print(f"  Recall   (weighted): {rec:.6f}")
    print(f"  F1       (weighted): {f1:.6f}")
    print(f"  AUC      (weighted OVR): {auc:.6f}")
    print(f"  Kappa    : {kappa:.6f}")
    print(
        "\n  （若您參考值約 Accuracy=0.9613, AUC=0.9917，請比對同一批圖檔、同一 .h5、"
        "且預處理須與當時評估一致；raw patch 與已 norm patch 結果會不同。）"
    )

    print(f"\n▶  Confusion matrix [rows=true, cols=pred] 順序 {CLASS_NAMES_REPORT}")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))

    print(f"\n▶  Classification report")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES_REPORT,
            digits=4,
            zero_division=0,
        )
    )

    df_auc_cls = per_class_auc_ovr(y_true, probs_all)
    print(f"\n▶  Per-class AUC (one-vs-rest, 與整體 weighted OVR 同源之二元 AUC)")
    _disp = df_auc_cls.copy()
    _disp["auc_ovr"] = _disp["auc_ovr"].map(
        lambda x: f"{x:.6f}" if pd.notna(x) else "nan"
    )
    print(_disp.to_string(index=False))
    _macro = np.nanmean(df_auc_cls["auc_ovr"].to_numpy(dtype=np.float64))
    if not np.isnan(_macro):
        print(f"\n  Macro AUC (各類 OVR 平均, 略過 nan): {_macro:.6f}")

    if args.save_preds:
        out = pd.DataFrame(
            {
                "image_path": paths,
                "y_true_label": [IDX_TO_TILSCOUT_COL[int(t)] for t in y_true],
                "y_pred_label": [IDX_TO_TILSCOUT_COL[int(p)] for p in y_pred],
                "prob_Positive": probs_all[:, 0],
                "prob_Negative": probs_all[:, 1],
                "prob_Other": probs_all[:, 2],
            }
        )
        if "case_id" in df.columns:
            out.insert(0, "case_id", df["case_id"].values)
        out.to_csv(args.save_preds, index=False, float_format="%.8f")
        print(f"\n✓ 預測明細: {args.save_preds}")


if __name__ == "__main__":
    main()
