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
import pandas as pd
import argparse

# ========= 命令列參數設定 =========
parser = argparse.ArgumentParser(description='InceptionResNetV2 Cross Validation Training')
parser.add_argument('--no-finetune', action='store_true', 
                    help='跳過階段二的微調，只執行階段一的凍結訓練')
parser.add_argument('--batch-size', type=int, default=16,
                    help='訓練的 batch size (預設: 16)')
args = parser.parse_args()

print(f"執行模式: {'僅凍結訓練' if args.no_finetune else '凍結訓練 + 微調'}")
print(f"Batch size: {args.batch_size}")

# ========= 共用常數 =========
IMG_SIZE  = 224
IMG_CH    = 3
NUM_CLASS = 3             # ← 請依資料集調整
TRAIN_DIR = "Train_new2/train"
TEST_DIR  = "Train_new2/test"   # ← 指定測試資料夾
SEED      = 42
FOLDS     = 10

# ------------------------------------------------------------
# ❶ 訓練集載入（含標籤）與編碼
# ------------------------------------------------------------
def load_images_from_folder(folder):
    imgs, labels = [], []
    for cls_dir in glob.glob(os.path.join(folder, "*")):
        label = os.path.basename(cls_dir)
        for img_fp in glob.glob(os.path.join(cls_dir, "*.tif")):
            img = cv2.imread(img_fp, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            imgs.append(img)
            labels.append(label)
    return np.array(imgs, dtype=np.uint8), np.array(labels)

print("▶  載入訓練資料 ...")
X_train, y_train_str = load_images_from_folder(TRAIN_DIR)
le = LabelEncoder()
y_train = le.fit_transform(y_train_str)
print(f"  ➜ 共 {len(X_train)} 張, 類別對應：{le.classes_.tolist()}")

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
    model.fit(X_train[tr_idx], y_train[tr_idx],
              validation_data=(X_train[val_idx], y_train[val_idx]),
              epochs=50, batch_size=args.batch_size, callbacks=cbs, verbose=1)

    # -- 階段二: 微調
    if not args.no_finetune:
        backbone.trainable = True
        model.compile(optimizer=keras.optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(X_train[tr_idx], y_train[tr_idx],
                  validation_data=(X_train[val_idx], y_train[val_idx]),
                  epochs=50, batch_size=args.batch_size, callbacks=cbs, verbose=1)

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

# ---- 詳細交叉驗證結果表格 ----
print("\n======== Detailed Cross-Validation Results ========")

# 創建交叉驗證結果表格
cv_data = []
for fold in range(FOLDS):
    cv_data.append({
        'Fold': f"Fold {fold+1:02d}",
        'Accuracy': f"{acc_fold[fold]:.4f}",
        'Precision': f"{prec_fold[fold]:.4f}",
        'Recall': f"{rec_fold[fold]:.4f}",
        'F1-Score': f"{f1_fold[fold]:.4f}",
        'AUC': f"{auc_fold[fold]:.4f}",
        'Kappa': f"{kappa_fold[fold]:.4f}",
        'Loss': f"{loss_fold[fold]:.4f}"
    })

# 添加平均行
cv_data.append({
    'Fold': 'Mean ± Std',
    'Accuracy': f"{np.mean(acc_fold):.4f} ± {np.std(acc_fold):.4f}",
    'Precision': f"{np.mean(prec_fold):.4f} ± {np.std(prec_fold):.4f}",
    'Recall': f"{np.mean(rec_fold):.4f} ± {np.std(rec_fold):.4f}",
    'F1-Score': f"{np.mean(f1_fold):.4f} ± {np.std(f1_fold):.4f}",
    'AUC': f"{np.mean(auc_fold):.4f} ± {np.std(auc_fold):.4f}",
    'Kappa': f"{np.mean(kappa_fold):.4f} ± {np.std(kappa_fold):.4f}",
    'Loss': f"{np.mean(loss_fold):.4f} ± {np.std(loss_fold):.4f}"
})

# 顯示交叉驗證表格
df_cv = pd.DataFrame(cv_data)
print(df_cv.to_string(index=False))

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

# ------------------------------------------------------------
# ❹ 詳細分類報告 (類似 classification_report 格式)
# ------------------------------------------------------------
print("\n======== Detailed Classification Report ========")

# 計算每個類別的詳細指標
from sklearn.metrics import classification_report

# 為每個fold收集詳細指標
fold_results = []
fold_aucs = []

print("\n======== Individual Fold Results ========")
for fold in range(1, FOLDS + 1):
    print(f"\n--- Fold {fold:02d} ---")
    
    # 載入該fold的模型
    model = keras.models.load_model(f"fold_{fold:02d}_best.h5")
    
    # 對測試集進行預測
    probs_fold = model.predict(X_test, verbose=0)
    y_pred_fold = np.argmax(probs_fold, axis=1)
    
    # 計算該fold的指標
    fold_acc = accuracy_score(y_test, y_pred_fold)
    fold_prec = precision_score(y_test, y_pred_fold, average='weighted')
    fold_rec = recall_score(y_test, y_pred_fold, average='weighted')
    fold_f1 = f1_score(y_test, y_pred_fold, average='weighted')
    fold_kappa = cohen_kappa_score(y_test, y_pred_fold)
    
    # 計算該fold的AUC
    fold_auc = roc_auc_score(y_test_bin, probs_fold, average='weighted', multi_class='ovr')
    fold_aucs.append(fold_auc)
    
    print(f"Accuracy: {fold_acc:.4f}")
    print(f"Precision: {fold_prec:.4f}")
    print(f"Recall: {fold_rec:.4f}")
    print(f"F1-Score: {fold_f1:.4f}")
    print(f"Kappa: {fold_kappa:.4f}")
    print(f"AUC: {fold_auc:.4f}")
    
    # 生成該fold的詳細分類報告
    report_fold = classification_report(y_test, y_pred_fold, target_names=le.classes_, 
                                       output_dict=True, zero_division=0)
    
    # 計算每個類別的AUC
    auc_per_class_fold = []
    for i in range(NUM_CLASS):
        auc = roc_auc_score((y_test == i).astype(int), probs_fold[:, i])
        auc_per_class_fold.append(auc)
    
    # 計算micro和macro平均AUC
    micro_auc_fold = roc_auc_score(y_test_bin, probs_fold, average='micro', multi_class='ovr')
    macro_auc_fold = roc_auc_score(y_test_bin, probs_fold, average='macro', multi_class='ovr')
    
    # 收集該fold的結果
    fold_result = {
        'fold': fold,
        'accuracy': fold_acc,
        'precision': fold_prec,
        'recall': fold_rec,
        'f1_score': fold_f1,
        'kappa': fold_kappa,
        'auc': fold_auc,
        'micro_auc': micro_auc_fold,
        'macro_auc': macro_auc_fold,
        'class_aucs': auc_per_class_fold,
        'class_metrics': report_fold
    }
    fold_results.append(fold_result)

# 找出最佳AUC的fold
best_fold_idx = np.argmax(fold_aucs)
best_fold = best_fold_idx + 1
print(f"\n======== Best Performing Fold ========")
print(f"Best Fold: {best_fold} (AUC: {fold_aucs[best_fold_idx]:.4f})")

# 計算所有fold的平均和標準差
print(f"\n======== K-fold Statistics ========")
metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'kappa', 'auc', 'micro_auc', 'macro_auc']
for metric in metrics_names:
    values = [fold_results[i][metric] for i in range(FOLDS)]
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{metric.capitalize():<12}: {mean_val:.4f} ± {std_val:.4f}")

# 計算每個類別的AUC統計
print(f"\n======== Class-wise AUC Statistics ========")
for i, class_name in enumerate(le.classes_):
    class_aucs = [fold_results[j]['class_aucs'][i] for j in range(FOLDS)]
    mean_auc = np.mean(class_aucs)
    std_auc = np.std(class_aucs)
    print(f"{class_name:<15}: {mean_auc:.4f} ± {std_auc:.4f}")

# 使用最佳fold的結果生成詳細分類報告
print(f"\n======== Detailed Classification Report (Best Fold {best_fold}) ========")
best_fold_result = fold_results[best_fold_idx]
best_report = best_fold_result['class_metrics']

# 創建結果表格
results_data = []
for i, class_name in enumerate(le.classes_):
    if class_name in best_report:
        results_data.append({
            'Class': class_name,
            'Precision': f"{best_report[class_name]['precision']:.4f}",
            'Recall': f"{best_report[class_name]['recall']:.4f}",
            'F1-Score': f"{best_report[class_name]['f1-score']:.4f}",
            'Support': best_report[class_name]['support'],
            'AUC': f"{best_fold_result['class_aucs'][i]:.4f}"
        })

# 添加總計行
results_data.append({
    'Class': 'Weighted Avg',
    'Precision': f"{best_report['weighted avg']['precision']:.4f}",
    'Recall': f"{best_report['weighted avg']['recall']:.4f}",
    'F1-Score': f"{best_report['weighted avg']['f1-score']:.4f}",
    'Support': f"{best_report['weighted avg']['support']:.0f}",
    'AUC': f"{best_fold_result['auc']:.4f}"
})

# 添加micro和macro平均
results_data.append({
    'Class': 'Micro Avg',
    'Precision': f"{best_report['micro avg']['precision']:.4f}",
    'Recall': f"{best_report['micro avg']['recall']:.4f}",
    'F1-Score': f"{best_report['micro avg']['f1-score']:.4f}",
    'Support': f"{best_report['micro avg']['support']:.0f}",
    'AUC': f"{best_fold_result['micro_auc']:.4f}"
})

results_data.append({
    'Class': 'Macro Avg',
    'Precision': f"{best_report['macro avg']['precision']:.4f}",
    'Recall': f"{best_report['macro avg']['recall']:.4f}",
    'F1-Score': f"{best_report['macro avg']['f1-score']:.4f}",
    'Support': f"{best_report['macro avg']['support']:.0f}",
    'AUC': f"{best_fold_result['macro_auc']:.4f}"
})

# 顯示表格
df_results = pd.DataFrame(results_data)
print(df_results.to_string(index=False))

# 顯示整體準確率
print(f"\nOverall Accuracy: {best_fold_result['accuracy']:.4f}")
print(f"Cohen's Kappa: {best_fold_result['kappa']:.4f}")

# 顯示所有fold的AUC比較
print(f"\n======== All Folds AUC Comparison ========")
for i, auc in enumerate(fold_aucs, 1):
    marker = " ★" if i == best_fold else ""
    print(f"Fold {i:02d}: {auc:.4f}{marker}")
