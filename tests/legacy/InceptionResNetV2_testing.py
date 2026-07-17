# -*- coding: utf-8 -*-
"""
InceptionResNetV2 Testing Script
獨立測試腳本 - 使用已訓練的K-fold模型對指定資料夾進行測試
"""

import argparse
import numpy as np
import glob, os, cv2, tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, cohen_kappa_score, classification_report
)
import pandas as pd
import warnings

# 抑制警告
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制TensorFlow警告
tf.get_logger().setLevel('ERROR')  # 抑制TensorFlow日誌

# ========= 常數設定 =========
IMG_SIZE = 224
IMG_CH = 3
NUM_CLASS = 3
FOLDS = 10
SEED = 42

def load_images_from_folder(folder):
    """載入指定資料夾中的圖片和標籤"""
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

def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='InceptionResNetV2 Testing Script')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='測試資料夾路徑 (包含類別子資料夾)')
    parser.add_argument('--model_dir', type=str, default='.',
                       help='模型檔案所在目錄 (預設為當前目錄)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='結果輸出檔案 (可選)')
    parser.add_argument('--csv_output', type=str, default=None,
                       help='CSV結果輸出檔案 (可選)')
    parser.add_argument('--csv_prefix', type=str, default='test_results',
                       help='CSV檔案前綴 (預設: test_results)')
    
    args = parser.parse_args()
    
    # 檢查測試資料夾是否存在
    if not os.path.exists(args.test_dir):
        print(f"錯誤: 測試資料夾 '{args.test_dir}' 不存在")
        return
    
    # 檢查模型檔案是否存在
    model_files = [f"fold_{i:02d}_best.h5" for i in range(1, FOLDS + 1)]
    missing_models = []
    for model_file in model_files:
        model_path = os.path.join(args.model_dir, model_file)
        if not os.path.exists(model_path):
            missing_models.append(model_file)
    
    if missing_models:
        print(f"錯誤: 缺少以下模型檔案: {missing_models}")
        print("請確保已完成K-fold訓練並生成所有模型檔案")
        return
    
    print("=" * 60)
    print("InceptionResNetV2 Testing Script")
    print("=" * 60)
    print(f"測試資料夾: {args.test_dir}")
    print(f"模型目錄: {args.model_dir}")
    print(f"Folds數量: {FOLDS}")
    
    # ------------------------------------------------------------
    # ❶ 載入測試資料
    # ------------------------------------------------------------
    print("\n▶  載入測試資料 ...")
    X_test, y_test_str = load_images_from_folder(args.test_dir)
    
    # 建立標籤編碼器 (假設與訓練時相同)
    le = LabelEncoder()
    # 這裡需要根據您的實際類別順序進行調整
    # 如果知道確切的類別順序，可以直接設定
    le.classes_ = np.array(['A_positive', 'B_negative', 'C_other'])  # 請根據實際情況調整
    y_test = le.transform(y_test_str)
    
    print(f"  ➜ 測試集 {len(X_test)} 張")
    print(f"  ➜ 類別對應: {le.classes_.tolist()}")
    
    # 檢查類別分布
    unique, counts = np.unique(y_test_str, return_counts=True)
    print("  ➜ 類別分布:")
    for cls, count in zip(unique, counts):
        print(f"    {cls}: {count} 張")
    
    # ------------------------------------------------------------
    # ❷ 載入所有fold模型並進行預測
    # ------------------------------------------------------------
    print(f"\n▶  載入 {FOLDS} 個fold模型並進行預測 ...")
    
    # 為每個fold收集詳細指標
    fold_results = []
    fold_aucs = []
    
    print("\n======== Individual Fold Results ========")
    for fold in range(1, FOLDS + 1):
        print(f"\n--- Fold {fold:02d} ---")
        
        # 載入該fold的模型
        model_path = os.path.join(args.model_dir, f"fold_{fold:02d}_best.h5")
        model = keras.models.load_model(model_path)
        
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
        y_test_bin = label_binarize(y_test, classes=range(NUM_CLASS))
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
    
    # ------------------------------------------------------------
    # ❸ 結果分析與統計
    # ------------------------------------------------------------
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
    if 'weighted avg' in best_report:
        results_data.append({
            'Class': 'Weighted Avg',
            'Precision': f"{best_report['weighted avg']['precision']:.4f}",
            'Recall': f"{best_report['weighted avg']['recall']:.4f}",
            'F1-Score': f"{best_report['weighted avg']['f1-score']:.4f}",
            'Support': f"{best_report['weighted avg']['support']:.0f}",
            'AUC': f"{best_fold_result['auc']:.4f}"
        })
    
    # 添加micro和macro平均
    if 'micro avg' in best_report:
        results_data.append({
            'Class': 'Micro Avg',
            'Precision': f"{best_report['micro avg']['precision']:.4f}",
            'Recall': f"{best_report['micro avg']['recall']:.4f}",
            'F1-Score': f"{best_report['micro avg']['f1-score']:.4f}",
            'Support': f"{best_report['micro avg']['support']:.0f}",
            'AUC': f"{best_fold_result['micro_auc']:.4f}"
        })
    
    if 'macro avg' in best_report:
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
    
    # ------------------------------------------------------------
    # ❹ 結果輸出 (可選)
    # ------------------------------------------------------------
    if args.output_file:
        print(f"\n▶  將結果儲存至: {args.output_file}")
        
        # 準備輸出資料
        output_data = {
            'test_directory': args.test_dir,
            'total_samples': len(X_test),
            'class_distribution': dict(zip(unique, counts)),
            'best_fold': best_fold,
            'best_fold_auc': fold_aucs[best_fold_idx],
            'fold_results': fold_results,
            'kfold_statistics': {
                metric: {
                    'mean': np.mean([fold_results[i][metric] for i in range(FOLDS)]),
                    'std': np.std([fold_results[i][metric] for i in range(FOLDS)])
                } for metric in metrics_names
            },
            'class_auc_statistics': {
                le.classes_[i]: {
                    'mean': np.mean([fold_results[j]['class_aucs'][i] for j in range(FOLDS)]),
                    'std': np.std([fold_results[j]['class_aucs'][i] for j in range(FOLDS)])
                } for i in range(NUM_CLASS)
            }
        }
        
        # 儲存為JSON格式
        import json
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        print("  ➜ 結果已成功儲存")
    
    # ------------------------------------------------------------
    # ❺ CSV結果輸出 (可選)
    # ------------------------------------------------------------
    if args.csv_output or args.csv_prefix:
        print(f"\n▶  將CSV結果儲存...")
        
        # 確定CSV輸出目錄
        csv_dir = args.csv_output if args.csv_output else '.'
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        # 1. 個別Fold結果CSV
        fold_csv_data = []
        for fold_result in fold_results:
            fold_csv_data.append({
                'Fold': fold_result['fold'],
                'Accuracy': fold_result['accuracy'],
                'Precision': fold_result['precision'],
                'Recall': fold_result['recall'],
                'F1_Score': fold_result['f1_score'],
                'Kappa': fold_result['kappa'],
                'AUC': fold_result['auc'],
                'Micro_AUC': fold_result['micro_auc'],
                'Macro_AUC': fold_result['macro_auc']
            })
        
        # 添加統計行
        fold_csv_data.append({
            'Fold': 'Mean ± Std',
            'Accuracy': f"{np.mean([fold_results[i]['accuracy'] for i in range(FOLDS)]):.4f} ± {np.std([fold_results[i]['accuracy'] for i in range(FOLDS)]):.4f}",
            'Precision': f"{np.mean([fold_results[i]['precision'] for i in range(FOLDS)]):.4f} ± {np.std([fold_results[i]['precision'] for i in range(FOLDS)]):.4f}",
            'Recall': f"{np.mean([fold_results[i]['recall'] for i in range(FOLDS)]):.4f} ± {np.std([fold_results[i]['recall'] for i in range(FOLDS)]):.4f}",
            'F1_Score': f"{np.mean([fold_results[i]['f1_score'] for i in range(FOLDS)]):.4f} ± {np.std([fold_results[i]['f1_score'] for i in range(FOLDS)]):.4f}",
            'Kappa': f"{np.mean([fold_results[i]['kappa'] for i in range(FOLDS)]):.4f} ± {np.std([fold_results[i]['kappa'] for i in range(FOLDS)]):.4f}",
            'AUC': f"{np.mean([fold_results[i]['auc'] for i in range(FOLDS)]):.4f} ± {np.std([fold_results[i]['auc'] for i in range(FOLDS)]):.4f}",
            'Micro_AUC': f"{np.mean([fold_results[i]['micro_auc'] for i in range(FOLDS)]):.4f} ± {np.std([fold_results[i]['micro_auc'] for i in range(FOLDS)]):.4f}",
            'Macro_AUC': f"{np.mean([fold_results[i]['macro_auc'] for i in range(FOLDS)]):.4f} ± {np.std([fold_results[i]['macro_auc'] for i in range(FOLDS)]):.4f}"
        })
        
        df_fold_csv = pd.DataFrame(fold_csv_data)
        fold_csv_path = os.path.join(csv_dir, f"{args.csv_prefix}_fold_results.csv")
        df_fold_csv.to_csv(fold_csv_path, index=False, encoding='utf-8-sig')
        print(f"  ➜ Fold結果已儲存至: {fold_csv_path}")
        
        # 2. 類別別AUC統計CSV
        class_auc_csv_data = []
        for i, class_name in enumerate(le.classes_):
            class_aucs = [fold_results[j]['class_aucs'][i] for j in range(FOLDS)]
            mean_auc = np.mean(class_aucs)
            std_auc = np.std(class_aucs)
            class_auc_csv_data.append({
                'Class': class_name,
                'Mean_AUC': mean_auc,
                'Std_AUC': std_auc,
                'Mean_Std_AUC': f"{mean_auc:.4f} ± {std_auc:.4f}"
            })
        
        df_class_auc_csv = pd.DataFrame(class_auc_csv_data)
        class_auc_csv_path = os.path.join(csv_dir, f"{args.csv_prefix}_class_auc_stats.csv")
        df_class_auc_csv.to_csv(class_auc_csv_path, index=False, encoding='utf-8-sig')
        print(f"  ➜ 類別AUC統計已儲存至: {class_auc_csv_path}")
        
        # 3. 最佳Fold詳細分類報告CSV
        best_classification_csv_data = []
        for i, class_name in enumerate(le.classes_):
            if class_name in best_report:
                best_classification_csv_data.append({
                    'Class': class_name,
                    'Precision': best_report[class_name]['precision'],
                    'Recall': best_report[class_name]['recall'],
                    'F1_Score': best_report[class_name]['f1-score'],
                    'Support': best_report[class_name]['support'],
                    'AUC': best_fold_result['class_aucs'][i]
                })
        
        # 添加平均行
        if 'weighted avg' in best_report:
            best_classification_csv_data.append({
                'Class': 'Weighted_Avg',
                'Precision': best_report['weighted avg']['precision'],
                'Recall': best_report['weighted avg']['recall'],
                'F1_Score': best_report['weighted avg']['f1-score'],
                'Support': best_report['weighted avg']['support'],
                'AUC': best_fold_result['auc']
            })
        
        if 'micro avg' in best_report:
            best_classification_csv_data.append({
                'Class': 'Micro_Avg',
                'Precision': best_report['micro avg']['precision'],
                'Recall': best_report['micro avg']['recall'],
                'F1_Score': best_report['micro avg']['f1-score'],
                'Support': best_report['micro avg']['support'],
                'AUC': best_fold_result['micro_auc']
            })
        
        if 'macro avg' in best_report:
            best_classification_csv_data.append({
                'Class': 'Macro_Avg',
                'Precision': best_report['macro avg']['precision'],
                'Recall': best_report['macro avg']['recall'],
                'F1_Score': best_report['macro avg']['f1-score'],
                'Support': best_report['macro avg']['support'],
                'AUC': best_fold_result['macro_auc']
            })
        
        df_best_classification_csv = pd.DataFrame(best_classification_csv_data)
        best_classification_csv_path = os.path.join(csv_dir, f"{args.csv_prefix}_best_fold_classification.csv")
        df_best_classification_csv.to_csv(best_classification_csv_path, index=False, encoding='utf-8-sig')
        print(f"  ➜ 最佳Fold分類報告已儲存至: {best_classification_csv_path}")
        
        # 4. 測試資訊摘要CSV
        summary_csv_data = [{
            'Test_Directory': args.test_dir,
            'Total_Samples': len(X_test),
            'Best_Fold': best_fold,
            'Best_Fold_AUC': fold_aucs[best_fold_idx],
            'Overall_Accuracy': best_fold_result['accuracy'],
            'Cohen_Kappa': best_fold_result['kappa']
        }]
        
        # 添加類別分布
        for cls, count in zip(unique, counts):
            summary_csv_data[0][f'Class_{cls}_Count'] = count
        
        df_summary_csv = pd.DataFrame(summary_csv_data)
        summary_csv_path = os.path.join(csv_dir, f"{args.csv_prefix}_test_summary.csv")
        df_summary_csv.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"  ➜ 測試摘要已儲存至: {summary_csv_path}")
        
        print("  ➜ 所有CSV檔案已成功儲存")
    
    print("\n" + "=" * 60)
    print("測試完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()