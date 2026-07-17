# -*- coding: utf-8 -*-
"""
InceptionResNetV2 Inference Script
使用訓練好的模型進行推論並輸出CSV結果
"""

import numpy as np
import glob
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ========= 命令列參數設定 =========
parser = argparse.ArgumentParser(description='InceptionResNetV2 Inference')
parser.add_argument('--model-path', type=str, default='baselines/best_InceptionResNetV2_model.h5',
                    help='模型檔案路徑 (預設: baselines/best_InceptionResNetV2_model.h5)')
parser.add_argument('--input-dir', type=str, default='./Overlap0_in151to150',
                    help='輸入資料夾路徑 (預設: ./Overlap0_in151to150)')
parser.add_argument('--output-dir', type=str, default='./results/inference_results',
                    help='輸出結果資料夾路徑 (預設: ./results/inference_results)')
parser.add_argument('--batch-size', type=int, default=8,
                    help='推論的 batch size (預設: 8)')
args = parser.parse_args()

# ========= 共用常數 =========
IMG_SIZE = 224
IMG_CH = 3
NUM_CLASS = 3

print(f"模型路徑: {args.model_path}")
print(f"輸入資料夾: {args.input_dir}")
print(f"輸出資料夾: {args.output_dir}")
print(f"Batch size: {args.batch_size}")
print(f"處理模式: 自動處理所有子資料夾")

# GPU記憶體管理
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 限制GPU記憶體增長
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ 找到 {len(gpus)} 個GPU，已啟用記憶體增長")
    except RuntimeError as e:
        print(f"GPU設定錯誤: {e}")
else:
    print("⚠ 未找到GPU，將使用CPU")

# ========= 模型建構函數 =========
def build_model():
    """建構與訓練時相同的模型架構 (推論時不使用資料增強)"""
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CH), dtype=tf.uint8)
    x = tf.cast(inp, tf.float32)
    x = preprocess_input(x)

    backbone = keras.applications.InceptionResNetV2(
        include_top=False, input_tensor=x, pooling="avg", weights="imagenet"
    )
    x = Dense(512, activation='relu')(backbone.output)
    out = Dense(NUM_CLASS, activation='softmax')(x)
    model = keras.Model(inp, out)
    return model

# ========= 載入影像函數 =========


def scan_png_files(folder_path):
    """掃描指定資料夾內的所有PNG檔案路徑"""
    img_paths = []
    
    print(f"開始掃描資料夾: {folder_path}")
    
    # 檢查資料夾是否存在
    if not os.path.exists(folder_path):
        print(f"✗ 資料夾不存在: {folder_path}")
        return img_paths
    
    # 掃描所有子資料夾中的PNG檔案
    png_count = 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                png_count += 1
                img_path = os.path.join(root, file)
                img_paths.append(img_path)
    
    print(f"總共找到 {png_count} 個PNG檔案")
    return img_paths



def create_dataloader(img_paths, batch_size=8):
    """創建DataLoader"""
    # 使用TensorFlow的DataLoader
    dataloader = tf.data.Dataset.from_tensor_slices(img_paths)
    dataloader = dataloader.map(
        lambda x: tf.py_function(
            lambda path: tf.convert_to_tensor(load_and_preprocess_image(path.numpy().decode()), tf.uint8),
            [x], tf.uint8
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataloader = dataloader.batch(batch_size)
    dataloader = dataloader.prefetch(tf.data.AUTOTUNE)
    
    return dataloader

def load_and_preprocess_image(img_path):
    """載入並預處理單張影像"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.uint8)
    else:
        print(f"✗ 無法載入影像: {img_path}")
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

# ========= 解析檔案路徑函數 =========
def parse_image_path(img_path):
    """解析影像路徑，提取img_name, x, y"""
    # 移除副檔名
    filename = os.path.basename(img_path)
    filename_without_ext = os.path.splitext(filename)[0]
    
    # 分割路徑以取得img_name (資料夾名稱)
    path_parts = img_path.split(os.sep)
    # 尋找包含patches/Tissue的路徑結構
    for i, part in enumerate(path_parts):
        if part == 'patches' and i + 1 < len(path_parts) and path_parts[i + 1] == 'Tissue':
            # img_name是patches/Tissue之前的資料夾名稱
            img_name = path_parts[i - 1]
            break
    else:
        # 如果找不到patches/Tissue結構，使用倒數第二個資料夾名稱
        img_name = path_parts[-3] if len(path_parts) >= 3 else "unknown"
    
    # 解析x;y座標
    if ';' in filename_without_ext:
        coords = filename_without_ext.split(';')
        if len(coords) == 2:
            img_x = coords[0]
            img_y = coords[1]
        else:
            img_x = "0"
            img_y = "0"
    else:
        img_x = "0"
        img_y = "0"
    
    return img_name, img_x, img_y

# ========= 輸出記錄函數 =========
def output_record(output_path, data_val_paths, test_predictions, test_conf, test_labels, test_score, class_map, class_back_map, Pred_wrong=False):
    """輸出推論結果到CSV檔案"""
    img_files = []
    x_list = []
    y_list = []
    
    for i_path in data_val_paths:
        img_name, img_x, img_y = parse_image_path(i_path)
        img_files.append(img_name)
        x_list.append(img_x)
        y_list.append(img_y)

    inference_dict = {
        "Image": img_files,
        "x": x_list,
        "y": y_list,
        "prediction": test_predictions,
        "confidence": test_conf,
        "Truth": test_labels,
    }

    # 添加每個類別的信心度
    for i_class in class_map:
        inference_dict[f'{i_class}_conf'] = test_score[:, class_map[i_class]]
    
    # 添加非 A_positive 類別的信心度 (假設第一個類別是 A_positive)
    if 0 in class_map.values():
        inference_dict['Not_A_positive_conf'] = [(1-x) for x in test_score[:, 0]]

    inference_result = pd.DataFrame(inference_dict)
    inference_result = inference_result.replace({'prediction': class_back_map, 'Truth': class_back_map})
    
    # 創建輸出資料夾
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f'創建輸出資料夾: {output_path}')

    # 為每個影像輸出CSV檔案
    csv_count = 0
    for i_img in set(inference_result['Image']):
        df_out = inference_result[inference_result['Image']==i_img].drop('Image', axis=1)
        output_file = os.path.join(output_path, f"{i_img}.csv")
        df_out.to_csv(path_or_buf=output_file, index=False)
        csv_count += 1
    
    print(f"完成輸出 {csv_count} 個CSV檔案")
    
    # 輸出預測錯誤的檔案
    if Pred_wrong:
        df_diff_out = inference_result[inference_result['prediction']!=inference_result['Truth']]
        wrong_file = os.path.join(output_path, 'Pred_wrong_patch.csv')
        df_diff_out.to_csv(path_or_buf=wrong_file, index=False)
        print(f"已輸出預測錯誤檔案: {wrong_file}")

# ========= 主程式 =========
def main():
    print("▶ 開始載入模型...")
    
    # 載入模型
    try:
        model = keras.models.load_model(args.model_path)
        print(f"✓ 成功載入模型: {args.model_path}")
    except Exception as e:
        print(f"✗ 載入模型失敗: {e}")
        print("嘗試重新建構模型...")
        model = build_model()
        model.load_weights(args.model_path)
        print("✓ 成功載入模型權重")
    
    # 設定類別對應 (請根據實際訓練時的類別調整)
    # 這裡假設類別順序為: ['A_positive', 'B_negative', 'C_other']
    # 請根據實際訓練時的LabelEncoder結果調整
    class_names = ['A_positive', 'B_negative', 'C_other']  # 請根據實際情況調整
    class_map = {name: i for i, name in enumerate(class_names)}
    class_back_map = {i: name for i, name in enumerate(class_names)}
    
    print(f"類別對應: {class_map}")
    
    print("▶ 掃描PNG檔案...")
    print(f"正在搜尋資料夾: {args.input_dir}")
    
    # 檢查輸入資料夾是否存在
    if not os.path.exists(args.input_dir):
        print(f"✗ 輸入資料夾不存在: {args.input_dir}")
        return
    
    # 掃描所有PNG檔案路徑
    img_paths = scan_png_files(args.input_dir)
    print(f"✓ 找到 {len(img_paths)} 個PNG檔案")
    
    if len(img_paths) == 0:
        print("✗ 沒有找到任何PNG影像檔案")
        print("請檢查:")
        print(f"  1. 資料夾 {args.input_dir} 是否存在")
        print(f"  2. 資料夾中是否包含 PNG 檔案")
        print(f"  3. 資料夾結構是否正確")
        return
    
    if len(img_paths) > 0:
        print(f"範例影像路徑: {img_paths[0]}")
    
    print("▶ 創建DataLoader...")
    dataloader = create_dataloader(img_paths, args.batch_size)
    print(f"✓ DataLoader創建完成，批次大小: {args.batch_size}")
    
    print("▶ 開始推論...")
    print(f"準備對 {len(img_paths)} 張影像進行推論")
    
    # 計算記憶體使用量
    import psutil
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    print(f"推論前記憶體使用: {memory_before:.1f} MB")
    
    try:
        # 批次進行預測
        all_predictions = []
        total_batches = (len(img_paths) + args.batch_size - 1) // args.batch_size
        
        # 使用tqdm顯示進度
        with tqdm(total=total_batches, desc="推論進度", unit="批次") as pbar:
            for batch in dataloader:
                # 進行預測
                batch_predictions = model.predict(batch, verbose=0)
                all_predictions.append(batch_predictions)
                pbar.update(1)
        
        # 合併所有預測結果
        predictions = np.vstack(all_predictions)
        print(f"預測完成，結果形狀: {predictions.shape}")
        
        # 取得預測結果
        y_pred = np.argmax(predictions, axis=1)
        y_conf = np.max(predictions, axis=1)
        
        # 轉換預測標籤為類別名稱
        y_pred_names = [class_back_map[pred] for pred in y_pred]
        
        # 由於這是純推論，沒有真實標籤，所以Truth欄位設為空
        y_true_names = [''] * len(y_pred_names)
        
        # 計算推論後記憶體使用量
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        print(f"推論後記憶體使用: {memory_after:.1f} MB")
        print(f"記憶體增加: {memory_after - memory_before:.1f} MB")
        
    except Exception as e:
        print(f"✗ 推論過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("▶ 輸出結果...")
    
    try:
        # 輸出結果
        output_record(
            output_path=args.output_dir,
            data_val_paths=img_paths,
            test_predictions=y_pred_names,
            test_conf=y_conf,
            test_labels=y_true_names,
            test_score=predictions,
            class_map=class_map,
            class_back_map=class_back_map,
            Pred_wrong=False  # 沒有真實標籤，所以不輸出錯誤檔案
        )
    except Exception as e:
        print(f"✗ 輸出結果時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 顯示統計資訊
    print("\n======== 推論統計 ========")
    print(f"總影像數: {len(img_paths)}")
    print(f"預測類別分布:")
    for class_name in class_names:
        count = y_pred_names.count(class_name)
        percentage = (count / len(y_pred_names)) * 100
        print(f"  {class_name}: {count} ({percentage:.2f}%)")
    
    print(f"\n平均信心度: {np.mean(y_conf):.4f}")
    print(f"最高信心度: {np.max(y_conf):.4f}")
    print(f"最低信心度: {np.min(y_conf):.4f}")
    
    print("\n✓ 推論完成!")

if __name__ == "__main__":
    main() 