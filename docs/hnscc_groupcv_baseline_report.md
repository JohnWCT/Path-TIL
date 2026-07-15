# HNSCC GroupCV Baseline 實驗報告

## 1. 摘要

- 實驗日期：2026-07-15
- 資料：TVGH HNSCC QuPath patches，10 cases、5,492 patches
- 驗證方式：5-fold case-level GroupCV，每 fold 7 train、1 validation、2 held-out test cases
- 模型：TILScout InceptionResNetV2 pretrained model
- 訓練：Stage 1 凍結 backbone；Stage 2 解凍 backbone fine-tune
- 正式 baseline：H&E normalization on、heavy augmentation、class weight on
- OOF 覆蓋：5,492/5,492 patches、10/10 cases，各 patch 恰好作為 held-out test 一次

主要結果：

1. HNSCC fine-tuning 明顯優於未 fine-tune 的 Stage 0。
2. Stage 2 的 positive-vs-rest AUC 最佳（0.8512），slide-level TIL MAE 也由 0.2708 降至 0.1625。
3. Stage 1 的 weighted OVR AUC 最佳（0.8678），顯示完全解凍後對 positive 類別較有利，但整體 weighted discrimination 不一定同步提升。
4. 依 validation AUC 選 stage 的模型具有最高 accuracy（0.7152）與最低 TIL MAE（0.1619），但 positive AUC（0.8199）低於固定使用 Stage 2。
5. 目前 TIL 誤差主要不是系統性低估，而是在多數 cases 上高估；H0008、H0013、H0006 的 absolute error 最大。

## 2. 資料與標籤

輸入 manifest：`qupath_dataset.csv`

| 類別 | Patch 數 | 比例 |
|---|---:|---:|
| positive | 510 | 9.29% |
| negative | 2,159 | 39.31% |
| other | 2,823 | 51.40% |
| 合計 | 5,492 | 100.00% |

類別索引固定為：

```text
positive = 0
negative = 1
other    = 2
```

每個 case 都包含三個類別。`image_path` 已全數驗證存在，沒有 null 或重複路徑。

## 3. Case-level folds

| Fold | Validation case | Held-out test cases |
|---:|---|---|
| 0 | H0018 | H0008、H0013 |
| 1 | H0103 | H0002、H0005 |
| 2 | H0005 | H0041、H0103 |
| 3 | H0006 | H0003、H0007 |
| 4 | H0041 | H0006、H0018 |

每個 fold：

- train：7 cases
- validation：1 case
- test：2 cases
- 同一 case 在同一 fold 只會有一個 role
- 每個 case 在五個 folds 中恰好作為 held-out test 一次

因此 OOF 評估不會讓同一 case 的 patches 同時進入該模型的 train 與 test。

## 4. 訓練設定

| 設定 | 值 |
|---|---|
| Pretrained model | `best_InceptionResNetV2_model.h5` |
| Image size | 224 × 224 |
| Color | OpenCV BGR → RGB |
| H&E normalization | on |
| H&E 條件 | mean < 230 且 std > 15 |
| Augmentation | heavy，僅 train |
| Class weight | on，僅由各 fold train cases 計算 |
| Stage 1 | backbone frozen，Adam 1e-3 |
| Stage 2 | 載入 Stage 1 best，backbone unfrozen，Adam 1e-5 |
| Epoch 上限 | Stage 1 30；Stage 2 30 |
| Batch size | 32 |
| Seed | 42 |
| Model selection | validation Keras multiclass AUC |
| Test 使用 | 所有訓練及 stage 選擇完成後才評估 |

資料使用規則：

- augmentation 只套用 train。
- validation/test 不套用 augmentation。
- class weights 只由 train labels 計算。
- H&E normalization 為逐 patch 決定性轉換，不估計跨資料集統計。
- Stage 0、1、2 都在訓練完成後輸出 train/validation/test predictions，test 不參與 checkpoint、early stopping、LR 調整或 stage 選擇。

## 5. H&E normalization 狀態

| 狀態 | Patch 數 |
|---|---:|
| 成功套用 | 5,474 |
| 因亮度／變異條件略過 | 0 |
| 數值失敗後退回原始 RGB | 18 |

18 個失敗 patches 中，17 個來自 H0003，1 個來自 H0018。錯誤主要為除以零、invalid divide 或 exponential overflow。流程沒有中止，而是使用未 normalization 的 RGB patch；這會形成小比例的混合前處理，後續應改善 normalization 穩定性並做敏感度分析。

## 6. OOF patch-level 結果

| Stage | Accuracy | Macro F1 | Weighted F1 | Kappa | Positive AUC | Macro OVR AUC | Weighted OVR AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage 0 pretrained | 0.6054 | 0.5593 | 0.6440 | 0.3909 | 0.7815 | 0.8107 | 0.8140 |
| Stage 1 frozen backbone | 0.6963 | 0.6247 | 0.7178 | 0.5060 | 0.8211 | **0.8554** | **0.8678** |
| Stage 2 fine-tune | 0.7096 | **0.6456** | 0.7240 | 0.5235 | **0.8512** | 0.8479 | 0.8473 |
| Validation-selected | **0.7152** | 0.6382 | **0.7297** | **0.5263** | 0.8199 | 0.8477 | 0.8581 |

相對 Stage 0：

- Stage 1：accuracy +0.0909、positive AUC +0.0396、weighted AUC +0.0538。
- Stage 2：accuracy +0.1042、positive AUC +0.0697、macro F1 +0.0863。
- Validation-selected：accuracy +0.1098、weighted AUC +0.0441。

解讀：

- Fine-tuning 確實改善 case-level held-out 泛化，不只是 train 指標上升。
- Stage 1 對整體三類 weighted AUC 最穩定。
- Stage 2 對研究重點 positive-vs-rest AUC 最有利。
- Validation-selected 在 Fold 3 選 Stage 1，其餘 folds 選 Stage 2；這提高 accuracy 並微幅降低 TIL MAE，但犧牲整體 positive AUC。

## 7. Stage 選擇

| Fold | Stage 0 val AUC | Stage 1 val AUC | Stage 2 val AUC | 選擇 |
|---:|---:|---:|---:|---:|
| 0 | 0.7350 | 0.8866 | **0.9416** | 2 |
| 1 | 0.7370 | 0.8290 | **0.8783** | 2 |
| 2 | 0.7064 | 0.8330 | **0.8345** | 2 |
| 3 | 0.7679 | **0.8734** | 0.8595 | 1 |
| 4 | 0.7572 | 0.8875 | **0.9144** | 2 |

Fold 2 的 Stage 1/2 validation AUC 差距只有 0.0016，stage 選擇對單一 validation case 的抽樣變異可能敏感。後續報告應同時保留固定 Stage 2 與 validation-selected 兩種結果，不應只呈現其中一種。

## 8. Slide-level TIL 結果

TIL score 維持 TILScout hard-class 定義：

```text
TIL score = Positive / (Positive + Negative)
```

`other` 不進入分子或分母。

| Stage | TIL MAE | Median absolute error | Spearman | Pearson |
|---|---:|---:|---:|---:|
| Stage 0 pretrained | 0.2708 | 0.2516 | 0.7091 | 0.6971 |
| Stage 1 frozen backbone | 0.1893 | 0.1852 | 0.7697 | 0.7210 |
| Stage 2 fine-tune | 0.1625 | 0.1645 | **0.8182** | **0.8377** |
| Validation-selected | **0.1619** | 0.1645 | **0.8182** | 0.8116 |

Fine-tuning 將 TIL MAE 相對 Stage 0 降低約 40%：

```text
(0.2708 - 0.1619) / 0.2708 ≈ 40.2%
```

### Validation-selected 每 case 結果

| Case | Fold | GT TIL | Pred TIL | Absolute error | 偏差方向 |
|---|---:|---:|---:|---:|---|
| H0002 | 1 | 0.0173 | 0.0513 | 0.0340 | 高估 |
| H0003 | 3 | 0.0838 | 0.1997 | 0.1160 | 高估 |
| H0005 | 1 | 0.4194 | 0.5200 | 0.1006 | 高估 |
| H0006 | 4 | 0.4156 | 0.6349 | 0.2193 | 高估 |
| H0007 | 3 | 0.7327 | 0.7105 | 0.0221 | 低估 |
| H0008 | 0 | 0.0182 | 0.3394 | **0.3212** | 高估 |
| H0013 | 0 | 0.4963 | 0.7910 | **0.2948** | 高估 |
| H0018 | 4 | 0.1250 | 0.2823 | 0.1573 | 高估 |
| H0041 | 2 | 0.1368 | 0.3185 | 0.1817 | 高估 |
| H0103 | 2 | 0.4151 | 0.2435 | 0.1716 | 低估 |

8/10 cases 為高估。最大誤差來自 H0008、H0013、H0006。現有資料沒有 pale-stain metadata，因此不能把這些誤差直接歸因於染色偏淡；需要先做 stain statistics 或人工 pathology review。

## 9. 目前結論

在 case-level held-out setting 下，pretrained TILScout 模型經 HNSCC fine-tuning 後：

- patch-level accuracy、F1、positive AUC 均改善；
- slide-level TIL MAE 明顯下降；
- TIL 排序相關性提升；
- 改善出現在完整 OOF held-out cases，不只是 train；
- 尚未解決 case-specific TIL 高估與少量 H&E normalization 數值失敗。

因此可以確認第一階段假設成立：HNSCC case-level fine-tuning 能改善本地資料的泛化，但下一階段應優先處理前處理穩定性、stage 選擇目標與 TIL calibration，而不是立即更換 backbone。

## 10. 下一步優化計畫

### Phase A：單因子 ablation

保持 folds、seed、pretrained checkpoint、epoch 上限及評估程式固定，每次只修改一個變因。

主要 endpoint：

1. OOF positive-vs-rest AUC
2. slide-level TIL MAE

次要 endpoint：

- macro OVR AUC
- macro F1
- Spearman/Pearson
- 每 case absolute error

建議順序：

#### A1. H&E normalization on vs off

目的：

- 確認 normalization 是否真正改善跨 case 泛化。
- 判斷 18 個 normalization failures 是否影響 H0003/H0018。
- 補上真正未 normalization 的 Stage 0 pretrained baseline。

固定其他設定為 heavy augmentation、class weight on。

#### A2. Heavy vs medium augmentation

在 A1 選出的 normalization 設定下比較：

- heavy：imgaug HED、noise、blur、cutout
- medium：TensorFlow flip、rotation、zoom、translation、contrast

目的：

- 檢查 heavy augmentation 是否過度扭曲 positive morphology。
- 比較 GPU 效率、訓練穩定性及 held-out positive AUC。

#### A3. Class weight on vs off

在 A1/A2 最佳設定下比較：

- class weight on
- class weight off

目前 positive 只有 9.29%，class weight 有合理性，但 Stage 2 對部分 cases 出現 TIL 高估，需確認是否因 positive class weight 過強。

### Phase B：錯誤分析與 calibration

#### B1. Case-level error audit

優先檢查：

- H0008
- H0013
- H0006
- H0103

產出：

- 每 case confusion matrix
- positive/negative/other confidence distribution
- 高信心 false-positive patches
- stain brightness、RGB/OD/HED statistics
- pathology error gallery

目標是區分：

- stain domain shift
- QuPath label noise
- positive/negative morphology confusion
- `other` 被錯分為 positive

#### B2. Stage 選擇策略

目前 validation-selected 對 TIL MAE 最佳，但固定 Stage 2 對 positive AUC 最佳。下一輪應預先指定主要用途：

- 若目標是 positive patch discrimination：固定 Stage 2。
- 若目標是 slide TIL estimation：使用 validation-selected，並評估 validation metric 與 TIL MAE 的一致性。

不可用 held-out test 指標反向選 stage。

#### B3. TIL calibration

保留 hard-class TIL 作為主要、可與 TILScout 比較的指標，另新增次要分析：

```text
soft TIL = sum(prob_positive) /
           (sum(prob_positive) + sum(prob_negative))
```

比較 hard/soft TIL 的 MAE、相關性及 case-specific bias。若進一步做 calibration，參數只能由 train/validation 或獨立 calibration set 估計，不可使用當 fold test case。

### Phase C：前處理可靠性

改善 `norm_HnE`：

- 對有效 OD pixels 數量設定最低門檻。
- 檢查 covariance、eigenvalues、concentration scale 是否 finite。
- 對 divide-by-zero/overflow 提供明確 fallback reason。
- 將 failure count 納入正式 metrics。
- 對 H0003/H0018 做 normalization on/off paired comparison。

### Phase D：進階模型

只有 Phase A–C 完成且 baseline 變因已釐清後，再依序考慮：

1. focal loss
2. EWC
3. relaxed patch filter
4. EfficientNet/ResNet backbone
5. ViT

不建議同時引入多項變更，否則無法判斷改善來源。

## 11. 建議的實驗決策門檻

相較目前 validation-selected baseline，候選設定至少符合以下之一，且不能造成另一主要 endpoint 明顯惡化：

- positive AUC 絕對提升 ≥ 0.02
- TIL MAE 絕對下降 ≥ 0.02
- 最大 per-case absolute error 明顯下降

由於只有 10 cases，除了點估計外，下一版評估應加入 case-level bootstrap 95% confidence intervals，並報告 paired per-case difference，避免只依單次平均值決策。

## 12. 結果位置

正式訓練：

```text
/workspace/results_groupcv_norm_heavy/
```

各 stage OOF：

```text
oof_stage_0/
oof_stage_1/
oof_stage_2/
oof_stage_selected/
```

每個 OOF 目錄包含：

```text
oof_predictions.csv
patch_auc_summary.csv
slide_til_score_summary.csv
eval_summary.json
```

本報告只提交彙整結果與方法，不提交 `.h5` checkpoints、逐 patch CSV 或其他大型實驗輸出。
