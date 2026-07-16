# HNSCC GroupCV 優化測試 1–6 比較報告

## 1. 摘要

- 實驗日期：2026-07-15
- 資料：TVGH HNSCC，10 cases、5,492 patches
- 驗證：固定 5-fold case-level GroupCV；每 case 恰好作為 held-out test 一次
- 固定項目：folds、seed 42、pretrained checkpoint、Stage 1/2 epoch 上限 30、batch size 32
- 主要 endpoints：OOF positive-vs-rest AUC、hard-class slide TIL MAE
- 模型選擇：只依各 fold validation multiclass AUC，不使用 held-out test

完成的六項測試：

1. H&E normalization on/off
2. heavy/medium augmentation
3. class weight on/off
4. case-level error audit
5. Stage 0/1/2 與 validation-selected 比較
6. hard/soft TIL 與 cross-fitted calibration

主要結論：

1. 下一輪確認實驗的候選主要設定為 **H&E normalization off + heavy augmentation + class weight on + validation-selected stage**。
2. 此設定相較原 baseline，positive AUC 由 0.8199 增至 0.8555，hard TIL MAE 由 0.1619 降至 0.1428。
3. medium augmentation 的 positive AUC 顯著低於 heavy；paired case bootstrap 的差值 95% CI 不含 0。
4. 關閉 class weight 會提高 accuracy，但降低 positive AUC 並提高 hard TIL MAE；若改用 soft TIL，則 class weight off 的 MAE 最低。
5. validation-selected 在 4 folds 選 Stage 2、1 fold 選 Stage 1；相較固定 Stage 2，positive AUC 幾乎相同而 hard TIL MAE 較低。
6. 線性 calibration 並未穩定改善最佳設定，暫不應加入正式推論流程。
7. 樣本只有 10 cases；除 heavy vs medium 的 positive AUC 外，多數差值 CI 仍跨 0，結果應視為探索性證據。
8. **old_base**（`qupath_results_heavy/fold04_stage2_best.h5` 固定權重、case-level OOF 推論）在 H&E on 下 patch AUC 很高，但 hard TIL MAE 明顯差於 candidate；見第 2 節。

## 2. old_base：舊 QuPath fold04 權重的 case-level OOF 推論

### 定義

| 項目 | 設定 |
|---|---|
| 權重 | `qupath_results_heavy/fold04_stage2_best.h5` |
| 訓練來源 | 舊 `InceptionResNetV2_QuPath_2stage.py`（heavy aug、預設 H&E on、**patch-level StratifiedKFold**） |
| 本節評估 | 不重訓；只在目前 **case-level GroupCV** 的各 fold held-out test cases 上推論 |
| Folds | 與本報告其餘實驗相同的 `folds_hnscc_group5.csv` |
| 主要對照 | `old_base`（H&E on，對齊舊訓練前處理）vs candidate（H&E off + GroupCV fine-tune） |

重要限制：舊權重是以 patch-level StratifiedKFold 在同一批 QuPath HNSCC patches 上訓練，**同一 case 的 patches 很可能已進入該模型的訓練集**。因此本節 patch AUC／accuracy 對 case 泛化可能偏樂觀；仍適合作為「舊推論管線套到現行 by-case test 報表」的可比分數。

輸出目錄：

```text
results_old_base_qupath_fold04_hne_on/
results_old_base_qupath_fold04_hne_off/
```

### OOF 整體比較

| 實驗 | Accuracy | Macro F1 | Positive AUC | Macro OVR AUC | Weighted OVR AUC | Hard TIL MAE | Soft TIL MAE | Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **old_base**（fold04_stage2，H&E on） | **0.8350** | 0.6683 | **0.8912** | **0.9341** | **0.9497** | 0.2113 | **0.1275** | 0.6454 |
| old_base（fold04_stage2，H&E off） | 0.8108 | 0.6071 | 0.8307 | 0.8986 | 0.9228 | 0.2399 | 0.1436 | 0.7585 |
| GroupCV Stage 0 pretrained（H&E on） | 0.6054 | 0.5593 | 0.7815 | 0.8107 | 0.8140 | 0.2708 | 0.2700 | 0.6971 |
| GroupCV selected（H&E on / heavy / weight on） | 0.7152 | 0.6382 | 0.8199 | 0.8477 | 0.8581 | 0.1619 | 0.1806 | 0.8116 |
| **candidate** selected（H&E off / heavy / weight on） | 0.7480 | **0.6728** | 0.8555 | 0.8922 | 0.9056 | **0.1428** | 0.1627 | **0.7704** |

解讀：

1. 相對 pure pretrained Stage 0，old_base（已在 QuPath HNSCC 上 fine-tune 過的 fold04 權重）patch 指標明顯較高。
2. 相對 candidate：old_base 的 positive AUC／accuracy 較高，但 **hard TIL MAE 較差**（0.2113 vs 0.1428），且相關性較低。
3. old_base 的主要問題是 **系統性低估 TIL**（hard-class positive 過少）；soft TIL MAE 反而較佳（0.1275），顯示機率質量尚可，但 argmax／硬分類比例失真。
4. 若目標是 slide-level hard TIL（與 TILScout 定義一致），candidate GroupCV 流程優於直接拿舊 fold04 權重做 by-case 推論。
5. 舊權重在 H&E on 優於 H&E off，符合其訓練時預設啟用 `norm_HnE`。

### old_base（H&E on）每 case hard TIL

| Case | GT TIL | old_base Pred | old_base AE | candidate Pred | candidate AE |
|---|---:|---:|---:|---:|---:|
| H0002 | 0.0173 | 0.0019 | 0.0154 | 0.0276 | 0.0103 |
| H0003 | 0.0838 | 0.0150 | 0.0688 | 0.3018 | 0.2180 |
| H0005 | 0.4194 | 0.0789 | **0.3404** | 0.4857 | 0.0664 |
| H0006 | 0.4156 | 0.1582 | **0.2574** | 0.3168 | 0.0988 |
| H0007 | 0.7327 | 0.1639 | **0.5687** | 0.7237 | 0.0090 |
| H0008 | 0.0182 | 0.0963 | 0.0781 | 0.3851 | 0.3670 |
| H0013 | 0.4963 | 0.2800 | 0.2163 | 0.8780 | 0.3818 |
| H0018 | 0.1250 | 0.0268 | 0.0982 | 0.1958 | 0.0708 |
| H0041 | 0.1368 | 0.0599 | 0.0769 | 0.3096 | 0.1728 |
| H0103 | 0.4151 | 0.0222 | **0.3929** | 0.3824 | 0.0327 |

old_base 在高 TIL cases（H0005、H0006、H0007、H0103）明顯低估；candidate 在這些 cases 的 hard TIL 誤差大幅縮小，但在 H0008／H0013 轉為高估。兩者錯誤型態不同：舊權重偏保守，candidate 偏積極召回 positive。

## 3. 實驗矩陣

每輪只修改一個主要變因，後一輪使用前一輪的 point-estimate 較佳設定。

| 實驗 | H&E norm | Augmentation | Class weight | Stage 輸出 |
|---|---|---|---|---|
| 原 baseline | on | heavy | on | validation-selected |
| A1 | off | heavy | on | validation-selected |
| A2 | off | medium | on | validation-selected |
| A3 | off | heavy | off | validation-selected |

所有 OOF 檔案均通過以下檢查：

- 5,492/5,492 patches；
- 10/10 cases；
- 無重複 patch；
- prediction fold 與 case assignment 一致；
- 每列三類機率總和在允許誤差內等於 1；
- validation/test 不使用 augmentation。

## 4. A1–A3 OOF 整體結果

| 實驗 | Accuracy | Macro F1 | Positive AUC | Macro OVR AUC | Weighted OVR AUC | Hard TIL MAE | Soft TIL MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原 baseline：H&E on / heavy / weight on | 0.7152 | 0.6382 | 0.8199 | 0.8477 | 0.8581 | 0.1619 | 0.1806 |
| A1：H&E off / heavy / weight on | 0.7480 | **0.6728** | **0.8555** | **0.8922** | **0.9056** | **0.1428** | 0.1627 |
| A2：H&E off / medium / weight on | 0.7380 | 0.6535 | 0.8020 | 0.8649 | 0.8886 | 0.1834 | 0.1986 |
| A3：H&E off / heavy / weight off | **0.7742** | 0.6272 | 0.8226 | 0.8736 | 0.8915 | 0.1769 | **0.1282** |

依預先指定的兩個主要 endpoints，A1 設定最佳：

```text
H&E normalization off
+ heavy augmentation
+ class weight on
+ validation-selected stage
```

A3 的最高 accuracy 主要受多數類別影響，不代表 positive 類別或 TIL score 最佳，因此不以 accuracy 單獨決定正式設定。

## 5. Paired case-cluster bootstrap

使用相同 10 cases、相同 OOF folds 做 2,000 次 paired case-cluster bootstrap，seed 為 42。AUC 差值為「比較實驗減參考實驗」；MAE 差值小於 0 才是改善。

| 比較 | Metric | 差值中位數 | 95% CI | 解讀 |
|---|---|---:|---:|---|
| H&E off − on；其餘 heavy/weight on | Positive AUC | +0.0360 | [-0.0054, 0.0967] | point estimate 改善，CI 跨 0 |
| H&E off − on；其餘 heavy/weight on | Hard TIL MAE | -0.0199 | [-0.0672, 0.0293] | point estimate 改善，CI 跨 0 |
| Medium − heavy；其餘 H&E off/weight on | Positive AUC | -0.0494 | [-0.1228, -0.0109] | medium 較差，CI 不含 0 |
| Medium − heavy；其餘 H&E off/weight on | Hard TIL MAE | +0.0396 | [-0.0191, 0.1144] | medium point estimate 較差，CI 跨 0 |
| Weight off − on；其餘 H&E off/heavy | Positive AUC | -0.0301 | [-0.0618, 0.0258] | weight off point estimate 較差，CI 跨 0 |
| Weight off − on；其餘 H&E off/heavy | Hard TIL MAE | +0.0307 | [-0.1294, 0.2027] | weight off point estimate 較差，CI 很寬 |

10 cases 導致 CI 偏寬，不能把所有 point-estimate 差異解讀為已證實的泛化改善。A1/A3 應在增加外部 cases 後再次確認。

## 6. A1：H&E normalization on/off

### 整體

關閉 normalization 後：

- positive AUC：0.8199 → 0.8555；
- weighted OVR AUC：0.8581 → 0.9056；
- hard TIL MAE：0.1619 → 0.1428；
- accuracy：0.7152 → 0.7480；
- 不再有 18 個 normalization 數值失敗後 fallback 的混合前處理。

### 每 case hard TIL 誤差

| Case | H&E on AE | H&E off AE | Off − on | 結果 |
|---|---:|---:|---:|---|
| H0002 | 0.0340 | 0.0103 | -0.0237 | 改善 |
| H0003 | 0.1160 | 0.2180 | +0.1020 | 惡化 |
| H0005 | 0.1006 | 0.0664 | -0.0343 | 改善 |
| H0006 | 0.2193 | 0.0988 | -0.1205 | 改善 |
| H0007 | 0.0221 | 0.0090 | -0.0132 | 改善 |
| H0008 | 0.3212 | 0.3670 | +0.0457 | 惡化 |
| H0013 | 0.2948 | 0.3818 | +0.0870 | 惡化 |
| H0018 | 0.1573 | 0.0708 | -0.0865 | 改善 |
| H0041 | 0.1817 | 0.1728 | -0.0090 | 改善 |
| H0103 | 0.1716 | 0.0327 | -0.1389 | 改善 |

H&E off 改善 7/10 cases，但明顯惡化 H0003、H0008、H0013。整體平均較佳並不表示所有染色 domain 都受益；後續若增加資料，應評估 stain-aware normalization 或由 validation 決定 normalization，而不是對 test case 事後選擇。

## 7. A2：heavy vs medium augmentation

在 H&E off、class weight on 下：

- heavy positive AUC 0.8555，medium 0.8020；
- heavy hard TIL MAE 0.1428，medium 0.1834；
- heavy macro F1 0.6728，medium 0.6535；
- positive AUC 的 paired bootstrap 差值支持 heavy 優於 medium。

目前 heavy pipeline 的 HED variation、noise/blur 與 cutout，對跨 case positive discrimination 比 medium 的幾何與 contrast augmentation 更有效。這不代表每個 heavy transform 都必要；若要再精簡，應逐項 ablate，而不是直接改成 medium。

## 8. A3：class weight on/off

在 H&E off、heavy augmentation 下：

- weight on：positive AUC 0.8555、hard TIL MAE 0.1428；
- weight off：positive AUC 0.8226、hard TIL MAE 0.1769；
- weight off accuracy 較高：0.7742 vs 0.7480；
- weight off soft TIL MAE 較低：0.1282 vs 0.1627。

positive patches 只有 9.29%。關閉 class weight 讓 argmax 預測更偏向多數類別，因此整體 accuracy 上升，但 positive ranking 與 hard TIL 變差。正式 patch classifier 應保留 class weight；若主要用途改成 slide TIL estimation，可將「weight off + soft TIL」保留為次要候選，但需更多 cases 驗證。

## 9. B1：case-level error audit

最佳主要設定共有 1,384/5,492 錯誤 patches，accuracy 0.7480；5,492 張影像均可讀取。

### Positive 類別

| Case | Support | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| H0002 | 9 | 6 | 7 | 3 | 0.4615 | 0.6667 |
| H0003 | 50 | 18 | 221 | 32 | 0.0753 | 0.3600 |
| H0005 | 39 | 24 | 61 | 15 | 0.2824 | 0.6154 |
| H0006 | 64 | 30 | 21 | 34 | 0.5882 | 0.4688 |
| H0007 | 74 | 38 | 17 | 36 | 0.6909 | 0.5135 |
| H0008 | 2 | 1 | 56 | 1 | 0.0175 | 0.5000 |
| H0013 | 133 | 124 | 128 | 9 | 0.4921 | 0.9323 |
| H0018 | 37 | 22 | 34 | 15 | 0.3929 | 0.5946 |
| H0041 | 58 | 40 | 112 | 18 | 0.2632 | 0.6897 |
| H0103 | 44 | 20 | 32 | 24 | 0.3846 | 0.4545 |

主要問題不是單純 false negative，而是 case-specific false positive：

- H0003：221 positive false positives；
- H0013：128；
- H0041：112；
- H0005：61；
- H0008：56，但真實 positive 只有 2 patches。

H0008 的極低 positive prevalence 使 56 個 false positives 將 GT TIL 0.0182 推高至 0.3851。H0013 則同時具有高 positive recall 與大量 false positives，TIL 由 0.4963 高估至 0.8780。

### Confidence 與 stain 統計

多數 cases 的錯誤 patch confidence 低於正確 patch。例如：

- H0003：錯誤平均 confidence 0.6254，正確 0.8381；
- H0008：錯誤 0.6558，正確 0.8113；
- H0041：錯誤 0.6865，正確 0.8827。

錯誤 patch 與正確 patch 的 stain statistics 在部分 cases 有差異：

- H0003 錯誤 patch 較暗且 saturation 較高：RGB mean 187.46 vs 196.58；saturation 71.44 vs 60.40。
- H0008 錯誤 patch 較暗且 saturation 較高：RGB mean 168.95 vs 172.41；saturation 92.83 vs 89.43。
- H0013 的 RGB、saturation、OD 均非常接近，單靠全圖色彩統計無法解釋錯誤。

因此 stain shift 可能是部分原因，但不足以解釋所有 cases。已輸出 pathology error gallery 的候選清單與最高 confidence 錯誤 patch 路徑，仍需人工確認 morphology confusion 或 QuPath label noise。

產出：

```text
results_groupcv_nohne_heavy/error_audit_selected/
├── case_class_metrics.csv
├── confidence_stain_by_case.csv
├── confusion_by_case.csv
├── error_audit_summary.json
├── high_confidence_errors.csv
└── patch_error_audit.csv
```

## 10. B2：Stage 選擇策略

以下均使用最佳主要設定 H&E off / heavy / class weight on。

| Stage | Accuracy | Macro F1 | Positive AUC | Weighted OVR AUC | Hard TIL MAE | Pearson |
|---|---:|---:|---:|---:|---:|---:|
| Stage 0 pretrained | 0.5657 | 0.5336 | 0.7588 | 0.8193 | 0.3223 | 0.6459 |
| Stage 1 frozen backbone | 0.7125 | 0.6327 | 0.8136 | 0.8760 | 0.1590 | 0.6962 |
| Stage 2 fine-tune | 0.7351 | 0.6613 | 0.8508 | 0.9028 | 0.1693 | 0.6441 |
| Validation-selected | **0.7480** | **0.6728** | **0.8555** | **0.9056** | **0.1428** | **0.7704** |

Validation stage 選擇：

| Fold | Stage 0 val AUC | Stage 1 val AUC | Stage 2 val AUC | 選擇 |
|---:|---:|---:|---:|---:|
| 0 | 0.7459 | 0.8917 | 0.9515 | 2 |
| 1 | 0.7423 | 0.8557 | 0.8566 | 2 |
| 2 | 0.7115 | 0.8011 | 0.8598 | 2 |
| 3 | 0.7836 | 0.8655 | 0.8455 | 1 |
| 4 | 0.7788 | 0.8606 | 0.8893 | 2 |

相較固定 Stage 2，validation-selected：

- positive AUC point estimate +0.0047；
- hard TIL MAE -0.0265；
- paired bootstrap positive AUC delta 95% CI [-0.0061, 0.0090]；
- paired bootstrap hard TIL MAE delta 95% CI [-0.0646, 0.0000]。

只有 Fold 3 改用 Stage 1，其餘 folds 與 Stage 2 相同。現有結果支持保留 validation-selected 作為主要輸出，同時繼續報告固定 Stage 2，避免單一 validation case 造成的 stage selection 不穩定被隱藏。

## 11. B3：hard/soft TIL 與 calibration

定義：

```text
hard TIL = predicted Positive count /
           (predicted Positive count + predicted Negative count)

soft TIL = sum(prob_positive) /
           (sum(prob_positive) + sum(prob_negative))
```

Calibration 使用 leave-one-case-out 線性模型：每次只用其餘 9 cases 的 raw prediction 與 GT TIL 擬合，再預測被排除的 case，最後 clip 至 [0, 1]。因此沒有使用目標 case label 擬合其自身參數。

| 設定 | Hard raw MAE | Soft raw MAE | Hard calibrated MAE | Soft calibrated MAE |
|---|---:|---:|---:|---:|
| H&E on / heavy / weight on | 0.1619 | 0.1806 | **0.1311** | 0.1372 |
| H&E off / heavy / weight on | **0.1428** | 0.1627 | 0.1597 | 0.1644 |
| H&E off / medium / weight on | 0.1834 | 0.1986 | 0.1887 | 0.1968 |
| H&E off / heavy / weight off | 0.1769 | **0.1282** | 0.2317 | 0.1592 |

結果並不一致：

- 原 baseline 的 cross-fitted calibration point estimate 改善 MAE；
- 最佳主要設定經 calibration 後反而惡化；
- class weight off 時 soft raw 最佳，但 calibration 仍惡化；
- 以上 paired bootstrap CI 均跨 0。

只有 10 個 case-level calibration points，線性 slope/intercept 對單一 case 很敏感。現階段不應把 calibration 寫入正式 inference；保留 hard raw 為主要 TIL 指標，soft raw 作次要分析。

## 12. 建議設定與下一步

### 實驗解讀限制

- 本輪使用同一組 OOF cases 比較多個設定並選出最佳設定，因此 0.8555 AUC 與 0.1428 MAE 已帶有 model-selection optimism，不能再視為完全未觸碰的最終 test estimate。
- 目前不是 nested cross-validation。最佳設定應在獨立外部 cohort 或預先鎖定的新 cases 上做一次 final confirmation。
- 只有 10 cases；patch 數很多不等於有 5,492 個獨立生物樣本，信賴區間必須以 case 為抽樣單位。
- stain audit 是全 patch RGB、saturation、optical-density 摘要，不能取代病理影像人工判讀或更完整的 HED/morphology 分析。
- high-confidence error gallery 已產生路徑清單，但尚未由病理專家確認 label noise 與 morphology 類型。

### 下一輪候選主要設定

```text
H&E normalization: off
augmentation: heavy
class weight: on
stage: validation-selected
patch endpoint: positive-vs-rest AUC
slide endpoint: hard TIL MAE
```

### 次要輸出

- 固定 Stage 2 OOF metrics；
- soft TIL；
- class weight off + soft TIL 作研究性候選，不取代主要設定；
- 每 case confusion 與 high-confidence false positives。

### 下一輪優先順序

1. 人工檢查 H0003、H0008、H0013、H0041 的 high-confidence positive false positives。
2. 確認錯誤是 morphology confusion、`other`→positive、label noise 或 stain domain shift。
3. 增加獨立 cases 或外部 cohort，縮小 case-cluster bootstrap CI。
4. 若仍要改善 class imbalance，優先做 focal loss 單因子測試；不可同時更換 backbone。
5. Calibration 必須等待更多 case-level calibration points，或使用獨立 calibration cohort。

## 13. 可重現輸出

比較與 audit 工具：

```text
scripts/analyze_hnscc_errors.py
scripts/compare_hnscc_experiments.py
scripts/compare_hnscc_til_estimators.py
scripts/eval_hnscc_oof.py
scripts/eval_old_base_case_oof.py
```

old_base 重跑：

```bash
python3 scripts/eval_old_base_case_oof.py \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --model qupath_results_heavy/fold04_stage2_best.h5 \
  --output-dir results_old_base_qupath_fold04_hne_on \
  --hne-norm on

python3 scripts/eval_hnscc_oof.py \
  --pred-dir results_old_base_qupath_fold04_hne_on \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --stage selected \
  --output results_old_base_qupath_fold04_hne_on/oof_summary
```

彙整結果：

```text
results_optimization_comparison/
├── ablation/
├── ablation_best_reference/
├── stage_selection/
├── stages/
└── til_estimators/
```

完整數值來源為各 OOF 目錄下的 `eval_summary.json`、`slide_til_score_summary.csv` 與 `slide_til_calibration_summary.csv`。

## 14. 完成確認（2026-07-16）

| 項目 | 狀態 |
|---|---|
| A1 H&E on/off 5-fold | 完成（`results_groupcv_norm_heavy` vs `results_groupcv_nohne_heavy`） |
| A2 heavy vs medium | 完成（`results_groupcv_nohne_medium`） |
| A3 class weight on/off | 完成（`results_groupcv_nohne_heavy_noweight`） |
| B1 case-level error audit | 完成（`results_groupcv_nohne_heavy/error_audit_selected`） |
| B2 Stage 0/1/2 vs selected | 完成（`results_optimization_comparison/stages`） |
| B3 hard/soft TIL + calibration | 完成（`results_optimization_comparison/til_estimators`） |
| old_base fold04 case-level OOF | 完成（`results_old_base_qupath_fold04_hne_on` / `_hne_off`） |
| Unit tests | 17/17 通過（後續方法學測試另計） |
| 候選設定 | H&E off + heavy + class weight on + validation-selected |
