# HNSCC 方法學優化工作流程

## 1. 摘要

本輪依《Path-TIL 方法學優化 IDE 操作手冊》建立可重現的方法學實驗骨架，並完成可離線評估的項目。

候選參考（不取代，除非同時通過成功標準）：

```text
H&E off + heavy + class weight on + validation multiclass AUC
positive AUC = 0.8555
hard TIL MAE = 0.1428
```

成功標準：

```text
positive AUC > 0.8555
hard TIL MAE < 0.1428
macro OVR AUC 不下降
weighted OVR AUC 不明顯下降（< 0.01）
```

## 2. 新增結構

```text
configs/method_*.yaml
path_til/losses.py
path_til/samplers.py
path_til/stage_selection.py
path_til/til_threshold.py
path_til/experiment_registry.py
path_til/backbones/
scripts/train_hnscc_method.py
scripts/compare_hnscc_stage_policies.py
scripts/tune_hnscc_thresholds.py
scripts/ablate_heavy_augmentation.py
scripts/train_hnscc_source_mix.py
scripts/compare_hnscc_methodology.py
scripts/summarize_methodology_table.py
tests/test_losses.py
tests/test_samplers.py
tests/test_stage_selection.py
tests/test_til_threshold.py
tests/test_methodology_registry.py
```

## 3. 本輪已完成的離線結果

在既有 `results_groupcv_nohne_heavy` 預測上：

| 方法 | Positive AUC | Hard TIL MAE | 決策 |
|---|---:|---:|---|
| candidate / validation multiclass AUC | 0.8555 | 0.1428 | reference |
| validation positive AUC | 0.8495 | 0.1447 | drop |
| validation macro OVR AUC | 0.8555 | 0.1428 | drop（未嚴格優於） |
| composite 0.7 pos + 0.3 macro | 0.8495 | 0.1447 | drop |
| fixed Stage 1 | 0.8136 | 0.1590 | drop |
| fixed Stage 2 | 0.8508 | 0.1693 | drop |
| validation-tuned positive threshold TIL | 0.8555 | 0.1566 | drop |

解讀：

1. 目前 validation multiclass AUC 選擇仍優於其他 stage policy。
2. Threshold-based hard TIL 在 validation 上可壓低單 fold 誤差，但 OOF hard TIL MAE 變差，不建議取代 hard raw。
3. Loss / sampler / heavy-aug component / source-mix 需要完整重訓；骨架與 YAML 已就緒。

詳細表見 [`hnscc_methodology_comparison_table.md`](hnscc_methodology_comparison_table.md)。

## 4. 執行方式

### Loss / sampler 訓練

```bash
docker exec TIL bash -lc '
cd /workspace
python3 scripts/train_hnscc_method.py \
  --config configs/method_focal_loss_gamma1.yaml \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --pretrained best_InceptionResNetV2_model.h5 \
  --output-dir results_method_focal_gamma1
'
```

同樣可替換：

- `configs/method_focal_loss_gamma2.yaml`
- `configs/method_logit_adjusted_ce.yaml`
- `configs/method_balanced_sampler.yaml`

訓練後：

```bash
python3 scripts/eval_hnscc_oof.py \
  --pred-dir results_method_focal_gamma1 \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --stage selected \
  --output results_method_focal_gamma1/oof_summary
```

### Stage policy（離線）

```bash
python3 scripts/compare_hnscc_stage_policies.py \
  --pred-dir results_groupcv_nohne_heavy \
  --output results_method_stage_policy
```

### Threshold TIL（validation-only tuning）

```bash
python3 scripts/tune_hnscc_thresholds.py \
  --pred-dir results_groupcv_nohne_heavy \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --output results_method_threshold_til
```

### Source mix

需要先準備 `tcga_train_dataset.csv`。腳本會只把 TCGA patches 混入各 fold 的 train，不動 held-out HNSCC test。

### 統一比較

```bash
python3 scripts/compare_hnscc_methodology.py \
  --reference results_groupcv_nohne_heavy/oof_stage_selected \
  --experiments \
    results_method_stage_policy/fixed_stage2/oof_summary \
    results_method_threshold_til \
  --output results_methodology_comparison
```

## 5. 測試

容器內目前使用 `unittest`（未預裝 pytest）：

```bash
python3 -m unittest discover -s tests -v
```

本輪：33/33 通過。

## 6. 下一步建議順序

1. Focal gamma 1 / 2 完整 5-fold
2. Logit-adjusted CE
3. Class-balanced sampler
4. Heavy aug component ablation（需把 component flag 接到 ImgAugSequence）
5. Source mix（需 TCGA manifest）
6. L2-SP / backbone registry / EWC

只有通過成功標準才取代目前 candidate。
