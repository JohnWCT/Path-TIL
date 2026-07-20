# HNSCC Living Scoreboard

自動產生；請勿手改分數。再生指令見文末。

- 產生時間（UTC）：`2026-07-20T08:51:03.707076+00:00`
- 候選：`candidate_source_mix_tcga_r50_50`
- Positive AUC：**0.8848**
- Positive PRC：**0.4196**
- 主要判斷：positive AUC / PRC；hard/soft TIL MAE 僅參考。

敘事報告：[`hnscc_groupcv_optimization_report.md`](hnscc_groupcv_optimization_report.md)

## 1. 目前候選與全域排行（依 positive AUC 降序）

| experiment | theme | positive_auc | positive_prc | macro_ovr_auc | weighted_ovr_auc | accuracy | macro_f1 | hard_til_mae | soft_til_mae | delta_auc | delta_prc | vs_candidate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old_base fold04_stage2 (H&E on) | Baseline / old_base | 0.8912 | 0.4605 | 0.9341 | 0.9497 | 0.8350 | 0.6683 | 0.2113 | 0.1275 | 0.0064 | 0.0409 | keep | meets_criteria;positive_prc_backfilled |
| HNSCC:TCGA = 0.50:0.50（目前候選） | Source mix 混入比例 | 0.8848 | 0.4196 | 0.9173 | 0.9288 | 0.8174 | 0.6888 | 0.1781 | 0.1319 | 0.0000 | 0.0000 | current_candidate | locked candidate reference |
| HNSCC:TCGA = 0.25:0.75 | Source mix 混入比例 | 0.8809 | 0.4503 | 0.9164 | 0.9291 | 0.8254 | 0.7059 | 0.1788 | 0.1333 | -0.0039 | 0.0307 | drop | positive_auc_not_improved;macro_ovr_auc_decreased |
| HNSCC:TCGA = 0.75:0.25 | Source mix 混入比例 | 0.8655 | 0.3998 | 0.9078 | 0.9227 | 0.8105 | 0.6820 | 0.1678 | 0.1331 | -0.0192 | -0.0197 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased |
| HNSCC:TCGA = 0.90:0.10 | Source mix 混入比例 | 0.8606 | 0.3829 | 0.9061 | 0.9222 | 0.8068 | 0.6711 | 0.1596 | 0.1311 | -0.0242 | -0.0367 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased |
| H&E off / heavy / weight on | A1 H&E normalization | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | -0.0293 | -0.0379 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| H&E off / heavy / weight on | A2 Augmentation 強度（heavy vs medium） | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | -0.0293 | -0.0379 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| H&E off / heavy / weight on | A3 Class weight | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | -0.0293 | -0.0379 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| validation-selected（no-mix 候選） | B2 Stage 選擇策略 | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | -0.0293 | -0.0379 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| GroupCV candidate (H&E off / heavy / weight on) | Baseline / old_base | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | -0.0293 | -0.0379 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| validation macro OVR AUC | B2 Stage 選擇策略 | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | -0.0293 | -0.0379 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| validation multiclass AUC（selected） | B2 Stage 選擇策略 | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | -0.0293 | -0.0379 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| validation-tuned TIL thresholds | Threshold TIL（參考） | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7289 | 0.6728 | 0.1566 | 0.1627 | -0.0293 | -0.0379 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| fixed Stage 2 | B2 Stage 選擇策略 | 0.8508 | 0.3655 | 0.8891 | 0.9028 | 0.7351 | 0.6613 | 0.1693 | 0.1719 | -0.0340 | -0.0540 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| composite positive macro | B2 Stage 選擇策略 | 0.8495 | 0.3762 | 0.8824 | 0.8946 | 0.7391 | 0.6623 | 0.1447 | 0.1647 | -0.0353 | -0.0434 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| validation positive AUC | B2 Stage 選擇策略 | 0.8495 | 0.3762 | 0.8824 | 0.8946 | 0.7391 | 0.6623 | 0.1447 | 0.1647 | -0.0353 | -0.0434 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| focal loss γ=1 | Loss / sampler | 0.8463 | 0.3279 | 0.8752 | 0.8850 | 0.7728 | 0.6317 | 0.1647 | 0.1252 | -0.0385 | -0.0917 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| logit-adjusted CE | Loss / sampler | 0.8338 | 0.3380 | 0.8761 | 0.8907 | 0.7769 | 0.6322 | 0.1676 | 0.1388 | -0.0510 | -0.0816 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| without geometric | Heavy-aug component ablation | 0.8317 | 0.3080 | 0.8704 | 0.8837 | 0.7684 | 0.6307 | 0.1588 | 0.1310 | -0.0531 | -0.1116 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| old_base fold04_stage2 (H&E off) | Baseline / old_base | 0.8307 | 0.3636 | 0.8986 | 0.9228 | 0.8108 | 0.6071 | 0.2399 | 0.1436 | -0.0540 | -0.0560 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased |
| without cutout | Heavy-aug component ablation | 0.8302 | 0.3419 | 0.8721 | 0.8867 | 0.7731 | 0.6422 | 0.1522 | 0.1336 | -0.0546 | -0.0777 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| focal loss γ=2 | Loss / sampler | 0.8271 | 0.3463 | 0.8790 | 0.8974 | 0.7779 | 0.6348 | 0.1717 | 0.1276 | -0.0577 | -0.0733 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| H&E off / heavy / weight off | A3 Class weight | 0.8226 | 0.3419 | 0.8736 | 0.8915 | 0.7742 | 0.6272 | 0.1769 | 0.1282 | -0.0622 | -0.0777 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| balanced sampler | Loss / sampler | 0.8210 | 0.3318 | 0.8658 | 0.8816 | 0.6897 | 0.6249 | 0.1886 | 0.1716 | -0.0638 | -0.0878 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| H&E on / heavy / weight on | A1 H&E normalization | 0.8199 | 0.3644 | 0.8477 | 0.8581 | 0.7152 | 0.6382 | 0.1619 | 0.1806 | -0.0649 | -0.0552 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| GroupCV baseline (H&E on / heavy / weight on) | Baseline / old_base | 0.8199 | 0.3644 | 0.8477 | 0.8581 | 0.7152 | 0.6382 | 0.1619 | 0.1806 | -0.0649 | -0.0552 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| without blur/noise | Heavy-aug component ablation | 0.8175 | 0.3409 | 0.8758 | 0.8964 | 0.7839 | 0.6574 | 0.1849 | 0.1483 | -0.0673 | -0.0787 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| without HED | Heavy-aug component ablation | 0.8174 | 0.3225 | 0.8727 | 0.8925 | 0.7820 | 0.6405 | 0.1997 | 0.1691 | -0.0673 | -0.0971 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| fixed Stage 1 | B2 Stage 選擇策略 | 0.8136 | 0.2846 | 0.8597 | 0.8760 | 0.7125 | 0.6327 | 0.1590 | 0.1890 | -0.0711 | -0.1349 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| H&E off / medium / weight on | A2 Augmentation 強度（heavy vs medium） | 0.8020 | 0.3132 | 0.8649 | 0.8886 | 0.7380 | 0.6535 | 0.1834 | 0.1986 | -0.0828 | -0.1064 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| GroupCV Stage 0 pretrained (H&E on) | Baseline / old_base | 0.7815 | 0.2036 | 0.8107 | 0.8140 | 0.6054 | 0.5593 | 0.2708 | 0.2700 | -0.1033 | -0.2160 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |


## 2. Baseline / old_base

| experiment | theme | positive_auc | positive_prc | macro_ovr_auc | weighted_ovr_auc | accuracy | macro_f1 | hard_til_mae | soft_til_mae | delta_auc | delta_prc | vs_candidate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old_base fold04_stage2 (H&E on) | Baseline / old_base | 0.8912 | 0.4605 | 0.9341 | 0.9497 | 0.8350 | 0.6683 | 0.2113 | 0.1275 | 0.0064 | 0.0409 | keep | meets_criteria;positive_prc_backfilled |
| old_base fold04_stage2 (H&E off) | Baseline / old_base | 0.8307 | 0.3636 | 0.8986 | 0.9228 | 0.8108 | 0.6071 | 0.2399 | 0.1436 | -0.0540 | -0.0560 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased |
| GroupCV baseline (H&E on / heavy / weight on) | Baseline / old_base | 0.8199 | 0.3644 | 0.8477 | 0.8581 | 0.7152 | 0.6382 | 0.1619 | 0.1806 | -0.0649 | -0.0552 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| GroupCV Stage 0 pretrained (H&E on) | Baseline / old_base | 0.7815 | 0.2036 | 0.8107 | 0.8140 | 0.6054 | 0.5593 | 0.2708 | 0.2700 | -0.1033 | -0.2160 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| GroupCV candidate (H&E off / heavy / weight on) | Baseline / old_base | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | -0.0293 | -0.0379 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |

本章最佳（positive AUC）：**old_base fold04_stage2 (H&E on)** = 0.8912（PRC 0.4605）


## 3. A1 H&E normalization

| experiment | theme | positive_auc | positive_prc | macro_ovr_auc | weighted_ovr_auc | accuracy | macro_f1 | hard_til_mae | soft_til_mae | delta_auc | delta_prc | vs_candidate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H&E on / heavy / weight on | A1 H&E normalization | 0.8199 | 0.3644 | 0.8477 | 0.8581 | 0.7152 | 0.6382 | 0.1619 | 0.1806 | -0.0649 | -0.0552 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| H&E off / heavy / weight on | A1 H&E normalization | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | -0.0293 | -0.0379 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |

本章最佳（positive AUC）：**H&E off / heavy / weight on** = 0.8555（PRC 0.3817）


## 4. A2 Augmentation 強度（heavy vs medium）

| experiment | theme | positive_auc | positive_prc | macro_ovr_auc | weighted_ovr_auc | accuracy | macro_f1 | hard_til_mae | soft_til_mae | delta_auc | delta_prc | vs_candidate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H&E off / heavy / weight on | A2 Augmentation 強度（heavy vs medium） | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | -0.0293 | -0.0379 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| H&E off / medium / weight on | A2 Augmentation 強度（heavy vs medium） | 0.8020 | 0.3132 | 0.8649 | 0.8886 | 0.7380 | 0.6535 | 0.1834 | 0.1986 | -0.0828 | -0.1064 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |

本章最佳（positive AUC）：**H&E off / heavy / weight on** = 0.8555（PRC 0.3817）


## 5. A3 Class weight

| experiment | theme | positive_auc | positive_prc | macro_ovr_auc | weighted_ovr_auc | accuracy | macro_f1 | hard_til_mae | soft_til_mae | delta_auc | delta_prc | vs_candidate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H&E off / heavy / weight on | A3 Class weight | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | -0.0293 | -0.0379 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| H&E off / heavy / weight off | A3 Class weight | 0.8226 | 0.3419 | 0.8736 | 0.8915 | 0.7742 | 0.6272 | 0.1769 | 0.1282 | -0.0622 | -0.0777 | drop | positive_prc_backfilled_from_predictions;positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |

本章最佳（positive AUC）：**H&E off / heavy / weight on** = 0.8555（PRC 0.3817）


## 6. B2 Stage 選擇策略

| experiment | theme | positive_auc | positive_prc | macro_ovr_auc | weighted_ovr_auc | accuracy | macro_f1 | hard_til_mae | soft_til_mae | delta_auc | delta_prc | vs_candidate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation multiclass AUC（selected） | B2 Stage 選擇策略 | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | -0.0293 | -0.0379 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| validation positive AUC | B2 Stage 選擇策略 | 0.8495 | 0.3762 | 0.8824 | 0.8946 | 0.7391 | 0.6623 | 0.1447 | 0.1647 | -0.0353 | -0.0434 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| validation macro OVR AUC | B2 Stage 選擇策略 | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | -0.0293 | -0.0379 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| composite positive macro | B2 Stage 選擇策略 | 0.8495 | 0.3762 | 0.8824 | 0.8946 | 0.7391 | 0.6623 | 0.1447 | 0.1647 | -0.0353 | -0.0434 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| fixed Stage 1 | B2 Stage 選擇策略 | 0.8136 | 0.2846 | 0.8597 | 0.8760 | 0.7125 | 0.6327 | 0.1590 | 0.1890 | -0.0711 | -0.1349 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| fixed Stage 2 | B2 Stage 選擇策略 | 0.8508 | 0.3655 | 0.8891 | 0.9028 | 0.7351 | 0.6613 | 0.1693 | 0.1719 | -0.0340 | -0.0540 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| validation-selected（no-mix 候選） | B2 Stage 選擇策略 | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | -0.0293 | -0.0379 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |

本章最佳（positive AUC）：**validation multiclass AUC（selected）** = 0.8555（PRC 0.3817）


## 7. Loss / sampler

| experiment | theme | positive_auc | positive_prc | macro_ovr_auc | weighted_ovr_auc | accuracy | macro_f1 | hard_til_mae | soft_til_mae | delta_auc | delta_prc | vs_candidate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| focal loss γ=1 | Loss / sampler | 0.8463 | 0.3279 | 0.8752 | 0.8850 | 0.7728 | 0.6317 | 0.1647 | 0.1252 | -0.0385 | -0.0917 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| focal loss γ=2 | Loss / sampler | 0.8271 | 0.3463 | 0.8790 | 0.8974 | 0.7779 | 0.6348 | 0.1717 | 0.1276 | -0.0577 | -0.0733 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| logit-adjusted CE | Loss / sampler | 0.8338 | 0.3380 | 0.8761 | 0.8907 | 0.7769 | 0.6322 | 0.1676 | 0.1388 | -0.0510 | -0.0816 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| balanced sampler | Loss / sampler | 0.8210 | 0.3318 | 0.8658 | 0.8816 | 0.6897 | 0.6249 | 0.1886 | 0.1716 | -0.0638 | -0.0878 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |

本章最佳（positive AUC）：**focal loss γ=1** = 0.8463（PRC 0.3279）


## 8. Heavy-aug component ablation

| experiment | theme | positive_auc | positive_prc | macro_ovr_auc | weighted_ovr_auc | accuracy | macro_f1 | hard_til_mae | soft_til_mae | delta_auc | delta_prc | vs_candidate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| without geometric | Heavy-aug component ablation | 0.8317 | 0.3080 | 0.8704 | 0.8837 | 0.7684 | 0.6307 | 0.1588 | 0.1310 | -0.0531 | -0.1116 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| without HED | Heavy-aug component ablation | 0.8174 | 0.3225 | 0.8727 | 0.8925 | 0.7820 | 0.6405 | 0.1997 | 0.1691 | -0.0673 | -0.0971 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| without blur/noise | Heavy-aug component ablation | 0.8175 | 0.3409 | 0.8758 | 0.8964 | 0.7839 | 0.6574 | 0.1849 | 0.1483 | -0.0673 | -0.0787 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| without cutout | Heavy-aug component ablation | 0.8302 | 0.3419 | 0.8721 | 0.8867 | 0.7731 | 0.6422 | 0.1522 | 0.1336 | -0.0546 | -0.0777 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |

本章最佳（positive AUC）：**without geometric** = 0.8317（PRC 0.3080）


## 9. Source mix 混入比例

| experiment | theme | positive_auc | positive_prc | macro_ovr_auc | weighted_ovr_auc | accuracy | macro_f1 | hard_til_mae | soft_til_mae | delta_auc | delta_prc | vs_candidate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HNSCC:TCGA = 0.90:0.10 | Source mix 混入比例 | 0.8606 | 0.3829 | 0.9061 | 0.9222 | 0.8068 | 0.6711 | 0.1596 | 0.1311 | -0.0242 | -0.0367 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased |
| HNSCC:TCGA = 0.75:0.25 | Source mix 混入比例 | 0.8655 | 0.3998 | 0.9078 | 0.9227 | 0.8105 | 0.6820 | 0.1678 | 0.1331 | -0.0192 | -0.0197 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased |
| HNSCC:TCGA = 0.50:0.50（目前候選） | Source mix 混入比例 | 0.8848 | 0.4196 | 0.9173 | 0.9288 | 0.8174 | 0.6888 | 0.1781 | 0.1319 | 0.0000 | 0.0000 | current_candidate | locked candidate reference |
| HNSCC:TCGA = 0.25:0.75 | Source mix 混入比例 | 0.8809 | 0.4503 | 0.9164 | 0.9291 | 0.8254 | 0.7059 | 0.1788 | 0.1333 | -0.0039 | 0.0307 | drop | positive_auc_not_improved;macro_ovr_auc_decreased |

本章最佳（positive AUC）：**HNSCC:TCGA = 0.50:0.50（目前候選）** = 0.8848（PRC 0.4196）


## 10. Threshold TIL（參考）

| experiment | theme | positive_auc | positive_prc | macro_ovr_auc | weighted_ovr_auc | accuracy | macro_f1 | hard_til_mae | soft_til_mae | delta_auc | delta_prc | vs_candidate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation-tuned TIL thresholds | Threshold TIL（參考） | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7289 | 0.6728 | 0.1566 | 0.1627 | -0.0293 | -0.0379 | drop | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |

本章最佳（positive AUC）：**validation-tuned TIL thresholds** = 0.8555（PRC 0.3817）


## 附錄

### 缺漏實驗（路徑不存在）

- （無）

### 再生指令（Docker TIL）

```bash
docker exec -w /workspace TIL python3 scripts/build_hnscc_scoreboard.py \
  --registry configs/scoreboard_experiments.yaml \
  --results-root results \
  --output-md docs/hnscc_living_scoreboard.md \
  --output-csv results/results_methodology_comparison_scoreboard/scoreboard.csv
```

CSV：`results/results_methodology_comparison_scoreboard/scoreboard.csv`
