# HNSCC Methodology Comparison（摘要）

**完整、分主題、可持續更新的分數比較請見：[`hnscc_living_scoreboard.md`](hnscc_living_scoreboard.md)**

該文件涵蓋 GroupCV 報告（A1–A3、old_base、stage）與方法學優化（loss、sampler、aug ablation、source mix、threshold），並以目前候選 **Source mix 0.50:0.50**（AUC 0.8848／PRC 0.4196）計算 Δ 與 keep/drop。

## 下一步（A 線 + B 線，已實作腳本）

| 主題 | 文件／腳本 |
|---|---|
| External lock-box 報告 | [`hnscc_external_lockbox_report.md`](hnscc_external_lockbox_report.md) |
| Seed stability 報告 | [`hnscc_candidate_stability_report.md`](hnscc_candidate_stability_report.md) |
| 完整路線說明 | [`hnscc_methodology_optimization.md` §6](hnscc_methodology_optimization.md) |
| Backbone vs candidate | `scripts/compare_backbone_and_candidate.py` |

## Source mix 比例（相對 0.75:0.25）

| 設定 | Positive AUC | Positive PRC | 決策 |
|---|---:|---:|---|
| **0.50 : 0.50** | **0.8848** | **0.4196** | **目前候選** |
| 0.25 : 0.75 | 0.8809 | 0.4503 | keep（最高 PRC） |
| 0.75 : 0.25 | 0.8655 | 0.3998 | 比例 ablation reference |
| 0.90 : 0.10 | 0.8606 | 0.3829 | drop |

## Seed stability（0.50:0.50）

| seed | Positive AUC | Positive PRC |
|---:|---:|---:|
| 42 | 0.8848 | 0.4196 |
| 7 | 0.8604 | 0.3905 |
| 21 | 0.8685 | 0.4056 |
| mean±std | 0.8712±0.0101 | 0.4052±0.0119 |

## L2-SP（0.50:0.50）

| λ | Positive AUC | Positive PRC | 決策 |
|---:|---:|---:|---|
| 1e-5 | 0.8319 | 0.3575 | drop |
| 1e-4 | 0.8283 | 0.3540 | drop |
| 1e-3 | 0.8628 | 0.4095 | drop |

## Backbone smoke（fold 0+1）

| backbone | smoke AUC | smoke PRC | 備註 |
|---|---:|---:|---|
| EfficientNetV2-S | 0.8983 | 0.5448 | 快篩有趨勢；需 5-fold 確認 |
| ConvNeXt-Tiny | 0.9004 | 0.5539 | 同上 |

## 再生指令（Docker TIL）

```bash
docker exec -w /workspace TIL python3 scripts/build_hnscc_scoreboard.py
```

CSV：`results/results_methodology_comparison_scoreboard/scoreboard.csv`  
登錄新實驗：編輯 `configs/scoreboard_experiments.yaml` 後重跑上述指令。
