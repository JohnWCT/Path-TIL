# HNSCC Methodology Comparison（摘要）

**完整、分主題、可持續更新的分數比較請見：[`hnscc_living_scoreboard.md`](hnscc_living_scoreboard.md)**

該文件涵蓋 GroupCV 報告（A1–A3、old_base、stage）與方法學優化（loss、sampler、aug ablation、source mix、threshold），並以目前候選 **Source mix 0.50:0.50**（AUC 0.8848／PRC 0.4196）計算 Δ 與 keep/drop。

## Source mix 比例（相對 0.75:0.25）

| 設定 | Positive AUC | Positive PRC | 決策 |
|---|---:|---:|---|
| **0.50 : 0.50** | **0.8848** | **0.4196** | **目前候選** |
| 0.25 : 0.75 | 0.8809 | 0.4503 | keep（最高 PRC） |
| 0.75 : 0.25 | 0.8655 | 0.3998 | 比例 ablation reference |
| 0.90 : 0.10 | 0.8606 | 0.3829 | drop |

## 再生指令（Docker TIL）

```bash
docker exec -w /workspace TIL python3 scripts/build_hnscc_scoreboard.py
```

CSV：`results/results_methodology_comparison_scoreboard/scoreboard.csv`  
登錄新實驗：編輯 `configs/scoreboard_experiments.yaml` 後重跑上述指令。
