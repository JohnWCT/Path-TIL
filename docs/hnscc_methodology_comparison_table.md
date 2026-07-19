# HNSCC Methodology Comparison — Source Mix Ratio Ablation

比較參考：`source_mix_tcga`（HNSCC:TCGA = 0.75:0.25）。  
完整方法學表見同目錄其他文件；比例細節見 `docs/hnscc_methodology_optimization.md`。

**新候選：`source_mix_tcga_r50_50`（0.50:0.50）** — 最高 positive AUC，且 PRC 同時提升。  
Pareto 備選：`source_mix_tcga_r25_75`（0.25:0.75）— 最高 PRC。

| experiment_name | keep_or_drop | positive_auc | positive_prc | macro_ovr_auc | weighted_ovr_auc | accuracy | macro_f1 | hard_til_mae | soft_til_mae | delta_positive_auc | delta_positive_prc | delta_hard_til_mae | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| source_mix_tcga (0.75:0.25) | reference | 0.8655 | 0.3998 | 0.9078 | 0.9227 | 0.8105 | 0.6820 | 0.1678 | 0.1331 | 0.0000 | 0.0000 | 0.0000 | previous source-mix candidate; decision uses AUC/PRC only |
| source_mix_tcga_r90_10 | drop | 0.8606 | 0.3829 | 0.9061 | 0.9222 | 0.8068 | 0.6711 | 0.1596 | 0.1311 | -0.0050 | -0.0170 | -0.0081 | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased |
| source_mix_tcga_r50_50 | keep | 0.8848 | 0.4196 | 0.9173 | 0.9288 | 0.8174 | 0.6888 | 0.1781 | 0.1319 | 0.0192 | 0.0197 | 0.0104 | meets_criteria; promoted to current candidate (best AUC) |
| source_mix_tcga_r25_75 | keep | 0.8809 | 0.4503 | 0.9164 | 0.9291 | 0.8254 | 0.7059 | 0.1788 | 0.1333 | 0.0153 | 0.0505 | 0.0110 | meets_criteria; Pareto alternative (best PRC) |
