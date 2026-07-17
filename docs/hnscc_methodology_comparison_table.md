# HNSCC Methodology Comparison

本表以舊 no-mix candidate 為比較參考。`source_mix_tcga` 為唯一 keep，已升為新候選（見 `docs/hnscc_methodology_optimization.md`、`path_til/experiment_registry.py`）。

| experiment_name | keep_or_drop | positive_auc | positive_prc | macro_ovr_auc | weighted_ovr_auc | accuracy | macro_f1 | hard_til_mae | soft_til_mae | delta_positive_auc | delta_positive_prc | delta_hard_til_mae | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_selected | reference | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | 0.0000 | 0.0000 | 0.0000 | previous no-mix candidate reference; decision uses AUC/PRC only; TIL MAE is diagnostic |
| focal_gamma1 | drop | 0.8463 | 0.3279 | 0.8752 | 0.8850 | 0.7728 | 0.6317 | 0.1647 | 0.1252 | -0.0092 | -0.0538 | 0.0219 | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| focal_gamma2 | drop | 0.8271 | 0.3463 | 0.8790 | 0.8974 | 0.7779 | 0.6348 | 0.1717 | 0.1276 | -0.0284 | -0.0354 | 0.0290 | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased |
| logit_adjusted_ce | drop | 0.8338 | 0.3380 | 0.8761 | 0.8907 | 0.7769 | 0.6322 | 0.1676 | 0.1388 | -0.0217 | -0.0438 | 0.0248 | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| balanced_sampler | drop | 0.8210 | 0.3318 | 0.8658 | 0.8816 | 0.6897 | 0.6249 | 0.1886 | 0.1716 | -0.0346 | -0.0499 | 0.0459 | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| source_mix_tcga | keep | 0.8655 | 0.3998 | 0.9078 | 0.9227 | 0.8105 | 0.6820 | 0.1678 | 0.1331 | 0.0100 | 0.0181 | 0.0250 | meets_criteria; promoted to current candidate |
| aug_without_geometric | drop | 0.8317 | 0.3080 | 0.8704 | 0.8837 | 0.7684 | 0.6307 | 0.1588 | 0.1310 | -0.0238 | -0.0738 | 0.0160 | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| aug_without_hed | drop | 0.8174 | 0.3225 | 0.8727 | 0.8925 | 0.7820 | 0.6405 | 0.1997 | 0.1691 | -0.0381 | -0.0592 | 0.0570 | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| aug_without_blur_noise | drop | 0.8175 | 0.3409 | 0.8758 | 0.8964 | 0.7839 | 0.6574 | 0.1849 | 0.1483 | -0.0380 | -0.0408 | 0.0421 | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased |
| aug_without_cutout | drop | 0.8302 | 0.3419 | 0.8721 | 0.8867 | 0.7731 | 0.6422 | 0.1522 | 0.1336 | -0.0253 | -0.0398 | 0.0095 | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| stage_fixed_stage2 | drop | 0.8508 | 0.3655 | 0.8891 | 0.9028 | 0.7351 | 0.6613 | 0.1693 | 0.1719 | -0.0047 | -0.0162 | 0.0265 | positive_auc_not_improved;positive_prc_not_improved;macro_ovr_auc_decreased |
| threshold_til | drop | 0.8555 | 0.3817 | 0.8922 | 0.9056 | 0.7289 | 0.6728 | 0.1566 | 0.1627 | 0.0000 | 0.0000 | 0.0138 | positive_auc_not_improved;positive_prc_not_improved |
