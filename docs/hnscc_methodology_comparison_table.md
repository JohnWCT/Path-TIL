# HNSCC Methodology Comparison

| experiment_name | keep_or_drop | positive_auc | macro_ovr_auc | weighted_ovr_auc | accuracy | macro_f1 | hard_til_mae | soft_til_mae | delta_positive_auc | delta_hard_til_mae | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oof_stage_selected | reference | 0.8555 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | 0.0000 | 0.0000 | current candidate reference |
| validation_positive_auc | drop | 0.8495 | 0.8824 | 0.8946 | 0.7391 | 0.6623 | 0.1447 | 0.1647 | -0.0060 | 0.0019 | positive_auc_not_improved;hard_til_mae_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| validation_macro_ovr_auc | drop | 0.8555 | 0.8922 | 0.9056 | 0.7480 | 0.6728 | 0.1428 | 0.1627 | 0.0000 | 0.0000 | positive_auc_not_improved;hard_til_mae_not_improved |
| composite_positive_macro | drop | 0.8495 | 0.8824 | 0.8946 | 0.7391 | 0.6623 | 0.1447 | 0.1647 | -0.0060 | 0.0019 | positive_auc_not_improved;hard_til_mae_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| fixed_stage1 | drop | 0.8136 | 0.8597 | 0.8760 | 0.7125 | 0.6327 | 0.1590 | 0.1890 | -0.0419 | 0.0162 | positive_auc_not_improved;hard_til_mae_not_improved;macro_ovr_auc_decreased;weighted_ovr_auc_clearly_decreased |
| fixed_stage2 | drop | 0.8508 | 0.8891 | 0.9028 | 0.7351 | 0.6613 | 0.1693 | 0.1719 | -0.0047 | 0.0265 | positive_auc_not_improved;hard_til_mae_not_improved;macro_ovr_auc_decreased |
| results_method_threshold_til | drop | 0.8555 | 0.8922 | 0.9056 | 0.7289 | 0.6728 | 0.1566 | 0.1627 | 0.0000 | 0.0138 | positive_auc_not_improved;hard_til_mae_not_improved |
