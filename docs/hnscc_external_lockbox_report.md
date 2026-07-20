# HNSCC External Lock-Box Report

> Auto-updated by `scripts/update_ab_plan_reports.py` at 2026-07-20 12:22 UTC.
> External results are report-only and must not be used for tuning.

## Reference (HNSCC OOF candidate)

```text
positive AUC = 0.8848
positive PRC = 0.4196
```

## TCGA Internal (`dataset/test`)

| metric | value |
|---|---:|
| positive AUC | 0.9950 |
| positive PRC | 0.9952 |
| macro OVR AUC | 0.9968 |
| weighted OVR AUC | 0.9971 |

## External Summary

| dataset | n_patches | positive AUC | positive PRC |
|---|---:|---:|---:|
| CPTAC_LUAD | 3000 | 0.9886 | 0.8923 |
| CPTAC_LUSC | 3000 | 0.9904 | 0.9748 |
| RUMC-BRCA | 3000 | 0.9972 | 0.9629 |
