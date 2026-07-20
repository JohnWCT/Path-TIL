# HNSCC External Lock-Box Report

> Report-only evaluation on `dataset/Testset`. Do **not** use these results for tuning.

## Cohorts

```text
dataset/Testset/CPTAC_LUAD
dataset/Testset/CPTAC_LUSC
dataset/Testset/RUMC-BRCA
```

## Commands

```bash
python scripts/eval_tcga_internal.py \
  --model-dir results/results_method_source_mix_tcga_r50_50 \
  --test-root dataset/test \
  --output-dir results/results_tcga_internal_r50_50 \
  --stage selected

python scripts/eval_external_testset.py \
  --model-dir results/results_method_source_mix_tcga_r50_50 \
  --testset-root dataset/Testset \
  --output-dir results/results_external_testset_r50_50 \
  --stage selected
```

## TCGA Internal (`dataset/test`)

| metric | value |
|---|---:|
| positive AUC | TBD |
| positive PRC | TBD |
| macro OVR AUC | TBD |
| weighted OVR AUC | TBD |

## External Summary

| dataset | n_patches | positive AUC | positive PRC |
|---|---:|---:|---:|
| CPTAC_LUAD | TBD | TBD | TBD |
| CPTAC_LUSC | TBD | TBD | TBD |
| RUMC-BRCA | TBD | TBD | TBD |

## Decision

- [ ] TCGA internal AUC/PRC acceptable vs reference training behavior
- [ ] External lock-box shows no obvious collapse
- [ ] Proceed to seed stability / L2-SP / backbone screening
