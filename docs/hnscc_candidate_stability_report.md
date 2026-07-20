# HNSCC Candidate Stability Report

> Template report for seed-stability checks on the locked `source_mix_tcga_r50_50` candidate.
> Fill after running seed 42 / 7 / 21 experiments and `scripts/summarize_candidate_stability.py`.

## Reference

```text
backbone: InceptionResNetV2
source mix: HNSCC:TCGA = 0.50:0.50
positive AUC = 0.8848
positive PRC = 0.4196
```

## Commands

```bash
python scripts/summarize_candidate_stability.py \
  --experiments \
    results/results_oof_with_prc/source_mix_tcga_r50_50 \
    results/results_oof_with_prc/source_mix_tcga_r50_50_seed7 \
    results/results_oof_with_prc/source_mix_tcga_r50_50_seed21 \
  --output results/results_candidate_stability_r50_50
```

## Results

| seed | positive AUC | positive PRC | macro OVR AUC | weighted OVR AUC |
|---:|---:|---:|---:|---:|
| 42 | TBD | TBD | TBD | TBD |
| 7 | TBD | TBD | TBD | TBD |
| 21 | TBD | TBD | TBD | TBD |

```text
mean_positive_auc: TBD
std_positive_auc: TBD
mean_positive_prc: TBD
std_positive_prc: TBD
```

## Decision

- [ ] Candidate stable enough to keep as reference
- [ ] Seed sensitivity requires reporting mean ± std in final manuscript
- [ ] One or more seeds failed keep criteria → investigate before backbone replacement
