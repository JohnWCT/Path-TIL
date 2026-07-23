# HNSCC Backbone Decision Log

> Updated: 2026-07-23

## Locked candidate（維持）

- Backbone: **InceptionResNetV2**
- Source mix: HNSCC:TCGA = **0.50:0.50**
- OOF positive AUC / PRC: **0.8848 / 0.4196**
- macro / weighted OVR: 0.9173 / 0.9288

## Pipeline outcome

| Phase | Result |
|---|---|
| B4 smoke | positive ↑, macro/weighted ↓ → specialist trend |
| B5 repair ×12 | 無人 `replace_candidate`；選出 2 組進 B6 |
| B6 full5 | EfficientNet：AUC 未過；ConvNeXt：macro 未過 |
| B7 external | 兩組 external 穩定；**不取代** |

## Final decision

```text
KEEP: IRV2 + source mix 0.50:0.50 as primary candidate
OPTIONAL: ConvNeXt-Tiny full5 as positive-specialist / Pareto reference
OPTIONAL: EfficientNetV2-S full5 as PRC / multiclass reference (AUC slightly lower)
DO NOT: replace primary candidate with either backbone
```

## Selected B5 → B6 configs

- EfficientNetV2-S: `b5_h4_more_tcga`
- ConvNeXt-Tiny: `b5_h6_low_lr`
