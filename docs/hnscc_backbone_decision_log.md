# HNSCC Backbone Decision Log

> Updated: 2026-07-22 03:44 UTC

## Locked candidate

- Backbone: InceptionResNetV2
- Source mix: 0.50:0.50
- OOF positive AUC / PRC: 0.8848 / 0.4196

## B4 smoke

- EfficientNetV2-S and ConvNeXt-Tiny: positive-specialist trend, pending B5/B6
- Not promoted: macro / weighted OVR decreased; smoke is not candidate-level evidence

## B5 / B6 / B7

- B5: repair macro/weighted OVR while keeping positive gains
- B6: at most one config per backbone for full 5-fold
- B7: external lock-box only after B6; report-only, never used for tuning
