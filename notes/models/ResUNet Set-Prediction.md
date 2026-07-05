---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - ResUNetSetPred
family: resunet
registry_key: resunet_setpred
summary: ResUNet backbone with set-prediction heads; per-slot parameter heads plus an existence-logit gate that blends amplitude toward a learned per-slot off level.
---

# ResUNet Set-Prediction

`ResUNetSetPred` (`models/backbone/resunet.py`) combines the stride-2 residual [[ResUNet]] backbone (`ResUNetBackbone` with the default `downsample="stride"`) with the set-prediction heads of [[UNet Set-Prediction]]: $K$ per-slot `PixelMLP` parameter heads plus an existence-logit head whose sigmoid gate blends each slot's amplitude toward a learned per-slot off level. It isolates the set-prediction-head question from the backbone question, exactly as [[ResUNet Multihead]] and [[ResUNet Per-Gaussian]] do for their head designs.

---

## Summary

The head mechanism, output contract, and rationale are identical to [[UNet Set-Prediction]] — see that note for the gating equations, symbols, and the slot-collapse argument. The output stays in the standard $3K$ channel layout, so training, normalization, clamping, and inference are unchanged, and the heads pair with the `hungarian` `param_matching` mode of the loss for permutation-invariant slot assignment.

---

## Design Rationale

Holding the residual encode-decode path fixed while swapping only the head structure completes the head-axis triple on the ResUNet backbone: single-head ([[ResUNet]]), parameter-type heads ([[ResUNet Multihead]]), per-slot heads ([[ResUNet Per-Gaussian]]), and gated set-prediction heads (this model). Comparing the four at matched capacity attributes any difference to the head's inductive bias about slot existence and parameter independence.

---

## Parameter Reference

See [[Configuration Layer]] → `ResUNetSetPredConfig` (identical to `ResUNetPerGaussianConfig`; the heads parameter group additionally covers the existence head and the off levels).

---

## Related Notes

- [[UNet Set-Prediction]] — Full mechanism description and equations
- [[ResUNet Per-Gaussian]] — The ungated per-slot ancestor on this backbone
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — ResUNetSetPredConfig
- [[DLR-TomoSAR Index]] — Master index
