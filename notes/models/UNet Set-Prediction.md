---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - UNetSetPred
family: unet
registry_key: unet_setpred
summary: UNet backbone with set-prediction heads; per-slot parameter heads plus an existence-logit gate that blends amplitude toward a learned per-slot off level.
---

# UNet Set-Prediction

`UNetSetPred` (`models/backbone/unet.py`) attaches set-prediction heads to the plain [[UNet]] backbone: $K$ per-slot `PixelMLP` parameter heads (as in [[UNet Per-Gaussian]]) plus an existence-logit `PixelMLP` whose sigmoid gate blends each slot's amplitude channel toward a learned per-slot off level. Together with the `hungarian` `param_matching` mode of the loss (`tools/loss/param_loss.py`), this realises DETR-style set prediction — permutation-invariant slot assignment with an explicit per-slot existence mechanism — while keeping the standard $3K$-channel output contract, so training, normalization, clamping, and inference run unchanged.

---

## Summary

The decoder embedding feeds $K$ per-slot heads emitting (amp, $\mu$, $\sigma$) each, and one existence head emitting $K$ logits. The gate $g_k = \sigma(\ell_k) \in (0, 1)$ mixes the raw amplitude with a learnable scalar off level $o_k$: an open gate passes the regressed amplitude, a closed gate pins the slot's amplitude to $o_k$, which training drives toward the normalised encoding of physical zero. $\mu$ and $\sigma$ pass through ungated — the loss already masks them for inactive ground-truth slots, so the gate only needs to control presence.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{e}$ | Decoder embedding (channels `features[0]`) |
| $a_k, \mu_k, \sigma_k$ | Raw slot-$k$ parameter maps from head $k$ (normalised space) |
| $\ell_k$ | Existence logit map for slot $k$ |
| $g_k$ | Existence gate, $g_k = \sigma(\ell_k)$ |
| $o_k$ | Learned scalar off level of slot $k$ (initialised 0) |
| $\hat{a}_k$ | Gated amplitude output of slot $k$ |
| $K$ | Number of Gaussian slots, $K = C_{\text{out}} / \text{params\_per\_gaussian}$ |

---

## Architecture

$$
\begin{aligned}
(a_k, \mu_k, \sigma_k) &= \text{PixelMLP}_k(\mathbf{e}), \qquad k = 1, \dots, K \\
\ell_k &= \text{PixelMLP}_{\text{exist}}(\mathbf{e})_k \\
g_k &= \sigma(\ell_k) \\
\hat{a}_k &= g_k \, a_k + (1 - g_k) \, o_k
\end{aligned}
$$

The output tensor interleaves $(\hat{a}_k, \mu_k, \sigma_k)$ into the standard $3K$ channel layout. Because the gate operates in the normalised output space, "off" cannot be the constant zero (normalised zero is not physical zero); the learnable $o_k$ lets the network place the off level exactly at the normalised encoding of physical amplitude zero, where the existing `amp_zero_thr` convention recognises the slot as inactive.

---

## Design Rationale

**An explicit mechanism against slot collapse.** The documented Gaussian-slot-collapse pathology — higher slots permanently zeroed as a loss optimum under ground-truth imbalance — arises because a slot's only way to be "absent" is to regress its amplitude toward zero through the same feature pathway that also regresses its magnitude when present. The gate decouples the two: the existence head makes the on/off decision, the parameter head keeps regressing meaningful (amp, $\mu$, $\sigma$) values even while the slot is off, so a collapsed slot remains recoverable instead of being trapped in the zero regime.

**Set prediction under the existing contract.** DETR-style set prediction has two ingredients: bipartite matching and per-element existence. Matching already exists in the loss (`param_matching: hungarian`, the default — exact per-pixel permutation optimisation over $G \le 6$ slots). This head supplies the second ingredient without changing the output shape; the full BCE-supervised variant (existence logits as separate loss targets) would require widening the output contract to $4K$ channels across normalization, clamping, and inference, and is deliberately left as future work.

**Pair with `hungarian` matching.** Under `sorted_gt` matching the gate merely relabels slots; the mechanism is designed for the permutation-invariant regime.

---

## Parameter Reference

See [[Configuration Layer]] → `UNetSetPredConfig` (identical to `UNetPerGaussianConfig`; the heads parameter group additionally covers the existence head and the off levels).

---

## Provenance

Adapted from the set-prediction decoding of DETR (Carion et al., *End-to-End Object Detection with Transformers*, ECCV 2020, arXiv:2005.12872): bipartite matching + per-element existence, transplanted onto dense per-pixel Gaussian slots. The gating formulation replaces the classification softmax because the output contract encodes absence as amplitude zero rather than as a class. Behavioural contract verified by `tests/models_backbone/test_set_prediction_heads.py` (closed gate pins amplitude to the off level, open gate passes raw amplitude, $\mu$/$\sigma$ untouched, gradients reach the existence head).

---

## Related Notes

- [[ResUNet Set-Prediction]] — Same heads on the residual backbone
- [[UNet Per-Gaussian]] — The ungated per-slot ancestor
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — UNetSetPredConfig
- [[DLR-TomoSAR Index]] — Master index
