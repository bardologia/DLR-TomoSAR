---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - UNetPerGaussian
  - GaussianHeadsMixin
family: head-variant
registry_key: unet_pergaussian
summary: UNet backbone with one independent PixelMLP head per Gaussian slot, predicting all its parameters jointly.
---

# UNet Per-Gaussian

`UNetPerGaussian` (`models/backbone/unet.py`, registry key `"unet_pergaussian"`) extends the shared `UNetBackbone` with one independent pixel-wise MLP output head per Gaussian component, each predicting all parameters $(a_k, \mu_k, \sigma_k)$ for its assigned slot.

---

## Summary

The encoder, bottleneck, and decoder are identical to [[UNet]]. The input is `in_channels` channels (default `1`, set per [[Signal Representation]] to the per-pass channel count). The shared embedding from the decoder is fed to $K$ independent `PixelMLP` heads, one per Gaussian slot, held in `gaussian_heads` (`nn.ModuleList`). Each head independently predicts all `params_per_gaussian` parameters for its slot. `forward` is `self._per_gaussian_forward(self.encode_decode(x))`, both defined by `GaussianHeadsMixin`.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{e}$ | Shared decoder embedding, $\mathbf{e} \in \mathbb{R}^{B \times F_1 \times P_H \times P_W}$ |
| $\hat{\boldsymbol{\theta}}_k$ | Predicted parameter tensor for Gaussian slot $k$, $\in \mathbb{R}^{B \times P_{\text{ppg}} \times P_H \times P_W}$ |
| $\hat{\boldsymbol{\theta}}$ | Full interleaved parameter output, $\in \mathbb{R}^{B \times K P_{\text{ppg}} \times P_H \times P_W}$ |
| $B, P_H, P_W$ | Batch size, patch height, patch width |
| $F_1$ | Embedding channel count (`embedding_channels = features[0]`; default `64`) |
| $H$ | Head hidden width (`hidden_channels = max(F_1 // 2, 16)`) |
| $K$ | Number of Gaussian components ($K = \texttt{out\_channels} / \texttt{params\_per\_gaussian}$; default $6/3 = 2$) |
| $P_{\text{ppg}}$ | Parameters per Gaussian (`params_per_gaussian`; default `3`) |
| $\text{PixelMLP}_k$ | Pixel-wise MLP head dedicated to slot $k$ ($\text{Conv}_{1\times1}(F_1 \to H) \to \text{Act} \to \text{Conv}_{1\times1}(H \to P_{\text{ppg}})$, bias on both convs) |

---

## Architecture

### Shared Backbone

Identical to [[UNet]]. The decoder output embedding is $\mathbf{e} \in \mathbb{R}^{B \times F_1 \times P_H \times P_W}$.

### Per-Gaussian Heads

For slot $k = 1, \dots, K$, each $\text{PixelMLP}_k$ maps $F_1 \to H \to P_{\text{ppg}}$ channels via two $1\times1$ convolutions with an activation between:

$$
\hat{\boldsymbol{\theta}}_k = \text{PixelMLP}_k(\mathbf{e}) \in \mathbb{R}^{B \times P_{\text{ppg}} \times P_H \times P_W}
$$

### Output Assembly

Head outputs are stacked along the Gaussian dimension and reshaped to the interleaved layout:

$$
\begin{aligned}
\mathbf{S} &= \text{stack}_{k=1}^K [\hat{\boldsymbol{\theta}}_k] \in \mathbb{R}^{B \times K \times P_{\text{ppg}} \times P_H \times P_W} \\
\hat{\boldsymbol{\theta}} &= \text{reshape}(\mathbf{S}, \; [B, K \cdot P_{\text{ppg}}, P_H, P_W])
\end{aligned}
$$

---

## Design Rationale

**Why per-Gaussian heads.** Each head specialises in a single scattering component. This design encourages the model to learn stable, slot-specific representations: head $k$ always predicts the $k$-th component (by the sorted $\mu$ convention), which may be easier to learn than predicting all components from a single head. The inductive bias is that slot roles are consistent across the image.

> **Slot-role specialisation**
> Each head owns one Gaussian slot across all parameter types, imposing slot-role consistency as the dominant inductive bias.

**Versus multihead.** See [[UNet Multihead]] for the parameter-type-specific alternative. Per-Gaussian heads impose slot-role specialisation; multihead imposes parameter-type specialisation. Both are hypotheses; empirical comparison on this dataset is required to determine which is better.

---

## Parameter Reference

See [[Configuration Layer]] → `UNetPerGaussianConfig`.

Identical to [[UNet Multihead]] configuration.

---

## Backbone consistency

**Backbone identity.** The encoder, bottleneck, and decoder of `UNetPerGaussian` are structurally identical to [[UNet]]. Both subclass the shared `UNetBackbone` (`models/backbone/unet.py`, `class UNetBackbone(nn.Module, GaussianHeadsMixin)`), which instantiates the same `Encoder`, `Decoder`, and `ConvBlock` symbols from `models/blocks.py` (`from .blocks import ConvBlock, Decoder, Encoder, GaussianHeadsMixin`), so the contracting path (`ConvBlock` $\to$ `MaxPool2d(2)` per level, skips stored before pooling), the bottleneck (`ConvBlock`, $F_{-1} \to F_{-1}\cdot\texttt{bottleneck\_factor}$), and the expanding path (`build_upsample` $\to$ `match_spatial_size` $\to$ `torch.cat([skip, x], dim=1)` $\to$ `ConvBlock`) are the same code. The shared `encode_decode` is `encoder` $\to$ `bottleneck` $\to$ `decoder(x, skip_connections[::-1])`, matching [[UNet]]. The sole structural divergence from [[UNet]] is the output stage: the single `output_head = Conv2d(F_1, out_channels, 1)` is replaced by $K$ `PixelMLP` heads (documented above).

**Config fields.** All structural fields of `UNetPerGaussianConfig` match `UNetConfig` field-for-field: `in_channels` (`1`), `out_channels` (`6`), `params_per_gaussian` (`3`), `features` (`[64, 128, 256, 512]`), `bottleneck_factor` (`2`), `dropout` (`0.15`), `activation` (`"relu"`), `normalization` (`"batch"`), `upsample_mode` (`"convtranspose"`), `conv_bias` (`False`), `init_mode` (`"default"`) share the same defaults. `UNetPerGaussianConfig` is identical to `UNetMultiHeadConfig` (same fields, defaults, and `heads_lr` / `heads_wd` grouping; `get_param_groups` groups the heads as `model.gaussian_heads`). The only difference vs `UNetConfig` is the head LR/WD naming (`heads_*` vs `output_head_*`).

**Head factorisation.** `_build_per_gaussian_heads` (in `GaussianHeadsMixin`, `models/blocks.py`) constructs $K = n_{\text{gaussians}}$ `PixelMLP` heads in a `nn.ModuleList` (`gaussian_heads`), one per Gaussian slot, each mapping the decoder embedding $\mathbf{e} \in \mathbb{R}^{B \times F_1 \times P_H \times P_W}$ through $F_1 \to H \to P_{\text{ppg}}$ to `params_per_gaussian` channels $(a_k, \mu_k, \sigma_k)$, with $H = \texttt{max}(F_1 // 2, 16)$. Each `PixelMLP` is $\text{Conv}_{1\times1} \to \text{Act} \to \text{Conv}_{1\times1}$ with bias on both convolutions, the same `PixelMLP` class used by [[UNet Multihead]]. The factorisation differs only in axis: per-Gaussian shares one MLP per slot across parameter types; multihead shares one MLP per parameter type across slots.

**Output layout.** `_per_gaussian_forward` runs `out = torch.stack(head_outputs, dim=1)` to $(B, K, n_{\text{params}}, P_H, P_W)$, then `out.view(B, n_{\text{gaussians}} \cdot n_{\text{params}}, P_H, P_W)` flattens with $K$ as the outer index and parameter type as the inner index, giving channel order $[a_0, \mu_0, \sigma_0, a_1, \mu_1, \sigma_1, \dots]$ for $n_{\text{params}}=3$. [[UNet Multihead]] `_triple_head_forward` uses `torch.stack([amp, mu, sigma], dim=2)` $\to (B, K, 3, P_H, P_W)$ then the same view, producing the identical ordering. This matches the training loss contract: `Loss.reconstruct_gaussians` (`pipelines/backbone/training/loss.py`) reshapes $(B, C, H, W) \to (B, n_{\text{gaussians}}, \texttt{ppg}, H, W)$ and reads index $0=a$, $1=\mu$, $2=\sigma$ per slot — exactly the interleaving produced here.

---

## Related Notes

- [[UNet]] — Shared backbone
- [[UNet Multihead]] — Alternative per-parameter-type head design
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — UNetPerGaussianConfig
- [[DLR-TomoSAR Index]] — Master index
