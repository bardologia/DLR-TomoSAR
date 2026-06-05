# UNet Per-Gaussian

`UNetPerGaussian` (`models/UNet_pergaussian.py`) extends the standard [[UNet]] backbone with one independent pixel-wise MLP output head per Gaussian component, each predicting all parameters $(a_k, \mu_k, \sigma_k)$ for its assigned slot.

---

## Summary

The encoder, bottleneck, and decoder are identical to [[UNet]]. The shared embedding from the decoder is fed to $K$ independent `PixelMLP` heads, one per Gaussian slot. Each head independently predicts all `params_per_gaussian` parameters for its slot.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{e}$ | Shared decoder embedding, $\mathbf{e} \in \mathbb{R}^{B \times F_1 \times P_H \times P_W}$ |
| $\hat{\boldsymbol{\theta}}_k$ | Predicted parameter tensor for Gaussian slot $k$, $\in \mathbb{R}^{B \times P_{\text{ppg}} \times P_H \times P_W}$ |
| $\hat{\boldsymbol{\theta}}$ | Full interleaved parameter output, $\in \mathbb{R}^{B \times K P_{\text{ppg}} \times P_H \times P_W}$ |
| $B, P_H, P_W$ | Batch size, patch height, patch width |
| $F_1$ | Embedding channel count (`features[0]`; default `64`) |
| $K$ | Number of Gaussian components |
| $P_{\text{ppg}}$ | Parameters per Gaussian (`params_per_gaussian`; default `3`) |
| $\text{PixelMLP}_k$ | Pixel-wise MLP head dedicated to slot $k$ ($\text{Conv}_{1\times1} \to \text{Act} \to \text{Conv}_{1\times1}$) |

---

## Architecture

### Shared Backbone

Identical to [[UNet]]. The decoder output embedding is $\mathbf{e} \in \mathbb{R}^{B \times F_1 \times P_H \times P_W}$.

### Per-Gaussian Heads

For slot $k = 1, \dots, K$:

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

**Review date:** 2026-06-04.

**Backbone identity.** The encoder, bottleneck, and decoder of `UNetPerGaussian` are structurally identical to [[UNet]]. The module imports the *same* `Encoder`, `Decoder`, and `ConvBlock` symbols directly from `models/UNet.py` (`from .UNet import ConvBlock, Encoder, Decoder`), so the contracting path (`ConvBlock` $\to$ `MaxPool2d(2)` per level, skips stored before pooling), the bottleneck (`ConvBlock`, $F_{-1} \to F_{-1}\cdot\texttt{bottleneck\_factor}$), and the expanding path (`build_upsample` $\to$ `match_spatial_size` $\to$ `torch.cat([skip, x], dim=1)` $\to$ `ConvBlock`) are byte-for-byte the same code. The `forward` sequence matches [[UNet]]: `encoder` $\to$ `bottleneck` $\to$ `decoder(x, skip_connections[::-1])`. The only structural divergence from [[UNet]] is at the output stage, where the single `output_head = Conv2d(F_1, out_channels, 1)` is replaced by $K$ `PixelMLP` heads (documented above).

**Config field comparison.** All structural fields of `UNetPerGaussianConfig` match `UNetConfig` field-for-field: `in_channels`, `out_channels`, `params_per_gaussian`, `features`, `bottleneck_factor`, `dropout`, `activation`, `normalization`, `upsample_mode`, `conv_bias`, `init_mode` share the same defaults. `UNetPerGaussianConfig` is itself identical to `UNetMultiHeadConfig` (same fields, defaults, and `heads_lr` / `heads_wd` grouping). The only non-structural difference vs `UNetConfig` is the head LR/WD naming (`heads_*` vs `output_head_*`). No structural divergence found.

**Head factorisation.** $K$ `PixelMLP` heads in a `nn.ModuleList` (`gaussian_heads`), one per Gaussian slot, each mapping the decoder embedding $\mathbf{e} \in \mathbb{R}^{B \times F_1 \times P_H \times P_W}$ to `params_per_gaussian` channels $(a_k, \mu_k, \sigma_k)$. Each `PixelMLP` is $\text{Conv}_{1\times1} \to \text{Act} \to \text{Conv}_{1\times1}$ with bias on both convolutions; this is the *same class* defined in `models/UNet_multihead.py` and imported here (`from .UNet_multihead import PixelMLP`), so head composition is provably identical to [[UNet Multihead]]. The factorisation differs only in axis: per-Gaussian shares one MLP per slot across parameter types; multihead shares one MLP per parameter type across slots.

**Output-layout equivalence.** `out = torch.stack(head_outputs, dim=1)` yields $(B, K, n_{\text{params}}, P_H, P_W)$, and `out.view(B, K\cdot n_{\text{params}}, P_H, P_W)` flattens with $K$ as the outer index and the parameter type as the inner index, giving channel order $[a_0, \mu_0, \sigma_0, a_1, \mu_1, \sigma_1, \dots]$ for $n_{\text{params}}=3$. [[UNet Multihead]] uses `torch.stack([amp, mu, sigma], dim=2)` $\to (B, K, 3, P_H, P_W)$ then the same view, producing the *identical* ordering. The two layouts are equivalent. This matches the training loss contract: `ParametricLoss.reconstruct_gaussians` (`pipelines/training_pipeline/loss.py:61-71`) reshapes $(B, C, H, W) \to (B, n_{\text{gaussians}}, \texttt{ppg}, H, W)$ and reads index $0=a$, $1=\mu$, $2=\sigma$ per slot — exactly the interleaving produced here.

**Verdict.** BACKBONE IDENTICAL. No structural divergences.

---

## Related Notes

- [[UNet]] — Shared backbone
- [[UNet Multihead]] — Alternative per-parameter-type head design
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — UNetPerGaussianConfig
- [[DLR-TomoSAR Index]] — Master index
