# UNet Multihead

`UNetMultiHead` (`models/UNet_multihead.py`) extends the standard [[UNet]] backbone with three independent pixel-wise MLP output heads — one for amplitude, one for mean elevation, and one for sigma — rather than a single shared 1×1 convolution.

---

## Summary

The encoder, bottleneck, and decoder are identical to [[UNet]]. The shared embedding from the decoder is fed to three parameter-type-specific `PixelMLP` heads that independently predict all $K$ values of their respective parameter type.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{e}$ | Shared decoder embedding, $\mathbf{e} \in \mathbb{R}^{B \times F_1 \times P_H \times P_W}$ |
| $\mathbf{u}_i$ | Intermediate activations within a head |
| $\mathbf{a}, \boldsymbol{\mu}, \boldsymbol{\sigma}$ | Amplitude, mean, and sigma outputs from their respective heads, each $\in \mathbb{R}^{B \times K \times P_H \times P_W}$ |
| $\hat{\boldsymbol{\theta}}$ | Interleaved parameter output, $\in \mathbb{R}^{B \times 3K \times P_H \times P_W}$ |
| $B, P_H, P_W$ | Batch size, patch height, patch width |
| $F_1$ | Embedding channel count (`features[0]`; default `64`) |
| $H$ | Hidden channel count, $H = \max(\lfloor F_1 / 2 \rfloor, 16)$ |
| $K$ | Number of Gaussian components, $K = \texttt{out\_channels} / \texttt{params\_per\_gaussian}$ (default `6/3 = 2`) |
| $\text{Conv}_{1\times1}$ | Pointwise (1×1) convolution with bias |
| $\text{Act}$ | Activation function (`activation`; default `"relu"`) |
| $\text{PixelMLP}$ | Pixel-wise MLP head ($\text{Conv}_{1\times1} \to \text{Act} \to \text{Conv}_{1\times1}$), output shape $(B, K, P_H, P_W)$ |

---

## Architecture

### Shared Backbone

The UNet encoder-decoder produces an embedding $\mathbf{e} \in \mathbb{R}^{B \times F_1 \times P_H \times P_W}$, where $F_1$ = `features[0]` (default: `64`).

### PixelMLP

Each head applies two 1×1 convolutions (which are pointwise MLPs):

$$
\begin{aligned}
\mathbf{u}_1 &= \text{Conv}_{1\times1,\, F_1 \to H}(\mathbf{e}) \\
\mathbf{u}_2 &= \text{Act}(\mathbf{u}_1) \\
\text{PixelMLP}(\mathbf{e}) &= \text{Conv}_{1\times1,\, H \to K}(\mathbf{u}_2)
\end{aligned}
$$

### Output Assembly

The three heads produce $\mathbf{a}, \boldsymbol{\mu}, \boldsymbol{\sigma} \in \mathbb{R}^{B \times K \times P_H \times P_W}$. These are stacked and reshaped to the interleaved parameter layout:

$$
\hat{\boldsymbol{\theta}} = \text{reshape}(\text{stack}[\mathbf{a}, \boldsymbol{\mu}, \boldsymbol{\sigma}], \; [B, 3K, P_H, P_W])
$$

The stacking order ensures the output layout is $[a_1, \mu_1, \sigma_1, a_2, \mu_2, \sigma_2, \dots]$.

---

## Design Rationale

**Why separate heads.** The amplitude, mean, and sigma parameters have different statistical properties and may require different inductive biases at the output. Separate heads allow each parameter type to have an independent non-linear transformation of the shared embedding. The shared encoder-decoder still captures joint spatial context.

> **Parameter-type specialisation**
> The multihead design shares one MLP per parameter type across all Gaussian slots, imposing parameter-type consistency as the dominant inductive bias.

**Versus per-Gaussian heads.** [[UNet Per-Gaussian]] uses one head per Gaussian slot rather than one head per parameter type. The multihead design shares the MLP for the same parameter type across all Gaussian slots, while the per-Gaussian design shares the MLP for the same slot across all parameter types. The appropriate choice depends on whether parameter-type consistency or slot-role consistency is the dominant inductive bias.

---

## Parameter Reference

See [[Configuration Layer]] → `UNetMultiHeadConfig`.

Additional parameter:

| Parameter | Default | Description |
|---|---|---|
| `params_per_gaussian` | `3` | Number of parameters per Gaussian component $(a, \mu, \sigma)$ |

All base UNet parameters apply.

---

## Backbone consistency

**Review date:** 2026-06-04.

**Backbone identity.** The encoder, bottleneck, and decoder of `UNetMultiHead` are structurally identical to [[UNet]]. Both import and instantiate the *same* `Encoder`, `Decoder`, `ConvBlock`, and `match_spatial_size` symbols directly from `models/UNet.py` (`from .UNet import ConvBlock, Encoder, Decoder, match_spatial_size`), so the contracting path (`ConvBlock` $\to$ `MaxPool2d(2)` per level, with skips stored before pooling), the bottleneck (`ConvBlock`, $F_{-1} \to F_{-1}\cdot\texttt{bottleneck\_factor}$), and the expanding path (`build_upsample` $\to$ `match_spatial_size` $\to$ `torch.cat([skip, x], dim=1)` $\to$ `ConvBlock`) are byte-for-byte the same code. The `forward` sequence is identical: `encoder` $\to$ `bottleneck` $\to$ `decoder(x, skip_connections[::-1])`. The only structural divergence from [[UNet]] is at the output stage, where the single `output_head = Conv2d(F_1, out_channels, 1)` is replaced by three `PixelMLP` heads (documented above).

**Config field comparison.** All structural fields of `UNetMultiHeadConfig` match `UNetConfig` field-for-field: `in_channels`, `out_channels`, `params_per_gaussian`, `features`, `bottleneck_factor`, `dropout`, `activation`, `normalization`, `upsample_mode`, `conv_bias`, `init_mode` all share the same defaults. The only differences are non-structural and follow from the head factorisation: the learning-rate / weight-decay group for the output is named `heads_lr` / `heads_wd` (vs `output_head_lr` / `output_head_wd`), and `get_param_groups` exposes the three heads as a single `heads` group. No structural divergence found.

**Head factorisation.** Three shared `PixelMLP` heads (`head_amp`, `head_mu`, `head_sigma`), each mapping the decoder embedding $\mathbf{e} \in \mathbb{R}^{B \times F_1 \times P_H \times P_W}$ to $K$ channels (one slot per Gaussian). Each `PixelMLP` is $\text{Conv}_{1\times1} \to \text{Act} \to \text{Conv}_{1\times1}$ with bias on both convolutions; this is the identical class shared with [[UNet Per-Gaussian]] (which imports `PixelMLP` from this module).

**Output-layout equivalence.** `out = torch.stack([amp, mu, sigma], dim=2)` yields $(B, K, 3, P_H, P_W)$, and `out.view(B, K\cdot3, P_H, P_W)` flattens with $K$ as the outer index and the parameter type as the inner index, giving channel order $[a_0, \mu_0, \sigma_0, a_1, \mu_1, \sigma_1, \dots]$. [[UNet Per-Gaussian]] uses `torch.stack(head_outputs, dim=1)` $\to (B, K, n_{\text{params}}, P_H, P_W)$ then `view(B, K\cdot n_{\text{params}}, P_H, P_W)`, which produces the *same* $[a_0, \mu_0, \sigma_0, a_1, \dots]$ ordering for $n_{\text{params}}=3$. The two layouts are equivalent. This matches the training loss contract: `ParametricLoss.reconstruct_gaussians` (`pipelines/training_pipeline/loss.py:61-71`) reshapes $(B, C, H, W) \to (B, n_{\text{gaussians}}, \texttt{ppg}, H, W)$ and reads index $0=a$, $1=\mu$, $2=\sigma$ per slot, so any interleaving mismatch would silently corrupt the loss — none exists here.

**Verdict.** BACKBONE IDENTICAL. No structural divergences.

---

## Related Notes

- [[UNet]] — Shared backbone
- [[UNet Per-Gaussian]] — Alternative head design
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — UNetMultiHeadConfig
- [[DLR-TomoSAR Index]] — Master index
