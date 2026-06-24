# UNet Multihead

`UNetMultiHead` (`models/backbone/unet.py`, registry name `unet_multihead`, config `UNetMultiHeadConfig`) extends the shared `UNetBackbone` with three independent pixel-wise MLP output heads — one for amplitude, one for mean elevation, and one for sigma — rather than a single shared 1×1 convolution.

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

## Backbone and head construction

**Backbone.** The encoder, bottleneck, and decoder of `UNetMultiHead` are those of the shared `UNetBackbone` (`models/backbone/unet.py`), which `UNet`, `UNetMultiHead`, and `UNetPerGaussian` all subclass. `UNetBackbone` imports `from .blocks import ConvBlock, Decoder, Encoder, GaussianHeadsMixin` and inherits from `nn.Module, GaussianHeadsMixin`. The contracting path is `ConvBlock` $\to$ `MaxPool2d(2)` per level (skips stored before pooling), the bottleneck is a single `ConvBlock` ($F_{-1} \to F_{-1}\cdot\texttt{bottleneck\_factor}$), and the expanding path is `build_upsample` $\to$ `match_spatial_size` $\to$ `torch.cat([skip, x], dim=1)` $\to$ `ConvBlock`. `encode_decode` runs `encoder` $\to$ `bottleneck` $\to$ `decoder(x, skip_connections[::-1])`. `UNetBackbone` exposes `embedding_channels = features[0]` and `hidden_channels = max(features[0] // 2, 16)`. `UNetMultiHead` differs from [[UNet]] only at the output stage: [[UNet]] applies `output_head = Conv2d(F_1, out_channels, 1)`, whereas `UNetMultiHead` carries no `output_head` and instead builds three `PixelMLP` heads.

**Config fields.** `UNetMultiHeadConfig` carries the same structural fields and defaults as `UNetConfig`: `in_channels=1`, `out_channels=6`, `params_per_gaussian=3`, `features=[64, 128, 256, 512]`, `bottleneck_factor=2`, `dropout=0.15`, `activation="relu"`, `normalization="batch"`, `upsample_mode="convtranspose"`, `conv_bias=False`, `init_mode="default"`. The optimiser group for the output is `heads_lr` / `heads_wd` (defaults `1e-3` / `1e-4`), and `get_param_groups` exposes `head_amp`, `head_mu`, `head_sigma` as a single `heads` group.

**Head factorisation.** `_resolve_gaussian_layout` (from `GaussianHeadsMixin`) sets `n_gaussians = out_channels // params_per_gaussian` and requires `out_channels % params_per_gaussian == 0`. `_build_triple_heads` builds three `PixelMLP` heads (`head_amp`, `head_mu`, `head_sigma`), each `PixelMLP(embedding_channels, hidden_channels, n_gaussians, activation)`, mapping the decoder embedding $\mathbf{e} \in \mathbb{R}^{B \times F_1 \times P_H \times P_W}$ to $K$ channels (one slot per Gaussian). `PixelMLP` (`models/blocks.py`) is $\text{Conv}_{1\times1} \to \text{Act} \to \text{Conv}_{1\times1}$ with `bias=True` on both convolutions; it is the same class [[UNet Per-Gaussian]] uses.

**Output layout.** `_triple_head_forward` computes `out = torch.stack([amp, mu, sigma], dim=2)` $\to (B, K, 3, P_H, P_W)$, then `out.view(B, K\cdot3, P_H, P_W)` flattens with $K$ as the outer index and the parameter type as the inner index, giving channel order $[a_0, \mu_0, \sigma_0, a_1, \mu_1, \sigma_1, \dots]$. [[UNet Per-Gaussian]]'s `_per_gaussian_forward` uses `torch.stack(head_outputs, dim=1)` $\to (B, K, n_{\text{params}}, P_H, P_W)$ then `view(B, K\cdot n_{\text{params}}, P_H, P_W)`, yielding the same $[a_0, \mu_0, \sigma_0, a_1, \dots]$ ordering for $n_{\text{params}}=3$. This layout matches the training loss contract: `Loss.reconstruct_gaussians` (`pipelines/backbone/training/loss.py`) reshapes $(B, C, H, W) \to (B, n_{\text{gaussians}}, \texttt{ppg}, H, W)$ and reads index $0=a$, $1=\mu$, $2=\sigma$ per slot.

---

## Related Notes

- [[UNet]] — Shared backbone
- [[UNet Per-Gaussian]] — Alternative head design
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — UNetMultiHeadConfig
- [[DLR-TomoSAR Index]] — Master index
