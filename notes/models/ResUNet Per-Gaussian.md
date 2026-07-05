---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - ResUNetPerGaussian
family: head-variant
registry_key: resunet_pergaussian
summary: ResUNet backbone combined with the per-slot PixelMLP heads of UNet Per-Gaussian.
---

# ResUNet Per-Gaussian

`ResUNetPerGaussian` (`models/backbone/resunet.py`) combines the [[ResUNet]] backbone with the per-slot output heads of [[UNet Per-Gaussian]]: one `PixelMLP` head per Gaussian component, each predicting that slot's $(a, \mu, \sigma)$ triple.

---

## Summary

The encoder, bottleneck, and decoder are identical to [[ResUNet]]. The decoder embedding is fed to $K$ independent `PixelMLP` heads (the shared class from `models/blocks.py`); each head outputs $(B, 3, P_H, P_W)$ and the outputs are stacked to the interleaved layout $[a_1, \mu_1, \sigma_1, a_2, \mu_2, \sigma_2, \dots]$, matching the loss and metric conventions.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{e}$ | decoder embedding, $\mathbf{e} \in \mathbb{R}^{B \times F_1 \times H \times W}$ |
| $B$ | batch size |
| $F_1$ | first (finest) encoder feature width, the embedding channel count |
| $H, W$ | feature-map height and width; written $P_H, P_W$ where patch dimensions are emphasised |
| $K$ | number of Gaussian components (slots), $K = \texttt{out\_channels}/\texttt{params\_per\_gaussian}$ |
| $n_p$ | number of parameters per Gaussian (`n_params`); the $(a, \mu, \sigma)$ triple |
| $a, \mu, \sigma$ | amplitude, mean elevation, and sigma outputs per Gaussian slot |

---

## Design Rationale

The per-Gaussian design shares one MLP per *slot* across all parameter types, imposing slot-role consistency as the dominant inductive bias — the complementary choice to [[ResUNet Multihead]], which shares one MLP per *parameter type* across all slots. Running both on the benchmark-winning [[ResUNet]] backbone isolates the head-structure question from the backbone question.

> Slot-role specialisation is plausible in this problem: the GT parameters are mu-sorted (see [[Parameter Matching]]), so slot $k$ consistently represents the $k$-th lowest scatterer, and a dedicated head can specialise on that role.

---

## Parameter Reference

See [[Configuration Layer]] → `ResUNetPerGaussianConfig`. All base ResUNet parameters apply; the param groups expose a separate `heads_lr`/`heads_wd` for the per-slot heads.

---

## Backbone consistency

The encoder, bottleneck, and decoder of `ResUNetPerGaussian` are the shared `ResUNetBackbone` (`models/backbone/resunet.py`); `ResidualConvBlock` and `match_spatial_size` are imported from `models/blocks.py`, so the block definitions are common to all backbone variants.

**Backbone identity.** `ResUNetPerGaussian` subclasses `ResUNetBackbone`, instantiated with the default `downsample="stride"`. The encoder loop applies `stride = 1 if index == 0 else 2` together with `first_unit = index == 0`, so encoder block 0 runs at full resolution and omits its leading `Norm → Act` (`ResidualConvBlock` skips them when `first_unit=True`); the bottleneck uses `stride = 2` with the default `first_unit = False`; and the decoder `ResidualConvBlock`s use the defaults `stride = 1`, `first_unit = False`. In stride mode `self.downsample_layers` is an empty `ModuleList` — built but never populated, since downsampling is folded into the stride-2 convolutions. Skip wiring, upsampling (`build_upsample` over `[bottleneck_channels] + features[::-1]`), and the `cat([skip, x], dim=1)` concatenation are identical to [[ResUNet]] and [[ResUNet Multihead]], which share the same backbone class.

**Head factorization.** The mixin `GaussianHeadsMixin._build_per_gaussian_heads` builds $K$ independent `PixelMLP` heads in `gaussian_heads` ($K = \texttt{out\_channels}/\texttt{params\_per\_gaussian}$, computed by `_resolve_gaussian_layout`), each mapping the decoder embedding $\mathbf{e}\in\mathbb{R}^{B\times F_1\times H\times W}$ to `n_params` channels — the full $(a,\mu,\sigma)$ triple for one slot. `PixelMLP` is a two-layer $1\times1$ conv stack `embedding_channels → hidden_channels → n_params` with one activation, where `hidden_channels = max(embedding_channels // 2, 16)`. This is the per-slot complement of the per-parameter-type factorization in [[ResUNet Multihead]].

**Layout equivalence.** `_per_gaussian_forward` assembles $[a_0,\mu_0,\sigma_0,a_1,\mu_1,\sigma_1,\dots]$ via
$$\texttt{stack}(\text{head\_outputs},\,\dim=1)\in\mathbb{R}^{B\times K\times n_p\times H\times W}\;\xrightarrow{\texttt{view}}\;\mathbb{R}^{B\times K n_p\times H\times W},$$
with slot index $K$ outer and parameter type ($n_p$) inner, yielding the interleaved layout. `UNetPerGaussian` in `models/backbone/unet.py` calls the same mixin method. The multihead variants (`ResUNetMultiHead`, `UNetMultiHead`) reach the same layout through `_triple_head_forward`, which builds `head_amp`/`head_mu`/`head_sigma` and applies `stack([amp, mu, sigma], dim=2)` then `view`, again slot outer and parameter type inner. All four variants therefore emit the same channel ordering and are interchangeable downstream.

**Param groups.** `ResUNetPerGaussianConfig.get_param_groups` covers `encoder_blocks`, `bottleneck`, `upsample_layers` + `decoder_blocks`, and `gaussian_heads`, dropping any group whose parameter list is empty (which removes the empty `downsample_layers` from the decoder group construction — `decoder_params` references only `upsample_layers` and `decoder_blocks`). Coverage is complete.

**Divergences.** None structural. The only differences from [[ResUNet]] are the output head ($K$ `PixelMLP`s vs. one $1\times1$ conv `output_head`) and the corresponding `heads_lr`/`heads_wd` param group. The config dataclass matches `ResUNetConfig` field-for-field except for this head naming (`heads_lr`/`heads_wd` vs. `output_head_lr`/`output_head_wd`).

---

## Related Notes

- [[ResUNet]] — Shared backbone
- [[UNet Per-Gaussian]] — Head design
- [[ResUNet Multihead]] — Alternative head design on the same backbone
- [[Parameter Matching]] — Why slots have consistent roles
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — ResUNetPerGaussianConfig
- [[DLR-TomoSAR Index]] — Master index
