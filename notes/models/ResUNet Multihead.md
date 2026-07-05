---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - ResUNetMultiHead
family: head-variant
registry_key: resunet_multihead
summary: ResUNet backbone combined with the three parameter-type-specific PixelMLP heads of UNet Multihead.
---

# ResUNet Multihead

`ResUNetMultiHead` (`models/backbone/resunet.py`) combines the [[ResUNet]] backbone with the parameter-type-specific output heads of [[UNet Multihead]]: three independent `PixelMLP` heads — amplitude, mean elevation, sigma — in place of the single shared 1×1 convolution of `ResUNet`.

---

## Summary

`ResUNetMultiHead` subclasses the shared `ResUNetBackbone` (`models/backbone/resunet.py`); the encoder, bottleneck, and decoder are therefore the same construction as [[ResUNet]] (pre-activation residual blocks with stride-2 convolution downsampling and skip concatenation). The decoder embedding $\mathbf{e} \in \mathbb{R}^{B \times F_1 \times P_H \times P_W}$ is fed to the three heads through the `GaussianHeadsMixin` (`models/blocks.py`), which both `ResUNetBackbone` and `UNetBackbone` inherit; the `PixelMLP` definition and the interleaved output assembly $[a_1, \mu_1, \sigma_1, a_2, \mu_2, \sigma_2, \dots]$ are shared with [[UNet Multihead]].

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{e}$ | decoder embedding, $\mathbf{e} \in \mathbb{R}^{B \times F_1 \times P_H \times P_W}$ |
| $B$ | batch size |
| $F_1$ | first (finest) encoder feature width, the embedding channel count |
| $P_H, P_W$ | patch (feature-map) height and width; written $H, W$ where unambiguous |
| $K$ | number of Gaussian components, $K = \texttt{out\_channels}/\texttt{params\_per\_gaussian}$ |
| $a, \mu, \sigma$ | amplitude, mean elevation, and sigma outputs per Gaussian slot |

---

## Design Rationale

This variant separates two factors that the plain-head [[ResUNet]] and the head-structure variants on the UNet backbone ([[UNet Multihead]], [[UNet Per-Gaussian]]) confound: backbone strength and head inductive bias. If parameter-type specialisation helps, it should help on top of the residual backbone, not only the plain-convolution one.

> The residual backbone affects only embedding quality; the head inductive bias (parameter-type consistency across all Gaussian slots) is the same as in [[UNet Multihead]].

---

## Parameter Reference

See [[Configuration Layer]] → `ResUNetMultiHeadConfig`. All base ResUNet parameters apply; the param groups expose a separate `heads_lr`/`heads_wd` for the three MLP heads.

---

## Backbone consistency

The encoder, bottleneck, and decoder of `ResUNetMultiHead` are the same construction as [[ResUNet]]. Both subclass `ResUNetBackbone` (`models/backbone/resunet.py`), and the `ResidualConvBlock`, `build_upsample`, and `match_spatial_size` are imported from `models/blocks.py`, so the block definition is shared.

**Backbone.** `ResUNetBackbone.__init__(config, downsample="stride")` builds the encoder, bottleneck, and decoder. With the default `downsample="stride"`: the encoder loop applies `stride = 1 if index == 0 else 2` together with `first_unit = index == 0` (so encoder block 0 omits its leading `Norm → Act` and operates on the raw input), and the bottleneck uses `stride = 2`; the decoder `ResidualConvBlock`s use the defaults `stride = 1`, `first_unit = False`. The `self.downsample_layers` `ModuleList` is always created but stays empty in stride mode; it is populated with `MaxPool2d(kernel_size=2)` only under `downsample="maxpool"` (the mode used by the `UNetSkip` subclass). Skip wiring (`skip_connections.append(x)` per encoder block, consumed in `reversed` order), upsampling (`build_upsample` over `[bottleneck_channels] + features[::-1]`), and the `cat([skip, x], dim=1)` concatenation are the same as `ResUNet`.

**Head factorization.** `_resolve_gaussian_layout` sets `n_gaussians = out_channels // params_per_gaussian` and `_build_triple_heads` constructs three `PixelMLP` heads — `head_amp`, `head_mu`, `head_sigma` — each mapping the decoder embedding $\mathbf{e}\in\mathbb{R}^{B\times F_1\times H\times W}$ through `embedding_channels → hidden_channels → n_gaussians`, where `hidden_channels = max(embedding_channels // 2, 16)`. With the config defaults (`out_channels = 6`, `params_per_gaussian = 3`) this gives $K = 2$ slots per head. The single 1×1 `output_head` of `ResUNet` is the only construction difference.

**Layout.** `_triple_head_forward` (`GaussianHeadsMixin`) assembles $[a_0,\mu_0,\sigma_0,a_1,\mu_1,\sigma_1,\dots]$ via
$$\texttt{stack}([a,\mu,\sigma],\,\dim=2)\in\mathbb{R}^{B\times K\times 3\times H\times W}\;\xrightarrow{\texttt{view}}\;\mathbb{R}^{B\times 3K\times H\times W},$$
with slot index $K$ outer and parameter type ($3$) inner, yielding the interleaved layout. The same mixin method serves `UNetMultiHead` (`models/backbone/unet.py`), which shares `GaussianHeadsMixin`. The per-Gaussian variants (`ResUNetPerGaussian` in `models/backbone/resunet.py`, `UNetPerGaussian` in `models/backbone/unet.py`) reach the same layout via `_per_gaussian_forward`: `stack(head_outputs, dim=1)` then `view`, with slot outer and `n_params` inner. All four variants emit the same channel ordering and are interchangeable downstream.

**Param groups.** `ResUNetMultiHeadConfig.get_param_groups` covers `encoder_blocks`, `bottleneck`, `upsample_layers` + `decoder_blocks`, and `head_amp` + `head_mu` + `head_sigma`, filtering out any group with no parameters. The empty `downsample_layers` ModuleList contributes no parameters in stride mode and is not referenced; coverage is complete.

**Divergences.** None structural. The only differences from `ResUNet` are the output head (three `PixelMLP`s built by `_build_triple_heads` vs. one 1×1 conv) and the corresponding `heads_lr`/`heads_wd` param group. The config dataclass matches `ResUNetConfig` field-for-field except for this head naming (`heads_lr`/`heads_wd` vs. `output_head_lr`/`output_head_wd`).

---

## Related Notes

- [[ResUNet]] — Shared backbone
- [[UNet Multihead]] — Head design and PixelMLP definition
- [[ResUNet Per-Gaussian]] — Alternative head design on the same backbone
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — ResUNetMultiHeadConfig
- [[DLR-TomoSAR Index]] — Master index
