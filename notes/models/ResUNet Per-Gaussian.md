# ResUNet Per-Gaussian

`ResUNetPerGaussian` (`models/ResUNet_pergaussian.py`) combines the [[ResUNet]] backbone with the per-slot output heads of [[UNet Per-Gaussian]]: one `PixelMLP` head per Gaussian component, each predicting that slot's $(a, \mu, \sigma)$ triple.

---

## Summary

The encoder, bottleneck, and decoder are identical to [[ResUNet]]. The decoder embedding is fed to $K$ independent `PixelMLP` heads (imported from [[UNet Multihead]]); each head outputs $(B, 3, P_H, P_W)$ and the outputs are stacked to the interleaved layout $[a_1, \mu_1, \sigma_1, a_2, \mu_2, \sigma_2, \dots]$, matching the loss and metric conventions.

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

*Reviewed 2026-06-04.*

Verified line-by-line that the encoder, bottleneck, and decoder of `ResUNetPerGaussian` are structurally identical to [[ResUNet]] in its post-correction form. The shared `ResidualConvBlock` and `match_spatial_size` are imported directly from `models/ResUNet.py`, so the block definition cannot diverge.

**Backbone identity (incl. both 2026-06-04 corrections).** The earlier-today corrections (i) replaced `MaxPool2d` downsampling with stride-2 first convolutions inside the residual units and (ii) added the `first_unit` flag so encoder block 0 omits its leading `Norm → Act` and operates on the raw input. `ResUNetPerGaussian.__init__` reproduces both exactly: the encoder loop applies `stride = 1 if index == 0 else 2` together with `first_unit = index == 0`, the bottleneck uses `stride = 2` (and the default `first_unit = False`), and the decoder `ResidualConvBlock`s use the defaults `stride = 1`, `first_unit = False`. There are no leftover `downsample_layers` or `MaxPool2d` modules. Skip wiring, upsampling (`build_upsample` over `[bottleneck] + features[::-1]`), and the `cat([skip, x], dim=1)` concatenation are character-for-character the same as the base. The construction is consistent with the base correction and with [[ResUNet Multihead]].

**Head factorization.** $K$ independent `PixelMLP` heads in `gaussian_heads` ($K = \texttt{out\_channels}/\texttt{params\_per\_gaussian}$), each mapping the decoder embedding $\mathbf{e}\in\mathbb{R}^{B\times F_1\times H\times W}$ to `n_params` channels — the full $(a,\mu,\sigma)$ triple for one slot. This is the per-slot complement of the per-parameter-type factorization in [[ResUNet Multihead]]; the backbone is otherwise unchanged from the base.

**Layout equivalence.** Forward assembles $[a_0,\mu_0,\sigma_0,a_1,\mu_1,\sigma_1,\dots]$ via
$$\texttt{stack}(\text{head\_outputs},\,\dim=1)\in\mathbb{R}^{B\times K\times n_p\times H\times W}\;\xrightarrow{\texttt{view}}\;\mathbb{R}^{B\times K n_p\times H\times W},$$
with slot index $K$ outer and parameter type ($n_p$) inner, yielding the interleaved layout. This is byte-identical to `UNet_pergaussian.py`. The multihead variants (`ResUNet_multihead.py`, `UNet_multihead.py`) reach the same layout by `stack([a,\mu,\sigma], dim=2)` then `view`, with slot outer and parameter type inner. All four variants therefore emit the same channel ordering and are interchangeable downstream.

**Param groups.** `ResUNetPerGaussianConfig.get_param_groups` covers `encoder_blocks`, `bottleneck`, `upsample_layers` + `decoder_blocks`, and `gaussian_heads`. Post-correction there is no `downsample_layers` module, and the method correctly does not reference one — coverage is complete.

**Divergences.** None structural. The only intended differences from the base are the output head ($K$ `PixelMLP`s vs. one 1×1 conv) and the corresponding `heads_lr`/`heads_wd` param group. The config dataclass matches `ResUNetConfig` field-for-field except for this head naming (`heads_lr`/`heads_wd` vs. `output_head_lr`/`output_head_wd`).

---

## Related Notes

- [[ResUNet]] — Shared backbone
- [[UNet Per-Gaussian]] — Head design
- [[ResUNet Multihead]] — Alternative head design on the same backbone
- [[Parameter Matching]] — Why slots have consistent roles
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — ResUNetPerGaussianConfig
- [[DLR-TomoSAR Index]] — Master index
