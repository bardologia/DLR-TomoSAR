# ResUNet Multihead

`ResUNetMultiHead` (`models/ResUNet_multihead.py`) combines the [[ResUNet]] backbone with the parameter-type-specific output heads of [[UNet Multihead]]: three independent `PixelMLP` heads — amplitude, mean elevation, sigma — replace the single shared 1×1 convolution.

---

## Summary

The encoder, bottleneck, and decoder are identical to [[ResUNet]] (pre-activation residual blocks with stride-2 convolution downsampling and skip concatenation). The decoder embedding $\mathbf{e} \in \mathbb{R}^{B \times F_1 \times P_H \times P_W}$ is fed to the three heads exactly as in [[UNet Multihead]]; the `PixelMLP` definition and the interleaved output assembly $[a_1, \mu_1, \sigma_1, a_2, \mu_2, \sigma_2, \dots]$ are imported from that model and shared verbatim.

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

[[ResUNet]] won the capacity-matched benchmark on the plain-head design, while the head structure experiments ([[UNet Multihead]], [[UNet Per-Gaussian]]) were only run on the weaker UNet backbone. This variant separates the two factors: if parameter-type specialisation helps, it should help on top of the strongest backbone, not only the baseline one.

> The residual backbone changes only the embedding quality; the head inductive bias (parameter-type consistency across all Gaussian slots) is unchanged from [[UNet Multihead]].

---

## Parameter Reference

See [[Configuration Layer]] → `ResUNetMultiHeadConfig`. All base ResUNet parameters apply; the param groups expose a separate `heads_lr`/`heads_wd` for the three MLP heads.

---

## Backbone consistency

*Reviewed 2026-06-04.*

Verified line-by-line that the encoder, bottleneck, and decoder of `ResUNetMultiHead` are structurally identical to [[ResUNet]] in its post-correction form. The shared `ResidualConvBlock` and `match_spatial_size` are imported directly from `models/ResUNet.py`, so the block definition cannot diverge.

**Backbone identity (incl. both 2026-06-04 corrections).** The earlier-today corrections (i) replaced `MaxPool2d` downsampling with stride-2 first convolutions inside the residual units and (ii) added the `first_unit` flag so encoder block 0 omits its leading `Norm → Act` and operates on the raw input. `ResUNetMultiHead.__init__` reproduces both exactly: the encoder loop applies `stride = 1 if index == 0 else 2` together with `first_unit = index == 0`, the bottleneck uses `stride = 2` (and the default `first_unit = False`), and the decoder `ResidualConvBlock`s use the defaults `stride = 1`, `first_unit = False`. There are no leftover `downsample_layers` or `MaxPool2d` modules. Skip wiring (`skip_connections.append(x)` per encoder block, consumed in `reversed` order), upsampling (`build_upsample` over `[bottleneck] + features[::-1]`), and the `cat([skip, x], dim=1)` concatenation are character-for-character the same as the base. The construction is therefore consistent with the base correction.

**Head factorization.** Three shared `PixelMLP` heads — `head_amp`, `head_mu`, `head_sigma` — each map the decoder embedding $\mathbf{e}\in\mathbb{R}^{B\times F_1\times H\times W}$ to $K$ channels (one slot per Gaussian, $K = \texttt{out\_channels}/\texttt{params\_per\_gaussian}$). The single shared 1×1 `output_head` of the base [[ResUNet]] is the only construction difference; the backbone is otherwise unchanged.

**Layout equivalence.** Forward assembles $[a_0,\mu_0,\sigma_0,a_1,\mu_1,\sigma_1,\dots]$ via
$$\texttt{stack}([a,\mu,\sigma],\,\dim=2)\in\mathbb{R}^{B\times K\times 3\times H\times W}\;\xrightarrow{\texttt{view}}\;\mathbb{R}^{B\times 3K\times H\times W},$$
with slot index $K$ outer and parameter type ($3$) inner, yielding the interleaved layout. This is byte-identical to `UNet_multihead.py` (same `stack(dim=2)`/`view` ops). The per-Gaussian variants (`ResUNet_pergaussian.py`, `UNet_pergaussian.py`) reach the same layout by `stack(head_outputs, dim=1)` then `view`, with slot outer and `n_params` inner. All four variants therefore emit the same channel ordering and are interchangeable downstream.

**Param groups.** `ResUNetMultiHeadConfig.get_param_groups` covers `encoder_blocks`, `bottleneck`, `upsample_layers` + `decoder_blocks`, and `head_amp` + `head_mu` + `head_sigma`. Post-correction there is no `downsample_layers` module, and the method correctly does not reference one — coverage is complete.

**Divergences.** None structural. The only intended differences from the base are the output head (three `PixelMLP`s vs. one 1×1 conv) and the corresponding `heads_lr`/`heads_wd` param group. The config dataclass matches `ResUNetConfig` field-for-field except for this head naming (`heads_lr`/`heads_wd` vs. `output_head_lr`/`output_head_wd`). Note: the **Summary** section above previously described "MaxPool downsampling"; this was corrected to stride-2 convolution downsampling on 2026-06-04.

---

## Related Notes

- [[ResUNet]] — Shared backbone
- [[UNet Multihead]] — Head design and PixelMLP definition
- [[ResUNet Per-Gaussian]] — Alternative head design on the same backbone
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — ResUNetMultiHeadConfig
- [[DLR-TomoSAR Index]] — Master index
