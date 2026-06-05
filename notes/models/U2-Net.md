# U2-Net

`U2NetLite` (`models/U2NetLite.py`) is a two-level nested U-structure ([[U2Net_Qin2020_2005.09007.pdf|Qin et al., 2020]]): the outer network is a U-Net whose blocks are themselves small U-Nets — Residual U-blocks (RSU) — so every stage mixes local and multi-scale context internally before the outer topology downsamples.

---

## Summary

The outer network has three encoder stages (RSU heights `rsu_heights`, default $(5, 4, 3)$, matched to the $64 \to 32 \to 16 \to 8$ resolution ladder), a dilated bridge block at the coarsest resolution, and three mirrored decoder stages with skip concatenation. Unlike the original U2-Net, no side outputs or deep supervision are used — a single 1×1 head at full resolution keeps the loss interface identical to every other model in the zoo.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | RSU input feature map |
| $C_{\text{out}}$ | RSU output channel width |
| $\text{RSU}_L(\mathbf{x})$ | Residual U-block output at height $L$ |
| $\mathcal{H}_{RSU}$ | RSU transform, $\mathcal{U}(\mathcal{F}_1(\mathbf{x})) + \mathcal{F}_1(\mathbf{x})$ |
| $\mathcal{F}_1$ | RSU input convolution projecting $\mathbf{x}$ to $C_{\text{out}}$ |
| $\mathcal{U}_L$, $\mathcal{U}$ | Inner mini U-Net of height $L$ ($L-1$ encoder convs + pooling, dilation-2 bottom conv, $L-1$ decoder convs with skip concat and upsampling) |
| $\text{Conv}$ | $3\times3$ convolution (BN, ReLU) |
| $L$ | RSU height (`rsu_heights`) |
| $m$, $M$ | Intermediate (mid) channel width, $m = \max(C_{\text{out}}/4, 8)$ |

---

## Architecture

### Residual U-block (RSU)

An RSU of height $L$ first projects its input to the output width, then runs a complete mini U-Net over it with narrow mid channels $m = \max(C_{\text{out}}/4, 8)$, and adds the result back:

$$
\text{RSU}_L(\mathbf{x}) = \mathcal{U}_L\big(\text{Conv}(\mathbf{x})\big) + \text{Conv}(\mathbf{x})
$$

The residual is taken over the *block-internal multi-scale transform*, not over a plain convolution as in [[ResUNet]].

### Dilated RSU Bridge

At $8 \times 8$ further pooling is useless, so the bridge replaces pooling with dilation rates $(1, 2, 4, 8)$ at constant resolution — the same multi-scale mixing without resolution change.

---

## Design Rationale

**Intra-stage multi-scale.** In a standard U-Net, a stage sees exactly one scale and relies on the outer topology for context. Each RSU gives every stage its own internal receptive-field pyramid, so even the first full-resolution stage can use 16-pixel context when estimating a pixel's parameters. This is the most aggressive multi-scale design in the zoo, complementary to [[MultiResUNet]] (parallel kernel sizes) and [[DeepLabV3+]] (parallel dilations at one stage).

---

## Parameter Reference

See [[Configuration Layer]] → `U2NetLiteConfig` (`features` — exactly four entries, `rsu_heights`).

---

## Paper fidelity

*Review date: 2026-06-04. Ground truth: Qin et al., "U$^2$-Net: Going Deeper with Nested U-Structure for Salient Object Detection," Pattern Recognition 106 (2020), arXiv:2005.09007 ([[U2Net_Qin2020_2005.09007.pdf|PDF]]). Scope: architecture fidelity of `models/U2NetLite.py` and its `U2NetLiteConfig` builder against Section 3.1 (RSU, Fig. 2e), Section 3.2 (nested-U, Fig. 5, Table 1) and Section 3.3 (supervision, Eq. 1, Fig. 5). Hyperparameters out of scope.*

**Verdict: faithful Lite implementation.** The non-negotiable structural primitives — the RSU residual-U-block mechanism, its dilated RSU-4F variant, and the two-level nested-U topology — are reproduced exactly. All divergences from the full $176.3$ MB U$^2$-Net are either accepted Lite scalings (fewer stages, smaller heights, narrower mid-channels) or one justified task-driven adaptation (removal of deep supervision for a regression head).

| # | Dimension | Paper ref | Code ref | Verdict |
|---|-----------|-----------|----------|---------|
| 1 | RSU input conv $\mathcal{F}_1(\mathbf{x})$ | §3.1(i), Fig. 2e | `U2NetLite.py:30,44` | MATCH |
| 2 | RSU inner encoder ($L-1$ convs + maxpool) | §3.1(ii), Fig. 2e | `U2NetLite.py:32-34,48-52` | MATCH |
| 3 | RSU bottom dilated conv ($d{=}2$) | Fig. 2e (`M,3×3,d=2`) | `U2NetLite.py:36,54` | MATCH |
| 4 | RSU inner decoder (concat skips + bilinear up) | §3.1(ii), Fig. 2e | `U2NetLite.py:38-41,56-60` | MATCH |
| 5 | RSU outer residual addition $\mathcal{U}(\mathcal{F}_1)+\mathcal{F}_1$ | §3.1(iii), Eq. $\mathcal{H}_{RSU}$ | `U2NetLite.py:62` | MATCH |
| 6 | RSU down/up operators (maxpool / bilinear) | Fig. 5 legend | `U2NetLite.py:52,59` | MATCH |
| 7 | RSU-4F dilation pattern $1,2,4/8/4,2,1$, constant res. | §3.1, Fig. 2d/5 | `U2NetLite.py:70-82` | MATCH |
| 8 | Nested-U encoder (RSU stages + pooling between) | §3.2(i), Fig. 5 | `U2NetLite.py:118-122,139-142` | MATCH |
| 9 | RSU-4F at coarsest resolution | §3.2 ("En\_5/En\_6 RSU-4F") | `U2NetLite.py:124,144` | MATCH (Lite placement rule preserved) |
| 10 | Nested-U decoder (RSU + concat skips + up) | §3.2(ii), Fig. 5 | `U2NetLite.py:126-130,146-148` | MATCH |
| 11 | Intermediate mid-channels $M$ | Table 1 (per-stage $M$) | `U2NetLite.py:115-116` | ACCEPTED ADAPTATION |
| 12 | Stage count / RSU heights | §3.2, Table 1 (6 enc / RSU-7..4) | `U2NetLite.py:118-130`; cfg `rsu_heights=(5,4,3)` | ACCEPTED ADAPTATION |
| 13 | Channel progression between stages | Table 1 | `models_config.py:1474` `features=[64,128,256,512]` | ACCEPTED ADAPTATION |
| 14 | Resolution bookkeeping (ceil pool + interpolate-to-skip) | Fig. 5 | `U2NetLite.py:52,58-59,147` | MATCH (robust to odd sizes) |
| 15 | Side outputs + deep supervision (6 maps, fuse) | §3.3, Eq. 1, Fig. 5 | absent; `U2NetLite.py:133,150-151` | ACCEPTED ADAPTATION |
| 16 | Output activation (sigmoid per side output) | §3.3, Fig. 5 legend | `U2NetLite.py:151` (linear $1\times1$) | ACCEPTED ADAPTATION |

### RSU and RSU-4F

The bottom-level U-block is reproduced verbatim. `RSU.forward` (`U2NetLite.py:43-62`) computes the local feature $\mathcal{F}_1(\mathbf{x}) = $ `conv_in` (line 44), runs the height-$L$ symmetric encoder–decoder $\mathcal{U}_L$, and returns $\mathcal{U}_L(\mathcal{F}_1(\mathbf{x})) + \mathcal{F}_1(\mathbf{x})$ (line 62) — exactly Eq. $\mathcal{H}_{RSU}(\mathbf{x}) = \mathcal{U}(\mathcal{F}_1(\mathbf{x})) + \mathcal{F}_1(\mathbf{x})$ of §3.1(iii). The inner encoder holds $L-1$ convolutions (one $C_{\text{out}}{\to}M$ plus $L-2$ at $M{\to}M$), maxpools between them with `ceil_mode=True`, applies a dilation-2 bottom convolution (matching Fig. 2e's `M,3×3,d=2`), and decodes with $L-1$ convolutions that concatenate the mirrored encoder skip and bilinearly upsample to its spatial size. The final decoder convolution narrows back to $C_{\text{out}}$ (line 40), so the residual addition is channel-consistent. The dilated variant `RSUDilated` (`U2NetLite.py:65-99`) drops all resolution change and uses dilations $1,2,4$ (encoder), $8$ (bottom), $4,2,1$ (decoder) with concat skips — the RSU-4F of §3.1 and Fig. 2d/5, reproduced exactly.

The intermediate width is set heuristically to $M = \max(C_{\text{out}}/4,\,8)$ (`U2NetLite.py:115-116`) rather than the per-stage $M$ tabulated in Table 1. This is an accepted Lite scaling: $M$ controls the bottleneck capacity of the intra-stage transform, not its topology, and the paper itself treats $M$ as a free configuration knob (Fig. 4 sweeps $M$).

### Nested-U topology

The outer network is a two-level nested U: three RSU encoder stages with maxpool between them (`U2NetLite.py:139-142`), a single RSU-4F bridge at the coarsest resolution (line 144), and three mirrored RSU decoder stages consuming concatenated skips after bilinear upsampling (lines 146-148). This is the encoder/bridge/decoder skeleton of Fig. 5. The Lite version uses three encoder stages versus the paper's six (En\_1..En\_6) and heights $(5,4,3)$ versus RSU-7/6/5/4; both are explicitly sanctioned scaling parameters. Critically, the **placement rule** the paper specifies for the dilated variant is preserved: RSU-4F is used precisely where the resolution has become too low for further pooling to be meaningful (the paper applies it at En\_5, En\_6, De\_5; the Lite version applies it at its single deepest/bridge position). The qualitative invariant — "dilated where resolution is too low, pooled otherwise" — holds.

### Deep supervision and output activation (the two non-MATCH structural items)

The implementation omits the saliency fusion module of §3.3: it does not emit six side outputs $\mathcal{S}_{side}^{(m)}$ via per-stage $3\times3$+sigmoid convolutions, upsample-and-concatenate them, and fuse with a $1\times1$ convolution (Fig. 5, Eq. 1 $\mathcal{L} = \sum_m w_{side}^{(m)} \ell_{side}^{(m)} + w_{fuse}\ell_{fuse}$). Instead a single linear $1\times1$ head at full resolution produces the output (`U2NetLite.py:133,151`).

This is classified as an **ACCEPTED ADAPTATION**, not a deviation, on two grounds. (a) Deep supervision in U$^2$-Net is a *training/loss* device: every side output is regressed against the same binary saliency mask with per-map BCE. The mechanism presupposes a single shared target map that is meaningful at every scale — appropriate for binary saliency, not for the multi-channel continuous regression target here (`out_channels=6`, Gaussian-mixture parameters), where a downsampled side output has no well-defined ground truth and a per-map sigmoid would clamp an unbounded regression range. (b) Removing it does not touch the nested-U feature extractor, which is the architectural contribution under test; it only changes the head and loss interface, deliberately aligning U2NetLite with the single-head convention of the rest of the model zoo. The same reasoning justifies the absence of the sigmoid activation (item 16): a linear head is the correct choice for unbounded regression.

If a future task reverts to a bounded/segmentation target, restoring fidelity is mechanical: add a $3\times3$ side-output convolution to each decoder stage and the bridge, `interpolate` each to input size, `cat` and fuse with a $1\times1$ convolution, and supervise each with the loss of Eq. 1.

### Resolution bookkeeping

The code is in fact more robust than the paper diagram for non-power-of-two inputs: `ceil_mode=True` pooling plus `interpolate(size=skip.shape[2:])` (rather than a fixed `scale_factor=2`) guarantee every decoder feature is resized to its exact skip resolution before concatenation, both inside RSU (lines 58-59) and in the outer decoder (line 147). No spatial-size mismatch is possible. MATCH.

### Deviations

None of structural or minor severity. All non-MATCH dimensions are accepted Lite scalings (items 11-13) or the justified supervision/activation adaptation for the regression task (items 15-16).

---

## Related Notes

- [[UNet]] — Outer topology
- [[MultiResUNet]] / [[DeepLabV3+]] — Alternative multi-scale mechanisms
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — U2NetLiteConfig
- [[DLR-TomoSAR Index]] — Master index
