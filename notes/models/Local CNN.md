---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - LocalCNN
  - Shallow CNN Baseline
family: local_cnn
registry_key: local_cnn
summary: Shallow full-resolution CNN control baseline; a fixed small receptive field with no downsampling, between PixelMLP (no context) and the encode-decode backbones (global context).
---

# Local CNN

`LocalCNN` (`models/backbone/pixel_baselines.py`) is the local-context control of the zoo: a stack of standard `ConvBlock`s (two $3\times3$ convolutions each) applied at full resolution with no downsampling, followed by a $1\times1$ `output_head`. Its receptive field is fixed and small by construction, so it measures how much of the task is solved by a pixel's immediate neighbourhood.

---

## Summary

The trunk chains $B$ `ConvBlock`s — the same double-conv block used by the [[UNet]] family — at constant resolution, each block widening (or keeping) the channel count per `features`. With two $3\times3$ convolutions per block, the receptive field after $B$ blocks is $(4B+1)\times(4B+1)$: the default `features = [256, 256, 256]` gives a $13\times13$ window. Together with [[PixelMLP]] it brackets the spatial-context question: PixelMLP has none, LocalCNN has a fixed local window, and the encode-decode backbones aggregate context across the whole patch. Comparing the three at matched capacity decomposes backbone performance into per-pixel mapping, local smoothing, and long-range context.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{h}^{(b)}$ | Feature map after block $b$ (channels `features[b]`, full resolution) |
| $B$ | Number of `ConvBlock`s, $B = \lvert\texttt{features}\rvert$ |
| $R$ | Receptive field side length, $R = 4B + 1$ |
| $\mathbf{y}$ | Output parameter map (channels $3K$) |

---

## Architecture

$$
\begin{aligned}
\mathbf{h}^{(0)} &= \mathbf{x} \\
\mathbf{h}^{(b)} &= \text{ConvBlock}\!\left(\mathbf{h}^{(b-1)}\right), \qquad b = 1, \dots, B \\
\mathbf{y} &= \text{Conv}_{1\times1}\!\left(\mathbf{h}^{(B)}\right)
\end{aligned}
$$

No pooling, striding, or upsampling appears anywhere: spatial resolution is preserved end to end, and context grows only through convolution overlap at $+4$ pixels per block.

---

## Design Rationale

**The middle rung of the context ladder.** If LocalCNN matches the full backbones, long-range context is unnecessary and the encode-decode round trip is overhead; if it clearly beats [[PixelMLP]] but trails the U-shaped models, both local smoothing and long-range aggregation contribute; if all three tie, the task is per-pixel and architecture choice is moot. Each outcome is directly interpretable, which is the point of the control pair.

**Relation to HRNet.** [[HRNet]] also maintains a full-resolution stream, but fuses it with downsampled branches; LocalCNN removes the branches entirely, isolating "full resolution, local context only" as a hypothesis.

---

## Parameter Reference

See [[Configuration Layer]] → `LocalCNNConfig` (`features` — channel width per `ConvBlock`, receptive field $4B+1$; `activation`, `normalization`, `dropout`).

---

## Provenance

Not derived from a reference paper: this is a purpose-built scientific control completing the context ladder PixelMLP → LocalCNN → encode-decode backbones.

---

## Related Notes

- [[PixelMLP]] — The no-context companion control
- [[HRNet]] — Full-resolution stream with multi-scale fusion
- [[UNet]] — Source of the shared ConvBlock
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — LocalCNNConfig
- [[DLR-TomoSAR Index]] — Master index
