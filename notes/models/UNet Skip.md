# UNet Skip

`UNetSkip` (`models/UNet_skip.py`) is a U-Net with residual conv blocks and `MaxPool2d` downsampling. It is the pre-2026-06-04 [[ResUNet]] preserved verbatim so that the checkpoint trained under that earlier architecture remains loadable.

---

## Summary

Each encoder stage applies a residual block (the `ResidualConvBlock` imported from [[ResUNet]], used at its default stride 1) followed by a `MaxPool2d(2)` downsampling step; the deepest stage is a single residual bottleneck; each decoder stage upsamples, concatenates the corresponding encoder skip, and applies a residual block. A final $1\times1$ convolution projects to the output channels. The topology, concatenative skip connections, and upsampling are identical to [[UNet]]; the only difference from [[UNet]] is that the plain double-convolution block is replaced by a residual block.

The module attribute names — `encoder_blocks`, `downsample_layers`, `bottleneck`, `upsample_layers`, `decoder_blocks`, `output_head` — are chosen to reproduce the pre-correction [[ResUNet]] state-dict layout. Because `MaxPool2d` carries no learnable parameters, `downsample_layers` contributes no keys to the state dict, so the saved tensors are exactly `encoder_blocks`, `bottleneck`, `upsample_layers`, `decoder_blocks`, and `output_head`.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{x}$ | input feature map, $\mathbf{x} \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}$ |
| $\mathbf{u}_i$ | intermediate activations of the main branch |
| $B$ | batch size |
| $H, W$ | feature-map height and width |
| $C_{\text{in}}, C_{\text{out}}$ | input and output channel counts |
| $\text{Conv}_{3\times3}$ | $3\times3$ convolution with padding 1 and stride 1 (no spatial size change); the first maps $C_{\text{in}} \to C_{\text{out}}$, the second $C_{\text{out}} \to C_{\text{out}}$ |
| $\text{Conv}_{1\times1}$ | $1\times1$ convolution used as the learned projection shortcut |
| $\text{Norm}$ | normalisation layer (`normalization`; default `"batch"`) |
| $\text{Act}$ | activation function (`activation`; default `"relu"`) |
| $\text{Proj}$ | identity or learned $1\times1$ projection (defined below) |
| $\text{ResBlock}$ | residual block (always stride 1) |
| $K$ | number of Gaussian components |

---

## Architecture

The block uses pre-activation ordering (`ResidualConvBlock`, reused from [[ResUNet]] at stride 1): each convolution is preceded by normalisation and activation. For input $\mathbf{x}$ with $C_{\text{in}}$ channels and output $C_{\text{out}}$:

$$
\begin{aligned}
\mathbf{u}_1 &= \text{Act}(\text{Norm}(\mathbf{x})) \\
\mathbf{u}_2 &= \text{Conv}_{3\times3}(\mathbf{u}_1) \\
\mathbf{u}_3 &= \text{Act}(\text{Norm}(\mathbf{u}_2)) \\
\mathbf{u}_4 &= \text{Conv}_{3\times3}(\mathbf{u}_3) \\
\text{ResBlock}(\mathbf{x}) &= \mathbf{u}_4 + \text{Proj}(\mathbf{x})
\end{aligned}
$$

Optional `Dropout2d` is appended after the final convolution of the main branch when `dropout > 0`.

$$
\text{Proj}(\mathbf{x}) = \begin{cases}
\mathbf{x} & C_{\text{in}} = C_{\text{out}} \\
\text{Conv}_{1\times1, C_{\text{in}} \to C_{\text{out}}}(\mathbf{x}) & C_{\text{in}} \neq C_{\text{out}}
\end{cases}
$$

The projection is applied when input and output channels differ. Because every block runs at stride 1, the shortcut is a $1\times1$ convolution only when channels change and the identity otherwise; the residual shortcut is summed (not concatenated) with the main branch output. Unlike the corrected [[ResUNet]], no stride is ever applied inside the block — all spatial reduction is delegated to the separate `MaxPool2d(2)` layers in the encoder.

---

## Relationship to ResUNet

`UNetSkip` **is** the [[ResUNet]] architecture as it existed before the 2026-06-04 correction. On that date, downsampling in `ResUNet.py` was changed from `MaxPool2d` to stride-2 convolutions inside the residual units, to bring it into line with the residual-unit formulation of Zhang et al. (2018), which states that "instead of using pooling operation to downsample the feature map size, a stride of 2 is applied to the first convolution block to reduce the feature map by half."

To keep that correction from invalidating the model already trained under the old topology, the pre-correction architecture was frozen into this separate module. The two now differ only in how spatial reduction is performed:

| Aspect | `UNetSkip` (pre-correction) | [[ResUNet]] (post-correction) |
|---|---|---|
| Encoder downsampling | `MaxPool2d(2)` after each residual block | stride-2 first conv inside residual blocks 1+ |
| Bottleneck stride | 1 (reduction via preceding pool) | 2 |
| Residual block stride | 1 everywhere | 1 (block 0, decoder) / 2 (downsampling units) |
| First-unit handling | full pre-activation everywhere (`first_unit` always default `False`) | encoder block 0 passes `first_unit = True`, dropping its leading `Norm → Act` |
| Residual block class | `ResidualConvBlock` (imported from `ResUNet.py`) | `ResidualConvBlock` (same class) |
| Skip / decoder mechanics | identical | identical |

Both share the same `ResidualConvBlock` definition; `UNetSkip` simply never passes a non-default `stride` or `first_unit`, so every block — including encoder block 0 — keeps the full pre-activation `Norm → Act → Conv → Norm → Act → Conv` ordering and stride 1. This preserves the pre-correction state-dict layout (in particular, `encoder_blocks.0.layers.0` remains a `BatchNorm`, see Checkpoint Continuity).

---

## Checkpoint Continuity

The reason this module exists is checkpoint continuity. The checkpoint trained before the 2026-06-04 correction corresponds to this architecture, and its state-dict keys/shapes match `UNetSkip` exactly:

- Top-level state-dict module prefixes: `encoder_blocks`, `bottleneck`, `upsample_layers`, `decoder_blocks`, `output_head` — i.e. every parameterised submodule of `UNetSkip`. `downsample_layers` (`MaxPool2d`) holds no parameters and therefore appears with zero keys, exactly as expected.
- Encoder block indices `0–3`, decoder block indices `0–3`, upsample indices `0–3`, plus a single `bottleneck`, consistent with the default four-level `features = [64, 128, 256, 512]`.
- The first encoder convolution weight is `encoder_blocks.0.layers.2.weight` of shape $(64, C_{\text{in}}, 3, 3)$ and the output head is `output_head.weight` of shape $(C_{\text{out}}, 64, 1, 1)$, confirming the leading $\text{Norm}\to\text{Act}$ pre-activation order (the BatchNorm at `layers.0` precedes the first conv at `layers.2`) and the $1\times1$ output projection.
- For the stored checkpoint these resolve to $C_{\text{in}} = 9$ and $C_{\text{out}} = 15$ (five Gaussians at three parameters each), set via config overrides rather than the dataclass defaults ($1$ and $6$).
- The container also carries the usual training-state keys (`epoch`, `global_step`, `best_val_loss`, `best_epoch`, `train_losses`, `val_losses`, `opt_state`, `batch_stats`, `ema_shadow`, `config`, `x_axis`, `scheduler_state`, `warmup_state`, `early_stopping_state`).

Loading this checkpoint into the corrected [[ResUNet]] would silently mismatch the spatial-reduction semantics (and, for the stride-2 shortcut convolutions, the parameter set); loading it into `UNetSkip` is exact.

---

## Parameter Reference

See [[Configuration Layer]] → `UNetSkipConfig`. The configurable fields and their defaults mirror [[UNet]] and [[ResUNet]].

| Parameter | Default | Description |
|---|---|---|
| `features` | `[64, 128, 256, 512]` | Encoder channel widths |
| `bottleneck_factor` | `2` | Bottleneck channel multiplier |
| `dropout` | `0.15` | Per-block dropout |
| `activation` | `"relu"` | Activation function |
| `normalization` | `"batch"` | Normalisation layer |
| `upsample_mode` | `"convtranspose"` | Upsampling mode |
| `in_channels` | `1` | Input channel count |
| `out_channels` | `6` | Output channel count ($3K$ for $K=2$ Gaussians) |

`UNetSkipConfig.get_param_groups` partitions parameters into `encoder` (encoder blocks **and** the `MaxPool` `downsample_layers`), `bottleneck`, `decoder` (upsample layers and decoder blocks), and `output_head`; empty groups are dropped. The `downsample_layers` are included for completeness even though `MaxPool2d` exposes no trainable parameters.

---

## Paper fidelity

**Review date:** 2026-06-04

`UNetSkip` is a **project-specific hybrid with no single publication**. Its three structural ingredients each trace to a different source:

- **Topology** — the encoder/bottleneck/decoder layout with concatenative skip connections follows [[UNet]] (O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," *MICCAI*, 2015; arXiv:1505.04597; [[UNet_Ronneberger2015_1505.04597.pdf|PDF]]).
- **Blocks** — the residual conv blocks follow the residual-unit (pre-activation BN-ReLU-Conv with summed shortcut) formulation of Z. Zhang, Q. Liu, Y. Wang, "Road Extraction by Deep Residual U-Net," *IEEE Geoscience and Remote Sensing Letters*, 2018 (arXiv:1711.10684), Fig. 1(b), Sec. II-A2, Eq. 1–3. [[ResUNet_Zhang2018_1711.10684.pdf|PDF]]
- **Downsampling** — `MaxPool2d(2)` **deviates from Zhang et al.**, who explicitly replace pooling with stride-2 convolutions (Sec. II-A; Table I). The pooling choice here is inherited from the classic [[UNet]] decimation operator.

Consequently this architecture cannot be checked against one paper. The residual-block internals are faithful to Zhang et al. (verified in the [[ResUNet]] note); the spatial-reduction operator is deliberately the [[UNet]] `MaxPool` rather than Zhang's stride-2 convolution. It is retained not as a fidelity target but as a frozen artefact preserving the pre-2026-06-04 [[ResUNet]] so that its trained checkpoint stays usable.

---

## Related Notes

- [[ResUNet]] — Post-correction architecture; this module is its pre-correction form
- [[UNet]] — Base topology and the source of the `MaxPool` downsampling
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — UNetSkipConfig
- [[DLR-TomoSAR Index]] — Master index
