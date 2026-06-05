# LinkNet

`LinkNet` (`models/LinkNet.py`) implements a compact encoder-decoder architecture where skip connections are additive rather than concatenative ([[LinkNet_Chaurasia2017_1707.03718.pdf|Chaurasia & Culurciello, 2017]]). The decoder adds a transformed version of the encoder skip back to the decoder feature map rather than doubling the channel count via concatenation.

---

## Summary

The encoder uses ResNet-style residual blocks. The decoder inverts each encoder block via a 1×1 projection, followed by transposed convolution for upsampling and another 1×1 projection. The skip contribution is added element-wise to the decoder output at each level.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{s}_l \in \mathbb{R}^{B \times F_l \times H_l \times W_l}$ | Encoder feature map (skip) at level $l$ |
| $\mathbf{d}_{l+1}^{\uparrow} \in \mathbb{R}^{B \times F_l \times H_l \times W_l}$ | Decoder feature from the deeper level, already upsampled to level $l$ |
| $\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$ | Intermediate bottleneck features: $1\times1$ compression, transposed-conv upsampling, $1\times1$ expansion, respectively |
| $\text{Decode}(\cdot)$ | Decoder-block output at level $l$ |
| $B$ | Batch size |
| $H_l, W_l$ | Spatial dimensions at level $l$ |
| $l$ | Decoder / encoder level index |
| $F_l$ | Encoder channel width at level $l$ (from `features`; default `[152, 312, 624, 1248]`) |
| $c$ | `decoder_bottleneck_ratio` (default `4`); each `BottleneckDecoderBlock` compresses to $\max(1, F_l // c)$ channels |
| $F_l / c$ | Intermediate bottleneck width |
| $m, n$ | Decoder-block input and output channel counts (Fig. 3 notation) |
| $k$ | `initial_kernel_size`, kernel size of the initial conv stem (default `7`) |
| $\text{Conv}_{1\times1, A \to B}$ | $1\times1$ bottleneck projection from $A$ to $B$ channels, each followed by `Norm` and `Act` |
| $\text{ConvT}_{3\times3, \cdot, s=2}$ | Transposed $3\times3$ convolution with stride 2 (and `output_padding=1`) for $2\times$ upsampling |
| $\text{BN}$ | Normalisation layer (`normalization`; default `"batch"`) |

---

## Architecture

The encoder uses ResNet-style residual stages over a convolutional stem; the decoder inverts each encoder block and adds the skip contribution element-wise.

### Decoder Block

At level $l$, with encoder skip $\mathbf{s}_l \in \mathbb{R}^{B \times F_l \times H_l \times W_l}$ and upsampled decoder input $\mathbf{d}_{l+1}^{\uparrow} \in \mathbb{R}^{B \times F_l \times H_l \times W_l}$:

$$
\begin{aligned}
\mathbf{b}_1 &= \text{Conv}_{1\times1, F_l \to F_l/4}(\mathbf{s}_l) \\
\mathbf{b}_2 &= \text{BN}\!\left(\text{ConvT}_{3\times3, F_l/4 \to F_l/4, s=2}(\mathbf{b}_1)\right) \\
\mathbf{b}_3 &= \text{Conv}_{1\times1, F_l/4 \to F_{l-1}}(\mathbf{b}_2) \\
\text{Decode}(\mathbf{s}_l, \mathbf{d}_{l+1}^{\uparrow}) &= \mathbf{b}_3 + \mathbf{d}_{l+1}^{\uparrow}
\end{aligned}
$$

The `+` denotes element-wise addition of the skip-derived features to the decoder features (requires matching spatial size, enforced by `match_spatial_size`, and matching channels).

The block diagram above uses $c = 4$ for compactness; the general compression ratio is the configurable `decoder_bottleneck_ratio`. An initial $k \times k$ convolution stem (`initial_kernel_size`, default $k = 7$) maps the `in_channels` input to `features[0]` channels before the first encoder stage. Each encoder stage (`ResidualEncoderBlock`) downsamples by a stride-2 convolution with a parallel stride-2 1×1 residual shortcut.

---

## Design Rationale

> **Additive vs. concatenative skips.** Concatenation doubles the decoder's input channel count at each level; additive skips preserve the channel count, yielding a significantly lighter decoder.

Concatenation doubles the decoder's input channel count at each level, increasing parameters. Additive skips preserve the channel count, yielding a significantly lighter decoder. This is motivated by the observation (LinkNet paper) that low-level encoder features primarily need to add residual spatial detail to the decoder, not replace its content.

---

## Parameter Reference

See [[Configuration Layer]] → `LinkNetConfig`.

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `features` | $F_l$ | `[152, 312, 624, 1248]` | Encoder channel widths per stage |
| `initial_kernel_size` | $k$ | `7` | Kernel size of the initial conv stem |
| `decoder_bottleneck_ratio` | $c$ | `4` | Compression ratio inside each decoder block |
| `dropout` | — | `0.15` | Encoder block dropout |
| `activation` | — | `"relu"` | Activation function |
| `normalization` | — | `"batch"` | Normalisation layer |
| `in_channels` | — | `1` | Input channel count |
| `out_channels` | — | `6` | Output channel count ($3K$ for $K=2$ Gaussians) |

`LinkNetConfig` has no `bottleneck_factor` or `upsample_mode` field; upsampling is always via transposed convolution inside the decoder blocks.

---

## Paper fidelity

**Review date:** 2026-06-04
**Reference:** Chaurasia, A. & Culurciello, E. (2017). *LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation.* arXiv:1707.03718. (Fig. 1 network, Fig. 2 encoder module, Fig. 3 decoder module, Tables I–II block specs.) [[LinkNet_Chaurasia2017_1707.03718.pdf|PDF]]
**Code:** `models/LinkNet.py`, `configuration/models_config.py` (`LinkNetConfig`, builders).

**Overall verdict:** Faithful implementation. The signature LinkNet idea — additive (not concatenative) skips, added before each decoder block input exactly as in Fig. 1 — is reproduced correctly, and this is the contribution the architecture is named for. The stem ($k\times k$ stride-1, no max-pool), the single-residual-block encoder depth, and the generic-decoder-block-plus-$1\times1$-head final stage differ from the paper's ResNet-18 encoder, but per the **referee ruling of 2026-06-04** these are deliberate dense-prediction and depth-hyperparameter adaptations rather than fidelity defects, and will not be changed (see [Adaptation ruling](#adaptation-ruling) below).

### Verdict table

| # | Dimension | Paper | Code | Verdict |
|---|---|---|---|---|
| 1 | Initial stem | $7\times7$ conv **stride 2** $+$ $3\times3$ max-pool stride 2 ($/4$); Fig. 1, Sec. III | $k\times k$ conv **stride 1**, no max-pool ($/1$); `LinkNet.py:141-152` | **ACCEPTED ADAPTATION** (referee 2026-06-04) |
| 2 | Encoder block depth | **Two** ResNet-18 basic blocks per stage (stride-2 projection block $+$ stride-1 identity block); Fig. 2, Sec. III ("residual blocks [6]", "ResNet18 as its encoder") | **One** residual block per stage; identity sub-block absent; `LinkNet.py:23-72,154-167` | **ACCEPTED ADAPTATION** (referee 2026-06-04) |
| 3 | Conv–BN–ReLU ordering | Post-activation: conv $\to$ BN $\to$ ReLU; Sec. III | Same; `LinkNet.py:34-53`, `models_config.py:25-50` | **MATCH** |
| 4 | Decoder block layout | $1\times1\,(m\!\to\!m/4)\to$ full-conv $3\times3$ $/2\,(m/4\!\to\!m/4)\to1\times1\,(m/4\!\to\!n)$, BN$+$ReLU each; Fig. 3, Table II | Identical sequence and bottleneck ratio $c=4$; `LinkNet.py:76-119` | **MATCH** |
| 5 | Skip linking | $\text{input}(\text{dec}_i)=\text{out}(\text{dec}_{i+1})+\text{out}(\text{enc}_i)$, add **before** decoder input; Fig. 1 | Same add point; deepest decoder fed directly by deepest encoder, no add; `LinkNet.py:212-222` | **MATCH** |
| 6 | Final block | full-conv $3\times3$ $/2\,(64\!\to\!32)\to$ conv $3\times3\,(32\!\to\!32)\to$ full-conv $2\times2$ $/2\,(32\!\to\!N)$; Fig. 1 | Generic bottleneck decoder block $(64\!\to\!64)$ $+$ $1\times1$ head $(64\!\to\!C)$; `LinkNet.py:181-196` | **ACCEPTED ADAPTATION** (referee 2026-06-04) |
| 7 | ReLU around additions | ReLU after encoder residual add (Fig. 2); no ReLU on decoder skip circles (Fig. 1) | ReLU after encoder add (`LinkNet.py:72`); decoder add not followed by ReLU | **MATCH** |
| 8 | Output resolution recovery | Recover from stem's $/4$ via decoder up-samples $+$ final block's two $\times2$; Sec. III | Stem at $/1$, encoder to $/16$, four stride-2 ConvT recover $/16\!\to\!/1$; `match_spatial_size` bilinear safety net; `LinkNet.py:11-19,224` | **ACCEPTED ADAPTATION** |
| 9 | Up-sampling operator | Full-convolution (transposed conv); Sec. III–IV | `nn.ConvTranspose2d` $3\times3$, stride 2, `output_padding=1`; `LinkNet.py:97-105` | **MATCH** |
| 10 | Encoder channel widths | $64,128,256,512$ (Table I) | Hyperparameter `features`, default $[152,312,624,1248]$; ResNet preset $[64,128,256,512]$ available | Out of scope (hyperparameter) |

### Prose

The decoder is the most faithful part of the port. The `BottleneckDecoderBlock` (`LinkNet.py:76-119`) reproduces Fig. 3 exactly: a $1\times1$ compression to $m/4$, a full $3\times3$ transposed convolution at stride 2, and a $1\times1$ expansion to $n$, each followed by normalisation and activation. The additive skip wiring in the forward pass (`LinkNet.py:212-222`) matches Fig. 1 to the letter — the encoder output at level $i$ is summed onto the decoder output coming *down* from level $i+1$, and that sum is the *input* to decoder block $i$, with the deepest decoder block fed directly by the deepest encoder block. This is the core LinkNet contribution and it is correct, including the post-add ReLU placement in the encoder (`LinkNet.py:72`) consistent with Fig. 2.

Three structural choices distinguish this implementation from the paper's encoder, all of which the 2026-06-04 referee review ruled to be deliberate adaptations rather than fidelity defects. First, the stem (`LinkNet.py:141-152`) is a stride-1 convolution with no max-pool, whereas Sec. III specifies a $7\times7$ stride-2 convolution followed by a $3\times3$ stride-2 max-pool, i.e. a $/4$ reduction before the first encoder stage. Second, each encoder *stage* in the paper is a full ResNet-18 stage — two basic blocks, the first with a stride-2 projection shortcut and the second with a stride-1 identity shortcut (Fig. 2 shows the two stacked add circles); the code uses a single `ResidualEncoderBlock` per stage (`LinkNet.py:23-72,154-167`), so the encoder is a four-stage single-residual-block downsampler rather than a literal ResNet-18 backbone. Third, the final stage differs from the paper's dedicated three-layer head (full-conv $3\times3$ $/2$, conv $3\times3$, full-conv $2\times2$ $/2$; Fig. 1): the code appends one more generic bottleneck decoder block ($64\!\to\!64$) and a $1\times1$ output head (`LinkNet.py:181-196`), with a bilinear `match_spatial_size` (`LinkNet.py:11-19,224`) guaranteeing the output matches the input resolution. Because the code's stem does not downsample, the four stride-2 ConvT blocks already recover full resolution, so the resolution arithmetic closes.

#### Adaptation ruling

**Referee ruling (2026-06-04).** The reviewer assessed all three differences and ruled them justified, not open deviations to be fixed. The reasoning, recorded for the record:
- *Stem and resolution scheme.* The paper's $/4$ stem and ResNet-18 backbone target ImageNet-pretrained classification transfer for natural-image semantic segmentation. The TomoSAR task is single-channel dense regression of Gaussian-mixture parameters with no pretrained-weight pathway, so the stem's $/4$ down-sampling and the matching two-up-sample final block carry no benefit. Keeping the stem at stride 1 and recovering resolution through the four decoder up-samples is a deliberate dense-prediction adaptation. It will not be changed.
- *Encoder depth.* Collapsing each stage to a single residual block is a depth-hyperparameter choice, not a topological violation of the LinkNet design; the additive-link structure that defines LinkNet is preserved unchanged. The pretrained-ResNet-18 claim of the paper is inapplicable here for the reason above. This will not be changed.
- *Final block.* The generic-decoder-block-plus-$1\times1$-head final stage is the dense-regression counterpart of the paper's segmentation head and closes the resolution arithmetic given the stride-1 stem. It is a deliberate adaptation and will not be changed.

The additive (not concatenative) skip-link contribution — the defining LinkNet idea, wired exactly as in Fig. 1 (`LinkNet.py:212-222`) with the post-add ReLU placement of Fig. 2 (`LinkNet.py:72`) — is faithful and is the part that matters for this port.

---

## Related Notes

- [[UNet]] — Concatenative skip alternative
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — LinkNetConfig
- [[DLR-TomoSAR Index]] — Master index
