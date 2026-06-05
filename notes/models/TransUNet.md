# TransUNet

`TransUNet` (`models/TransUNet.py`) fuses a Vision Transformer (ViT) patch encoder with a CNN U-Net decoder ([[TransUNet_Chen2021_2102.04306.pdf|Chen et al., 2021]]). The CNN backbone first extracts feature maps; the ViT processes flattened patch tokens from the deepest CNN feature map; the decoder recovers spatial resolution.

---

## Summary

TransUNet applies a CNN encoder to produce multi-scale feature maps. The deepest feature map is tokenised into non-overlapping patches and processed by a standard ViT with multi-head self-attention. The ViT output tokens are reshaped back into a spatial feature map and fed to a CNN decoder with skip connections from the CNN encoder.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{f}$ | Bottleneck feature map, $\mathbb{R}^{B \times C_b \times H_f \times W_f}$ |
| $\mathbf{z}_n$ | Token $n$, $\mathbb{R}^d$ |
| $\mathbf{e}_n^{\text{pos}}$ | Learnable position embedding, $\mathbb{R}^d$, bilinearly interpolated when the actual grid size differs from `image_size // 2^{len(cnn_features)} // patch_size` |
| $\mathbf{Z}$ | Input token sequence to a transformer layer, $\mathbb{R}^{N \times d}$ |
| $\hat{\mathbf{Z}}$ | Intermediate sequence after attention + residual |
| $\mathbf{Z}'$ | Transformer layer output sequence |
| $B$ | Batch size |
| $H_f, W_f$ | Spatial dimensions of the bottleneck feature map |
| $N$ | Number of patch tokens, $(H_f / p) \cdot (W_f / p)$ |
| $C_b$ | Bottleneck channel count, equal to `cnn_features[-1]` times `bottleneck_factor` |
| $d$ | Transformer embedding dimension, equal to $C_b$ |
| $p$ | `patch_size` (default `1`; with $p = 1$ each spatial location is one token) |
| $\text{MSA}$ | Multi-head self-attention (full, no window restriction) |
| $\text{MLP}$ | Feed-forward network |
| $\text{LN}$ | Layer normalisation |

---

## Architecture

### ViT Patch Tokenisation

The CNN encoder produces multi-scale feature maps; a pre-transformer `ConvBlock` then maps the deepest feature map (`cnn_features[-1]` channels) to the bottleneck width $C_b$, given by `cnn_features[-1]` times `bottleneck_factor`. Tokenisation (`PatchEmbedding`) is a strided convolution with kernel = stride = $p$, so the token embedding dimension equals the bottleneck channel count $d = C_b$ (the implementation passes `embedding_dim = bottleneck_channels`).

Given bottleneck feature map $\mathbf{f} \in \mathbb{R}^{B \times C_b \times H_f \times W_f}$:

1. Partition into $N = (H_f / p) \cdot (W_f / p)$ non-overlapping patches of size $p \times p$.
2. Project each patch and add the position embedding:

$$
\mathbf{z}_n = \text{Conv}_{C_b \to d,\, p\times p,\, s=p}(\mathbf{f})[n] + \mathbf{e}_n^{\text{pos}}
$$

### Transformer Layer

$$
\hat{\mathbf{Z}} = \text{MSA}(\text{LN}(\mathbf{Z})) + \mathbf{Z}
$$

$$
\mathbf{Z}' = \text{MLP}(\text{LN}(\hat{\mathbf{Z}})) + \hat{\mathbf{Z}}
$$

---

## Design Rationale

> **Hybrid motivation.** TransUNet retains CNN feature extraction for local structure while adding global attention at the bottleneck, typically outperforming both pure CNN and pure transformer baselines.

Pure ViT models (e.g., [[UNETR]]) lack inductive biases for local structure that CNNs naturally capture. TransUNet retains CNN feature extraction (low-level spatial filters) while adding global attention at the bottleneck. This hybrid approach typically outperforms both pure CNN and pure transformer baselines on dense prediction tasks at moderate resolution.

**Limitation.** The ViT operates on the deepest CNN feature map, so global attention is applied only at the coarsest scale. Long-range dependencies at fine scales are not directly modelled.

---

## Parameter Reference

See [[Configuration Layer]] → `TransUNetConfig`. The transformer embedding dimension is not configured directly; it is derived as `cnn_features[-1] * bottleneck_factor`.

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `image_size` | — | `256` | Expected input patch side length (sets positional grid) |
| `cnn_features` | $F_l$ | `[32, 72, 136, 272]` | CNN encoder channel widths per level |
| `bottleneck_factor` | — | `2` | Multiplier giving bottleneck/embedding dim $C_b$ |
| `transformer_layers` | — | `6` | Number of transformer blocks |
| `transformer_heads` | — | `4` | Attention heads per block |
| `transformer_mlp_ratio` | — | `4.0` | FFN hidden expansion ratio |
| `patch_size` | $p$ | `1` | Tokenisation kernel/stride on the bottleneck map |
| `attention_dropout` | — | `0.0` | Dropout on attention weights |
| `stochastic_depth_rate` | — | `0.0` | Maximum DropPath rate across blocks |
| `dropout` | — | `0.15` | ConvBlock and FFN dropout |
| `activation` | — | `"relu"` | CNN activation |
| `ffn_activation` | — | `"gelu"` | Transformer FFN activation |
| `normalization` | — | `"batch"` | CNN normalisation layer |
| `upsample_mode` | — | `"convtranspose"` | Decoder upsampling mode |
| `in_channels` | — | `1` | Input channel count |
| `out_channels` | — | `6` | Output channel count ($3K$ for $K=2$ Gaussians) |

With the defaults, $d = C_b = 272 \cdot 2 = 544$, and `transformer_heads = 4` divides $544$.

---

## Paper fidelity

**Review date:** 2026-06-04
**Citation:** Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A. L., & Zhou, Y. (2021). *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*. arXiv:2102.04306. (Ground-truth PDF: [[TransUNet_Chen2021_2102.04306.pdf]], Section 3, Fig. 1, Eq. 1-3.)

**Overall verdict:** Faithful. The structural skeleton the paper defines — CNN-first hybrid encoder, patch embedding on the CNN feature map, pre-norm transformer with the Eq. 2-3 residual structure and a final $\text{LN}$, token-to-2D reshape, cascaded upsampler with skips drawn exclusively from the CNN encoder — is reproduced correctly. The two decoder items previously flagged were resolved by referee ruling on 2026-06-04: the "missing initial $D\to512$ channel-reduction $3\times3$ conv" was REFUTED, because that channel reduction is folded into the first upsampling operator and the paper's block order is preserved (now item 6a: MATCH); and the extra full-resolution skip was ruled a documented benign consequence of the configurable CNN stem rather than a deviation (now item 7: accepted adaptation). The remaining minor item is the richer decoder block (double-conv + norm versus the paper's single conv + ReLU), an accepted U-Net-style substitution. The CNN backbone (plain double-conv stack versus ResNet-50) and the upsample operator are explicitly hyperparameters and out of scope.

| # | Component | Paper ref | Code | Verdict |
|---|---|---|---|---|
| 1 | Hybrid order (CNN then transformer) | §3.2 "CNN-Transformer Hybrid as Encoder" | `TransUNet.py:275-304` | MATCH |
| 1b | CNN backbone type | §3.2, §4.2 (ResNet-50) | `TransUNet.py:187-202` (double-conv stack) | ACCEPTED ADAPTATION (backbone is hyperparameter) |
| 1c | Patch embedding on CNN feature map, $1\times1$ patches | §3.2 "$1\times1$ patches extracted from the CNN feature map" | `TransUNet.py:213-217`, `patch_size=1` default `models_config.py:720` | MATCH |
| 2 | Trainable linear projection (Eq. 1) | Eq. 1, $E \in \mathbb{R}^{(P^2\cdot C)\times D}$ | `PatchEmbedding` Conv2d k=s=p `TransUNet.py:137-142` | MATCH (strided conv = patch linear projection) |
| 2b | Learned positional embedding added, no class token | Eq. 1, $E_{pos}\in\mathbb{R}^{N\times D}$ | `TransUNet.py:237-240,299` | MATCH |
| 2c | Extra $\text{LN}$ inside patch embedding | not in Eq. 1 | `TransUNet.py:143,149` | DEVIATION (minor) |
| 3 | Pre-norm MSA + residual (Eq. 2) | Eq. 2 | `TransUNet.py:120-121` | MATCH |
| 3b | Pre-norm MLP + residual (Eq. 3) | Eq. 3 | `TransUNet.py:122-123` | MATCH |
| 3c | Final $\text{LN}$ after layer $L$ | $\mathbf{z}_L$ encoded representation | `TransUNet.py:232,304` | MATCH |
| 4 | MLP with GELU, ratio 4 | Eq. 3, ViT MLP | `TransUNet.py:110-117`, `ffn_activation="gelu"` | MATCH |
| 5 | Reshape tokens $HW/P^2 \to H/P\times W/P$ | §3.2 | `tokens_to_feature_map` `TransUNet.py:306` | MATCH |
| 6a | Initial $3\times3$ conv reshaping $D\to512$ before cascade | Fig. 1(b), §3.2 | channel reduction folded into the first upsampling operator `TransUNet.py:246-252` | MATCH (referee ruling 2026-06-04: paper block order preserved) |
| 6b | CUP block op order: 2x upsample -> $3\times3$ conv -> ReLU | §3.2 "Cascaded Upsampler" | upsample then ConvBlock `TransUNet.py:315-318` | MATCH (order preserved) |
| 6c | CUP block internal composition | §3.2 (single conv + ReLU) | double-conv + norm `ConvBlock` `TransUNet.py:11-47` | ACCEPTED ADAPTATION (U-Net-style block) |
| 7 | Skip connections from CNN at $1/2,1/4,1/8$, none from transformer | Fig. 1(b) | 4 skips (incl. full-res), all from CNN `TransUNet.py:277,312-314` | ACCEPTED ADAPTATION (referee ruling 2026-06-04: benign stem consequence) |
| 8 | Segmentation head | Fig. 1(b) | $1\times1$ conv `TransUNet.py:264-268` | MATCH |
| 9 | Patch grid $H/16$ | §3.2, §4.2 ($P=16$ via ResNet $1/16$) | `cnn_downsample=2^4=16`, grid $H/16$ `TransUNet.py:234-235` | MATCH |
| 10 | Penultimate $(16,H,W)$ stage then head | Fig. 1(b) | last decoder level `cnn_features[0]` channels at full res `TransUNet.py:264-265` | MATCH |
| -- | Upsample operator type | Fig. 1 (bilinear 2x) | `convtranspose` default, `bilinear` available `models_config.py:724,53-75` | ACCEPTED ADAPTATION (hyperparameter) |

The transformer core (Eq. 1-3, pre-norm, final $\text{LN}$, GELU MLP, learned position embedding without class token) is an exact match to the paper. The hybrid ordering and the defining $1\times1$-patch-on-CNN-feature design (§3.2) are honoured, and skip connections originate solely from the CNN encoder with no transformer-sourced skip, as the paper requires.

Two items previously listed here as open deviations were closed by referee ruling on 2026-06-04, leaving a single genuine deviation in the embedding plumbing rather than the conceptual architecture:

1. **Initial channel-reduction conv — REFUTED (referee ruling 2026-06-04; Fig. 1(b), §3.2; `TransUNet.py:246-252`).** The earlier claim was that the paper's "reshape $\mathbf{z}_L$ to $(D, H/16, W/16)$ -> $3\times3$-conv + ReLU to $512$ channels *before* the first upsampling step" was absent. The referee found this is not a deviation: the channel reduction is folded into the first upsampling operator (`build_upsample(reversed_features[0] -> reversed_features[1])`), which reduces $D$ to the first decoder width as it upsamples, matching the paper's block order (reshape -> reduce -> upsample-then-conv cascade). No fix required. Recorded as item 6a: MATCH.

2. **Extra full-resolution skip — benign stem consequence (referee ruling 2026-06-04; Fig. 1(b); `TransUNet.py:277,312-314`).** Skips are appended for every CNN level before pooling, yielding 4 skips with the finest at full input resolution ($1/1$) against the paper's 3 at $1/2,1/4,1/8$. The referee ruled this a documented benign consequence of the generic, depth-configurable CNN stem (structurally still all-CNN, none from the transformer), not an open deviation requiring a fix. Recorded as item 7: accepted adaptation.

3. **Extra $\text{LN}$ in patch embedding (`TransUNet.py:143,149`).** Eq. 1 maps patches by linear projection and immediately adds $E_{pos}$ with no normalisation; the code applies a `LayerNorm` to the projected tokens before the positional embedding is added. *Severity: minor (the sole remaining open deviation). Proposed fix:* drop the `self.norm` in `PatchEmbedding`, or move normalisation to match a recognised ViT variant, if exact Eq. 1 fidelity is required.

The `ConvBlock` double-conv-with-norm decoder block, the ResNet-50-vs-double-conv backbone, the `convtranspose` default upsampler, and `DropPath` regularisation are accepted adaptations: the first is a standard U-Net decoder substitution preserving the paper's upsample-then-conv ordering, and the remainder are explicitly hyperparameters (backbone choice, upsample mode, stochastic depth) outside the structural scope.

---

## Related Notes

- [[Model Zoo]] — Architecture comparison
- [[SwinUNet]] — Alternative hierarchical transformer architecture
- [[UNETR]] — Pure transformer encoder
- [[UNet]] — CNN-only baseline
- [[Configuration Layer]] — TransUNetConfig
- [[DLR-TomoSAR Index]] — Master index
