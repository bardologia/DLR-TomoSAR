# MultiResUNet

`MultiResUNet` (`models/MultiResUNet.py`) replaces the U-Net's double-conv blocks with MultiRes blocks — chained 3×3 convolutions whose intermediate outputs are concatenated, approximating parallel 3×3/5×5/7×7 inception branches at a fraction of the cost — and filters each skip connection through a residual "ResPath" ([[MultiResUNet_Ibtehaz2020_1902.04049.pdf|Ibtehaz & Rahman, 2020]]).

---

## Summary

The macro-topology is identical to [[UNet]] (MaxPool down, transposed-conv up, skip concatenation). Each block is a `MultiResBlock`; each skip passes through a `ResPath` whose length decreases with depth (`n_levels − i` residual conv units), compensating the semantic gap between shallow encoder features and deep decoder features.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | Block input feature map |
| $\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3$ | Outputs of the three chained convolutions (receptive fields 3, 5, 7 px) |
| $\mathbf{s}$ | $1\times1$ shortcut of $\mathbf{x}$ |
| $\mathbf{c}$ | Normalized concatenation $\text{Norm}([\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3])$ |
| $\mathbf{y}$ | Block output |
| $\text{Norm}$ | Normalization (batch norm) |
| $\text{Act}$ | Activation (ReLU) |
| $\text{Conv}_{k\times k}$ | $k\times k$ convolution |
| $f_1, f_2, f_3$ | The three chained $3\times3$ convolution units |
| $W$ | MultiRes block width, $W = \alpha U$ |
| $U$ | Base encoder block width (32/64/128/256) |
| $\alpha$ | Width scaling factor ($\alpha = 1.67$) |
| $c_1, c_2, c_3$ | Channel split of $W$: $\lfloor W/6\rfloor$, $\lfloor W/3\rfloor$, $W - c_1 - c_2$ |
| $n_{\text{levels}}$ | Number of encoder levels |
| $i$ | Encoder level index |

---

## Architecture

### MultiRes Block

Three chained convolutions with channel split $\left(\lfloor\tfrac{W}{6}\rfloor, \lfloor\tfrac{W}{3}\rfloor, W - \lfloor\tfrac{W}{6}\rfloor - \lfloor\tfrac{W}{3}\rfloor\right)$ of the block width $W$ — a paper-faithful $1{:}2{:}3$ progression in which the third conv absorbs the floor remainder so the three terms sum to exactly $W$:

$$
\begin{aligned}
\mathbf{u}_1 &= f_1(\mathbf{x}) \\
\mathbf{u}_2 &= f_2(\mathbf{u}_1) \\
\mathbf{u}_3 &= f_3(\mathbf{u}_2) \\
\mathbf{s} &= \text{Conv}_{1\times1}(\mathbf{x}) \\
\mathbf{c} &= \text{Norm}([\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3]) \\
\mathbf{y} &= \text{Act}(\mathbf{c} + \mathbf{s})
\end{aligned}
$$

$\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3$ have effective receptive fields of 3, 5, and 7 pixels, so the concatenation sees three spatial scales simultaneously.

### ResPath

Skip connections at level $i$ pass through $n_{\text{levels}} - i$ units of $\text{Act}(\text{Conv}_{3\times3}(\mathbf{x}) + \text{Conv}_{1\times1}(\mathbf{x}))$ — shallow skips (largest semantic gap) get the most processing.

---

## Design Rationale

**Multi-scale within the block.** Tomographic structures span scales — single bright scatterers, building facades, extended layover regions. The MultiRes block gives every level explicit access to three receptive-field sizes, where standard blocks rely on depth alone. The ResPath addresses a known U-Net weakness: raw shallow features are too low-level to concatenate directly into a nearly-final decoder representation.

---

## Parameter Reference

See [[Configuration Layer]] → `MultiResUNetConfig`. All base UNet-style parameters apply.

---

## Paper fidelity

**Review date:** 2026-06-04
**Reference:** Ibtehaz, N. & Rahman, M. S. (2020). *MultiResUNet: Rethinking the U-Net Architecture for Multimodal Biomedical Image Segmentation.* Neural Networks, 121, 74–87. arXiv:1902.04049. Ground truth: Section 3 (Proposed Methodology), Section 4 (Proposed Architecture, Eq. 1), Fig. 3–5, Table 1. [[MultiResUNet_Ibtehaz2020_1902.04049.pdf|PDF]]

Equation-by-equation, figure-by-figure verification of `models/MultiResUNet.py` and the `MultiResUNetConfig` builder (`configuration/models_config.py:1339`) against the paper. Hyperparameters (filter counts, $\alpha$, dropout, learning rates) are out of scope; the channel-allocation *ratio scheme* is treated as a paper design point.

### Verdict table

| # | Component | Paper reference | Code | Verdict |
|---|---|---|---|---|
| 1 | MultiRes block: three chained $3\times3$ convs, intermediate outputs concatenated, $1\times1$ shortcut added | §3.1, Fig. 3c; §4 | `MultiResBlock` `MultiResUNet.py:10-44` | MATCH |
| 1b | BN/activation ordering around the add | Fig. 3c + released Keras impl | `MultiResUNet.py:41-42` | ACCEPTED ADAPTATION |
| 2 | Channel allocation $\lfloor W/6\rfloor,\lfloor W/3\rfloor,\lfloor W/2\rfloor$ | §4, Eq. 1, Table 1 | `MultiResUNet.py:13-15` | ACCEPTED ADAPTATION |
| 3 | ResPath: chain of $3\times3$ conv + parallel $1\times1$ shortcut, summed then activated | §3.2, Fig. 4 | `ResPath` `MultiResUNet.py:47-66` | MATCH |
| 4 | ResPath lengths 4, 3, 2, 1 by depth | §4, Table 1 | `MultiResUNet.py:90` | MATCH |
| 5 | Encoder downsampling via $2\times2$ max pooling | §2, §4 | `MultiResUNet.py:89` | MATCH |
| 6 | Decoder upsampling via transposed conv + skip concat | §2, §4, Fig. 5 | `MultiResUNet.py:99-106,121-125` | MATCH |
| 7 | Five levels, filters double per level, $W=\alpha U$ scaling | §4, Eq. 1 | `MultiResUNet.py:79-95`; config | ACCEPTED ADAPTATION |
| 8 | Output head: $1\times1$ conv, Sigmoid | §2, §4 | `MultiResUNet.py:108,127` | DEVIATION (minor) |
| 9 | Res paths on all skips; bridge/bottleneck excluded | Fig. 5, Table 1 | `MultiResUNet.py:90,114-119` | MATCH |
| 10 | ResPath width = $U$ (32/64/128/256), block width $\approx W$ | Table 1 | `MultiResUNet.py:88-90` | MATCH |

### Prose

**MultiRes block (MATCH, with one accepted ordering adaptation).** Fig. 3c and §3.1 specify three successive $3\times3$ conv blocks with the second and third approximating $5\times5$ and $7\times7$ receptive fields; their outputs are concatenated and a parallel $1\times1$ shortcut is added to the concatenation. The code reproduces this exactly: `out1,out2,out3` are computed sequentially (`MultiResUNet.py:37-39`), concatenated and added to `shortcut(x)` (`:41-42`). On the BN/activation ordering: the paper's *released Keras implementation* applies conv→BN per branch, concatenates, applies a BN on the concat, adds the (BN'd) $1\times1$ shortcut, then ReLU, then a final BN. The code does conv→BN→ReLU per branch (`:17-22`), BN on the concat (`concat_norm`, `:28,41`), adds the BN'd shortcut (`:29-32`), then ReLU (`:42`). This omits only the trailing post-activation BN of the released code and inserts intra-branch ReLUs; the placement matching the paper's *text* ("all convolutional layers ... are batch-normalized" and ReLU-activated) is faithful. Classified ACCEPTED ADAPTATION — the add-then-activate residual structure and the normalization-before-add are preserved; the dropped final BN is a benign, common simplification.

**Channel allocation (ACCEPTED ADAPTATION; the 2026-06-04 correction is faithful).** §4 assigns $\lfloor W/6\rfloor$, $\lfloor W/3\rfloor$, $\lfloor W/2\rfloor$ filters to the three convs, with $W=\alpha U$, $\alpha=1.67$. Crucially, the paper's true block output width is the **sum** of those three terms, which is slightly *below* $W$ (e.g. for $U=32$, $W=53$, the split is $8,17,26$ summing to $51$ — exactly Table 1's MultiRes Block 1 entries $8/17/26$ with the $1\times1$ shortcut at $51$). All five Table 1 rows were reproduced numerically ($8/17/26{\to}51$, $17/35/53{\to}105$, $35/71/106{\to}212$, $71/142/213{\to}426$, $142/284/427{\to}853$). The corrected code (`c1 = W//6`, `c2 = W//3`, `c3 = W - c1 - c2`, `:13-15`) applies the **same** $1{:}2{:}3$ ratio but fixes the total to `output_channels` exactly, letting $c_3$ absorb the floor remainder ($c_3$ is larger by 1 channel than $\lfloor W/2\rfloor$). This is a faithful rendering of the ratio scheme: the per-conv proportions are identical, the only difference is that the code treats the requested width as the realized block output rather than letting it float a couple channels below $W$. The earlier $1/4,1/4,1/2$ split was a genuine deviation (wrong ratio, equal first two convs); the 2026-06-04 change to floor-based $1/6,1/3,1/2$ corrects it to the paper's gradually-increasing $1{:}2{:}3$ progression. Verdict: ACCEPTED ADAPTATION.

**ResPath (MATCH).** §3.2 and Fig. 4 specify a chain of units, each $\text{ReLU}(\text{Conv}_{3\times3}(x)+\text{Conv}_{1\times1}(x))$ with the $1\times1$ on the residual branch. `ResPath.forward` (`:63-66`) does exactly $\text{Act}(\text{conv}(x)+\text{shortcut}(x))$ per unit, both convs BN'd (`:53-60`). Lengths $4,3,2,1$ from shallow to deep (Table 1, Res Path 1–4) are realized as `n_levels - index` (`:90`) which for the default 4-level config yields $4,3,2,1$. ResPath width equals the encoder block width $U$ (32/64/128/256 in Table 1), matching `feature_size` passed at `:90`. MATCH.

**Encoder / decoder / topology (MATCH).** Five-level encoder–decoder (four downsampling stages plus bottleneck), $2\times2$ max pooling (`:89`), transposed-conv upsampling (default `convtranspose`, `:99-106`), skip concatenation in the decoder (`:124`). Filters double per level and $W$ doubles after each pool/deconv per §4 — realized through `config.features` and the bottleneck factor (`:79-95`). Res paths apply to all four encoder skips and the bridge (bottleneck `MultiResBlock` at `:93`) is correctly excluded from any ResPath, matching Fig. 5 / Table 1. MATCH.

**Output head (DEVIATION, minor).** §2 and §4 state the final $1\times1$ conv is activated by **Sigmoid** (binary medical segmentation in $[0,1]$). The code's `output_head` is a bare $1\times1$ `Conv2d` with no activation (`:108,127`). This is an intentional and correct adaptation for the TomoSAR regression task (Gaussian-mixture parameter outputs, `out_channels=6`, `params_per_gaussian=3`), where a Sigmoid would be wrong — but relative to the paper as ground truth it is a deviation. Severity: minor (task-driven, not an architecture-fidelity error in the body of the network). No fix proposed; activation belongs in the task head / loss, not the backbone.

**Note on $\alpha$ and `features`.** The paper fixes $\alpha=1.67$ and derives $W$ from $U\in\{32,64,128,256,512\}$. The code exposes `features` directly (default `[64,128,256,512]`) as the realized block widths and a `bottleneck_factor`, i.e. it absorbs $\alpha$ into the chosen widths rather than computing $W=\alpha U$ at runtime. This is a configuration-surface choice (hyperparameter, out of scope) and does not alter the architecture.

### Overall

Architecture is a faithful rendering of MultiResUNet. All structural components — MultiRes block residual concat structure, $1{:}2{:}3$ channel ratio, ResPath residual chains with depth-decreasing lengths $4/3/2/1$, max-pool down / transposed-conv up topology, and exclusion of the bridge from ResPaths — match the paper and Table 1. The only deviations are task-driven and benign: the output head omits Sigmoid (correct for regression), and the MultiRes block omits the released implementation's trailing BN. The 2026-06-04 channel-split correction ($1/4,1/4,1/2 \to$ floor-based $1/6,1/3,1/2$) brought the block into agreement with §4 and Table 1 and is confirmed faithful.

---

## Related Notes

- [[UNet]] — Shared topology
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — MultiResUNetConfig
- [[DLR-TomoSAR Index]] — Master index
