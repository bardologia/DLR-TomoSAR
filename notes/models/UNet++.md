# UNet++

`UNetPlusPlus` (`models/UNetPlusPlus.py`) extends the [[UNet]] skip connections to a nested, dense structure ([[UNetPlusPlus_Zhou2018_1807.10165.pdf|Zhou et al., 2018]]). Intermediate nodes between encoder and decoder re-aggregate features from multiple encoder levels, reducing the semantic gap between the contracting and expanding paths.

---

## Summary

In standard UNet, skip connection at level $l$ passes encoder features $\mathbf{s}_l$ directly to decoder level $l$. UNet++ introduces intermediate convolution nodes $\mathbf{x}^{i,j}$ (where $i$ is the encoder level and $j$ is the dense block index) that aggregate all preceding nodes at the same scale:

$$
\begin{aligned}
\mathbf{u} &= \text{Up}(\mathbf{x}^{i+1,j-1}) \\
\mathbf{c} &= \text{cat}\!\left[\mathbf{x}^{i,0}, \mathbf{x}^{i,1}, \dots, \mathbf{x}^{i,j-1},\; \mathbf{u}\right] \\
\mathbf{x}^{i,j} &= \text{ConvBlock}(\mathbf{c})
\end{aligned}
$$

The decoder at level $l$ receives $\mathbf{x}^{l, L-l}$ (the final dense node at that level) instead of the raw encoder skip.

The implementation (`UNetPlusPlus.__init__`) fixes the depth to exactly four encoder levels: `features` must contain exactly four channel sizes, raising `ValueError` otherwise. The five backbone nodes are `encoder_0_0` through `encoder_4_0`, where `encoder_4_0` produces the bottleneck of width $F_3 \cdot r$ with $r$ = `bottleneck_factor`. The dense nodes `dense_i_j` follow the indexing above.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{x}^{i,j}$ | Re-aggregated feature node at encoder level $i$, dense index $j$ |
| $\mathbf{x}^{i,0}$ | Original encoder output at level $i$ (backbone node) |
| $\mathbf{s}_l$ | Encoder feature map (raw skip) at level $l$ |
| $\mathbf{u}$ | Upsampled lower-level node $\mathbf{x}^{i+1,j-1}$ |
| $\mathbf{c}$ | Channel-wise concatenation of the $j$ same-level predecessors with $\mathbf{u}$ |
| $\hat{\boldsymbol{\theta}}^{(j)} \in \mathbb{R}^{B \times C_{\text{out}} \times P_H \times P_W}$ | Prediction from the $j$-th dense node at level 0 |
| $B$ | Batch size |
| $P_H, P_W$ | Output (patch) spatial dimensions |
| $i$ | Encoder level index |
| $j$ | Dense block index |
| $l$ | Decoder / skip level |
| $L$ | Total number of encoder levels |
| $F_i$ | `features[i]`, the level-$i$ channel width |
| $F_0$ | Level-0 channel width, `features[0]` (default: `56`) |
| $r$ | `bottleneck_factor`, bottleneck width multiplier for `encoder_4_0` |
| $C_{\text{out}}$ | `out_channels`, output channel count (default: `6`) |
| $\text{Up}(\cdot), \mathcal{U}$ | Bilinear or transposed-convolution upsampling |
| $\text{cat}, [\,\cdot\,]$ | Channel-wise concatenation |
| $\text{ConvBlock}, \mathcal{H}$ | Dense convolution block |
| $\text{Conv}_{1\times1,\, A \to B}$ | $1\times1$ output-head convolution from $A$ to $B$ channels |

---

## Architecture

### Deep Supervision

When `deep_supervision = True`, four separate 1×1 output heads are applied to the dense nodes $\mathbf{x}^{0,1}, \mathbf{x}^{0,2}, \mathbf{x}^{0,3}, \mathbf{x}^{0,4}$, and the forward pass returns a list of four prediction tensors:

$$
\{\hat{\boldsymbol{\theta}}^{(j)} = \text{Conv}_{1\times1,\, F_0 \to C_{\text{out}}}(\mathbf{x}^{0,j})\}_{j=1}^{4}
$$

When `deep_supervision = False` (the default), only $\mathbf{x}^{0,4}$ is passed through a single output head, returning one tensor.

---

## Design Rationale

**Semantic gap reduction.** In standard UNet, the encoder output $\mathbf{s}_l$ is directly concatenated with the upsampled decoder output. These two feature maps are at different semantic levels (low-level encoder features vs. high-level decoded features). UNet++ refines the skip features through a series of ConvBlocks, bridging the semantic gap before concatenation.

> **Semantic gap**
> The nested dense blocks refine encoder features through successive ConvBlocks so that, by the time they reach the decoder, they are semantically closer to the upsampled decoder features they are concatenated with.

---

## Parameter Reference

See [[Configuration Layer]] → `UNetPlusPlusConfig`. Parameters shared with [[UNet]] retain the same semantics, but the default `features` list differs.

| Parameter | Default | Description |
|---|---|---|
| `features` | `[56, 112, 216, 440]` | Encoder channel widths for the four levels (must have length 4) |
| `bottleneck_factor` | `2` | Multiplier for the bottleneck node `encoder_4_0` width |
| `deep_supervision` | `False` | When `True`, returns four predictions from nodes $\mathbf{x}^{0,1..4}$ |
| `dropout` | `0.15` | Per-ConvBlock dropout |
| `activation` | `"relu"` | Activation function |
| `normalization` | `"batch"` | Normalisation layer |
| `upsample_mode` | `"convtranspose"` | Upsampling mode |
| `in_channels` | `1` | Input channel count |
| `out_channels` | `6` | Output channel count ($3K$ for $K=2$ Gaussians) |

---

## Paper fidelity

**Review date:** 2026-06-04
**Citation:** Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, J. Liang. *UNet++: A Nested U-Net Architecture for Medical Image Segmentation.* arXiv:1807.10165, 2018. [[UNetPlusPlus_Zhou2018_1807.10165.pdf|PDF]]

Ground-truth reference is the paper PDF. The verification compares the implementation in `models/UNetPlusPlus.py` and `configuration/models_config.py` against Eq. 1 and Fig. 1 of the paper, node by node.

### Verdict table

| # | Dimension | Paper ref | Code ref | Verdict |
|---|---|---|---|---|
| 1 | Node wiring per Eq. 1 ($\mathbf{x}^{i,j}=\mathcal{H}([\,\mathbf{x}^{i,0..j-1},\,\mathcal{U}(\mathbf{x}^{i+1,j-1})\,])$) | Eq. 1, §3.1 | `UNetPlusPlus.py:142-168` | MATCH |
| 2 | Channel bookkeeping of each concatenation vs `ConvBlock` `input_channels` | Eq. 1, Fig. 1b | `UNetPlusPlus.py:100-112` | MATCH |
| 3 | Conv block $\mathcal{H}$ composition | §3.1 ("convolution followed by an activation") | `UNetPlusPlus.py:11-47` | ACCEPTED ADAPTATION |
| 4 | Downsampling between encoder column-0 nodes | Fig. 1a (down-sampling arrows) | `UNetPlusPlus.py:82, 136-140` | MATCH |
| 5 | Upsampling operator $\mathcal{U}$ | Eq. 1, Fig. 1a/1b | `UNetPlusPlus.py:84-90, 127-132` | MATCH |
| 6 | Deep supervision: heads on $\mathbf{x}^{0,1..0,4}$, accurate (average) vs fast (pruned) modes | §3.2, §4, Fig. 1c | `UNetPlusPlus.py:114-117, 171-173` | DEVIATION (minor) |
| 7 | Output head ($1\times1$ conv) | §4 | `UNetPlusPlus.py:114-123, 172, 175` | ACCEPTED ADAPTATION |
| 8 | Depth / levels structure (triangular grid $i\in\{0..4\}$) | Fig. 1a | `UNetPlusPlus.py:93-112` | MATCH |
| 9 | Dense skip-pathway convs ($j+1$ convs along top pathway) | §3.1, Fig. 1b | `UNetPlusPlus.py:99-112` | MATCH |
| 10 | Bottleneck node $\mathbf{x}^{4,0}$ width | Fig. 1a, Table 2 | `UNetPlusPlus.py:76, 97` | ACCEPTED ADAPTATION |

**Overall verdict:** Faithful implementation. The nested dense-skip topology of Eq. 1 and Fig. 1 is reproduced exactly at every node; the only true deviation is the deep-supervision inference-mode handling.

### Wiring and channel verification (Eq. 1)

Every instantiated node was checked against $\mathbf{x}^{i,j}=\mathcal{H}([\,\mathbf{x}^{i,0},\dots,\mathbf{x}^{i,j-1},\,\mathcal{U}(\mathbf{x}^{i+1,j-1})\,])$. Each dense node receives precisely all $j$ same-level predecessors plus the upsampled lower-level node $\mathbf{x}^{i+1,j-1}$, with no missing or spurious inputs:

- $j=1$: `node_0_1` $\leftarrow[\mathbf{x}^{0,0},\mathcal{U}(\mathbf{x}^{1,0})]$, and analogously `node_1_1`, `node_2_1`, `node_3_1` (`:144-150`).
- $j=2$: `node_0_2` $\leftarrow[\mathbf{x}^{0,0},\mathbf{x}^{0,1},\mathcal{U}(\mathbf{x}^{1,1})]$, plus `node_1_2`, `node_2_2` (`:154-158`).
- $j=3$: `node_0_3` $\leftarrow[\mathbf{x}^{0,0},\mathbf{x}^{0,1},\mathbf{x}^{0,2},\mathcal{U}(\mathbf{x}^{1,2})]$, plus `node_1_3` (`:162-164`).
- $j=4$: `node_0_4` $\leftarrow[\mathbf{x}^{0,0},\mathbf{x}^{0,1},\mathbf{x}^{0,2},\mathbf{x}^{0,3},\mathcal{U}(\mathbf{x}^{1,3})]$ (`:168`).

The concatenation channel counts ($j\cdot F_i + F_{i+1}$, where $F_4 = F_3\cdot r$ is the bottleneck width) match the declared `ConvBlock` `input_channels` term by term (`:100-112`), since every node at level $i$ — encoder and dense alike — emits exactly $F_i$ channels. This is a full match, including the $j+1$-input rule for $j>1$ stated in §3.1.

### Accepted adaptations

- **Conv block $\mathcal{H}$ (#3).** The paper defines $\mathcal{H}(\cdot)$ loosely as "a convolution operation followed by an activation function." The implementation uses the standard two-conv block with normalisation: Conv–Norm–Act–Conv–Norm–Act (`:22-41`), matching the authors' own reference implementation and Fig. 1b (which shows each node as one dense conv block on the pathway). BatchNorm and the optional `Dropout2d` are modernisations not present in the 2018 text but consistent with its intent. Justified.
- **Output head (#7).** The paper appends a $1\times1$ conv followed by a **sigmoid** to each target node for binary segmentation (§4). The TomoSAR task is regression of Gaussian-mixture parameters ($C_{\text{out}}=3K$), so the sigmoid is correctly dropped and a bare $1\times1$ conv is used (`:114-123`). Domain-justified adaptation.
- **Bottleneck width (#10).** Fig. 1a / Table 2 use a strict power-of-two pyramid ($F_4 = 2 F_3$). The code parameterises this as $F_4 = F_3\cdot r$ with `bottleneck_factor` $r$ (default $2$), recovering the paper's ratio at the default while remaining configurable. Justified.

### Deviations

- **Deep-supervision inference modes (#6) — severity: minor.** The paper (§3.2, Fig. 1c) specifies two inference modes when trained with deep supervision: *accurate mode* averages the outputs of all four segmentation branches $\{\mathbf{x}^{0,j}\}_{j=1}^{4}$, and *fast mode* selects a single pruned branch $\mathbf{x}^{0,i}$ (UNet++ $L^i$). The implementation correctly attaches four $1\times1$ heads to $\mathbf{x}^{0,1..0,4}$ (`:114-117`) and returns the raw list of four predictions (`:171-173`), but the model itself realises **neither** mode: it does not average the branches (accurate) nor expose a pruned single-branch path (fast). No averaging of the branch list was found downstream (loss/inference layers operate per-element, not as a branch ensemble). Note also that fast-mode pruning is purely an inference-time graph trim; the current build always executes the full graph.
  - *Proposed fix:* in `forward`, when `deep_supervision` is `True` and the module is in `eval()` mode, return the element-wise mean of the four head outputs (accurate mode); during training continue returning the list so each branch is supervised. Optionally add a `prune_level: int | None` config field that, when set, restricts the forward pass to nodes feeding $\mathbf{x}^{0,\text{prune\_level}}$ and returns that single head (fast mode), matching Fig. 1c.

### Notes on unused config helper

`build_upsample` (`models_config.py:53-75`) defines a bilinear branch with a trailing $1\times1$ conv, but `UNetPlusPlus` uses its own internal `nn.Upsample` (`:90`) and per-width `ConvTranspose2d` modules (`:86-88`); the builder is not on this model's path, so the difference is inert and not a fidelity issue.

---

## Related Notes

- [[UNet]] — Base architecture
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — UNetPlusPlusConfig
- [[DLR-TomoSAR Index]] — Master index
