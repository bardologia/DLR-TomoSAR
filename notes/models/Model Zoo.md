# Model Zoo

This note summarises all model architectures available in DLR-TomoSAR, their key properties, and guidance on when to prefer each.

---

## Architecture Summary

| Key | Note | Backbone | Skip Mechanism | Output Head | Parameters |
|---|---|---|---|---|---|
| `unet` | [[UNet]] | CNN encoder-decoder | Direct concatenation | Single 1×1 conv | ~31.0M |
| `attention_unet` | [[Attention UNet]] | CNN encoder-decoder | Attention-gated concat | Single 1×1 conv | ~32.4M |
| `unetplusplus` | [[UNet++]] | CNN encoder-decoder | Nested dense skips | Single 1×1 conv | ~31.1M |
| `resunet` | [[ResUNet]] | Residual CNN encoder-decoder (stride-2) | Skip + residual | Single 1×1 conv | ~32.4M |
| `unet_skip` | [[UNet Skip]] | Residual CNN encoder-decoder (MaxPool) | Skip + residual | Single 1×1 conv | ~32.4M |
| `linknet` | [[LinkNet]] | ResNet encoder | Additive skip | Single 1×1 conv | ~31.0M |
| `swin_unet` | [[SwinUNet]] | Swin Transformer | Hierarchical Swin | 1×1 conv | ~28.8M |
| `transunet` | [[TransUNet]] | CNN + ViT hybrid | Transformer patch tokens | CNN decoder | ~30.6M |
| `unetr` | [[UNETR]] | Pure ViT | Transformer skip outputs | CNN decoder | ~34.4M |
| `unet_multihead` | [[UNet Multihead]] | CNN encoder-decoder | Direct concatenation | 3 PixelMLP heads | ~31.0M |
| `unet_pergaussian` | [[UNet Per-Gaussian]] | CNN encoder-decoder | Direct concatenation | $K$ PixelMLP heads | ~31.0M |
| `resunet_multihead` | [[ResUNet Multihead]] | Residual CNN encoder-decoder | Skip + residual | 3 PixelMLP heads | ~32.4M |
| `resunet_pergaussian` | [[ResUNet Per-Gaussian]] | Residual CNN encoder-decoder | Skip + residual | $K$ PixelMLP heads | ~32.4M |
| `deeplabv3plus` | [[DeepLabV3+]] | Residual CNN, output stride 8 | Low-level fusion in decoder | ASPP + light decoder | ~10.3M |
| `segformer` | [[SegFormer]] | Hierarchical transformer (MiT-style) | Pyramid to all-MLP decoder | 1×1 conv | ~5.2M |
| `convnext_unet` | [[ConvNeXt UNet]] | ConvNeXt blocks in U-Net | Direct concatenation | Single 1×1 conv | ~19.1M |
| `dense_unet` | [[DenseUNet]] | FC-DenseNet (Tiramisu) | Dense concat + U skips | Single 1×1 conv | ~1.0M |
| `hrnet` | [[HRNet]] | Parallel multi-resolution branches | Repeated branch fusion | Concat + 1×1 conv | ~3.1M |
| `multires_unet` | [[MultiResUNet]] | MultiRes blocks in U-Net | ResPath-filtered concat | Single 1×1 conv | ~14.4M |
| `fpn` | [[FPN]] | Residual bottom-up + FPN top-down | Lateral 1×1 additions | Summed pyramid + 1×1 conv | ~7.8M |
| `u2net` | [[U2-Net]] | Nested RSU mini-U-Nets | Direct concatenation | Single 1×1 conv | ~8.3M |

Parameter counts are approximate for default configurations with $C_{\text{in}} = 1$ (`in_channels`) and $C_{\text{out}} = 6$ (`out_channels`), corresponding to $K = C_{\text{out}} / \text{params\_per\_gaussian} = 6/3 = 2$ Gaussian components. The [[Capacity Matching]] stage of the [[Benchmark Pipeline]] rescales all of them to the UNet reference budget before benchmarking.

---

## Selection Guidance

### Use UNet when
The dataset is small or GPU memory is limited. The standard UNet provides a strong baseline with minimal complexity. Default starting point for any new experiment.

### Use AttentionUNet when
There is spatial heterogeneity in the scene (e.g., mixed urban–vegetated regions) where different areas require different processing. The attention gates learn to suppress irrelevant skip features region-specifically.

### Use UNet++ when
Skip connection quality is suspected to be a bottleneck. UNet++ provides graduated feature fusion across multiple encoder levels, which can reduce the semantic gap between encoder and decoder.

### Use ResUNet when
Training is unstable or very deep (more than 5 levels). Residual connections in the encoder prevent gradient vanishing and allow deeper architectures.

### Use UNet Skip when
A residual encoder-decoder with MaxPool downsampling (rather than the stride-2 downsampling of [[ResUNet]]) is wanted. `UNetSkip` is a `ResUNetBackbone` instantiated with `downsample="maxpool"` (`resunet.py`): every encoder stage is a `ResidualConvBlock` at stride 1, followed by a separate `MaxPool2d(2)`, so the residual blocks never carry the downsampling stride themselves.

### Use LinkNet when
Computational efficiency is a priority. LinkNet's additive (rather than concatenative) skip connections reduce the decoder channel count and parameter budget substantially.

### Use SwinUNet / TransUNet / UNETR when
Long-range spatial dependencies are important (e.g., large homogeneous structures spanning many pixels). Transformer-based architectures attend globally, unlike CNN-only designs that are limited by receptive field.

### Use UNet Multihead / UNet Per-Gaussian when
The standard single-head UNet shows systematic bias across parameter types (multihead) or across Gaussian slots (per-Gaussian). These variants impose stronger inductive biases about parameter independence.

### Use ResUNet Multihead / ResUNet Per-Gaussian when
Combining the stride-2 residual [[ResUNet]] backbone (`ResUNetBackbone` with the default `downsample="stride"`) with the specialised head designs. These isolate the head-structure question from the backbone question while holding the residual encode-decode path fixed.

### Use DeepLabV3+ when
Multi-scale context at fixed dilation rates is suspected to matter (e.g., characteristic facade/layover scales). ASPP aggregates several receptive fields in parallel without extra downsampling.

### Use SegFormer when
Testing whether a strong hierarchical attention encoder with a near-trivial MLP decoder suffices — the complementary hypothesis to the heavy-decoder transformer designs.

### Use ConvNeXt UNet when
Evaluating modern convolution design (large depthwise kernels, inverted bottlenecks, LayerNorm, GELU) under the unchanged U-Net topology.

### Use DenseUNet when
Parameter efficiency through feature reuse is the question; at matched capacity it is the deepest, thinnest model in the zoo.

### Use HRNet when
Per-pixel position accuracy (peak-location error) is the priority: the full-resolution stream is never downsampled, avoiding the encode-decode round trip entirely.

### Use MultiResUNet when
Structures span several spatial scales; every block sees effective 3/5/7-pixel receptive fields in parallel, and ResPaths reduce the encoder-decoder semantic gap.

### Use FPN when
Probing how much decoder capacity the task actually needs: the FPN decoder is deliberately minimal, concentrating parameters in the encoder.

### Use U2-Net when
Intra-stage multi-scale mixing is desired: every stage is itself a small U-Net, so even full-resolution layers integrate wide context.

---

## Common Configuration

All architectures share:
- Input: $(C_{\text{in}}, P_H, P_W)$ patch tensor
- Output: $(3K, P_H, P_W)$ parameter tensor
- Optimised with AdamW (see [[Training Pipeline]])
- Per-layer learning rate groups defined in each `*Config.get_param_groups()`

Architectures requiring image size (`swin_unet`, `transunet`, `unetr`) receive `image_size = patch_height` from the dataset configuration.

---

## Shared Building Blocks

The recurring sub-modules live in a single `models/blocks.py` and are imported by every architecture. The canonical shared blocks are:

| Block | Role | Consumers |
|---|---|---|
| `ConvBlock` | Double $3\times3$ conv $\to$ Norm $\to$ Act (optional `Dropout2d`) | `unet`, `attention_unet`, `unetplusplus`, `transunet`, `unetr`, `convnext_unet` (decoder), `multires_unet`, `u2net` |
| `ResidualConvBlock` | Pre-activation residual unit with stride and `first_unit` controls | `resunet`, `unet_skip` and their head variants |
| `Encoder` / `Decoder` | Symmetric `ConvBlock` stacks with `MaxPool2d` / `build_upsample` and concatenative skips | `unet` family (`UNetBackbone`) |
| `PixelMLP` | $1\times1$ conv $\to$ Act $\to$ $1\times1$ conv pointwise head | multihead / per-Gaussian output heads |
| `GaussianHeadsMixin` | Resolves the $K = C_{\text{out}}/\texttt{params\_per\_gaussian}$ layout and builds the triple-head or per-Gaussian heads | `unet_multihead`, `unet_pergaussian`, `resunet_multihead`, `resunet_pergaussian` |
| `MultiHeadSelfAttention` / `TransformerBlock` | Pre-norm full self-attention and the transformer layer (LN $\to$ MSA $\to$ residual $\to$ LN $\to$ FFN $\to$ residual, with `DropPath`) | `transunet`, `unetr` |
| `PatchEmbedding` / `tokens_to_feature_map` | Strided-conv patch tokeniser and the inverse token $\to$ feature-map reshape | `transunet`, `unetr`; `tokens_to_feature_map` also used by `swin_unet` |
| `build_activation` / `build_norm2d` / `build_upsample` / `initialize_weights` / `match_spatial_size` / `DropPath` | Factory and utility helpers | all architectures |

The `unet` and `resunet` families share a thin backbone (`UNetBackbone`, `ResUNetBackbone`) that holds the encode-decode path; the single-head, triple-head (`*MultiHead`), and per-Gaussian (`*PerGaussian`) classes differ only in the output head they attach via `GaussianHeadsMixin`. The plain single-head models map the backbone embedding straight to `out_channels` with one `nn.Conv2d(embedding_channels, out_channels, kernel_size=1)` and emit those channels directly, with no Gaussian-layout resolution. Only the head variants call `GaussianHeadsMixin._resolve_gaussian_layout`, which sets $K = \texttt{out\_channels} / \texttt{params\_per\_gaussian}$ and raises `ValueError` when `out_channels` is not divisible by `params_per_gaussian`; `_build_triple_heads` then builds three `PixelMLP` heads each emitting $K$ channels (interleaved to $3K$ via `torch.stack(..., dim=2)`), while `_build_per_gaussian_heads` builds $K$ `PixelMLP` heads each emitting `params_per_gaussian` channels. Swin-specific window attention (`WindowAttention`, `SwinTransformerBlock`, `PatchMerging`, `PatchExpanding`, `SwinStage`) remains local to `models/backbone/SwinUNet.py` because it is not reused elsewhere.

---

## Configuration Persistence (ModelConfigIO)

Each training run persists the full architecture configuration so a checkpoint can be reloaded into a structurally identical model at inference time. `ModelConfigIO` (`tools/data/io.py`) serialises every dataclass field of the model config except `shape_logger_types` (the `EXCLUDED` set) to `model_config.json` in the run's metadata directory, recording `model_name`, `config_type` (the config class name), and the field/value map under `config`.

`ModelConfigIO.load` reconstructs the config strictly: it normalises `model_name` (lowercase, `-`/space to `_`), instantiates the dataclass from `CONFIG_REGISTRY[model_name]`, then overwrites each persisted field, raising loudly if the file is missing, the `model_name` is unknown, or a persisted field is not an attribute of the dataclass. Tuple-typed fields (e.g. `atrous_rates`, `rsu_heights`, `num_heads`, `sr_ratios`) are coerced back from their JSON list form when the live default is a tuple. There is no tolerant fallback: an absent `model_config.json` raises `FileNotFoundError` and forces a retrain rather than a best-effort reconstruction. This makes the hyperparameters in each model note (channel widths, depths, head counts, dilation rates) the exact quantities captured per run.

---

## Related Notes

- [[DLR-TomoSAR Index]] — Master index
- [[Training Pipeline]] — Model instantiation and training loop
- [[UNet]] — Baseline architecture
- [[Attention UNet]] — Attention-gated variant
- [[UNet Multihead]] — Parameter-type-specific heads
- [[UNet Per-Gaussian]] — Per-slot heads
- [[Configuration Layer]] — UNetConfig and variants
