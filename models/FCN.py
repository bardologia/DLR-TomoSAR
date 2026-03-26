from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import FCNConfig, build_activation, build_norm2d, initialize_weights


# Double 3x3 convolution block: Conv -> Norm -> Act -> Conv -> Norm -> Act (+ optional dropout)
class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channels:  int,
        output_channels: int,
        dropout:         float = 0.0,
        activation:      str   = "relu",
        normalization:   str   = "batch",
        bias:            bool  = False,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = 3,
                padding      = 1,
                bias         = bias,
            ),
            build_norm2d(normalization, output_channels),
            build_activation(activation),
            nn.Conv2d(
                in_channels  = output_channels,
                out_channels = output_channels,
                kernel_size  = 3,
                padding      = 1,
                bias         = bias,
            ),
            build_norm2d(normalization, output_channels),
            build_activation(activation),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# Resizes source tensor to match the spatial dimensions of reference
def match_spatial_size(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if source.shape[2:] != reference.shape[2:]:
        return functional.interpolate(
            input         = source,
            size          = reference.shape[2:],
            mode          = "bilinear",
            align_corners = False,
        )
    return source


class FCN(nn.Module):
    """Fully Convolutional Network (Long, Shelhamer & Darrell, CVPR 2015).

    Supports three coarse-to-fine variants:

    * **FCN-32s** – upsample the deepest score map directly to input
      resolution.
    * **FCN-16s** – fuse the deepest score map with the second-deepest
      encoder features (*pool-4* equivalent) via a learned 1×1 score
      projection and element-wise addition before upsampling.
    * **FCN-8s**  – further fuse the third-deepest encoder features
      (*pool-3* equivalent) for the finest predictions.

    Score maps at each fused level are produced by 1×1 convolutions that
    project encoder features directly to ``out_channels`` dimensions,
    matching the paper's design.  The skip-connection score layers
    (``score_pool4`` / ``score_pool3``) are zero-initialised so that
    training begins as FCN-32s and gradually learns to incorporate finer
    features.
    """

    def __init__(self, config: FCNConfig | None = None):
        super().__init__()
        if config is None:
            config = FCNConfig()
        self.config = config

        feature_sizes = config.features
        variant = config.variant

        if len(feature_sizes) == 0:
            raise ValueError("features must contain at least one channel size")
        if variant not in ("32s", "16s", "8s"):
            raise ValueError(f"variant must be '32s', '16s', or '8s', got '{variant}'")
        if variant == "16s" and len(feature_sizes) < 2:
            raise ValueError("FCN-16s requires at least 2 encoder stages")
        if variant == "8s" and len(feature_sizes) < 3:
            raise ValueError("FCN-8s requires at least 3 encoder stages")

        # ---- encoder --------------------------------------------------------
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = config.in_channels
        for feature_size in feature_sizes:
            self.encoder_blocks.append(
                ConvBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                )
            )
            self.downsample_layers.append(nn.MaxPool2d(kernel_size=2))
            channels = feature_size

        # ---- classifier (replaces VGG fc6 + fc7 with convolutions) ----------
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor
        classifier_layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels  = feature_sizes[-1],
                out_channels = bottleneck_channels,
                kernel_size  = 3,
                padding      = 1,
                bias         = config.conv_bias,
            ),
            build_norm2d(config.normalization, bottleneck_channels),
            build_activation(config.activation),
        ]
        if config.dropout > 0:
            classifier_layers.append(nn.Dropout2d(config.dropout))
        classifier_layers += [
            nn.Conv2d(
                in_channels  = bottleneck_channels,
                out_channels = bottleneck_channels,
                kernel_size  = 1,
                bias         = config.conv_bias,
            ),
            build_norm2d(config.normalization, bottleneck_channels),
            build_activation(config.activation),
        ]
        if config.dropout > 0:
            classifier_layers.append(nn.Dropout2d(config.dropout))
        self.classifier = nn.Sequential(*classifier_layers)

        # ---- score projection layers ----------------------------------------
        self.score_final = nn.Conv2d(
            in_channels  = bottleneck_channels,
            out_channels = config.out_channels,
            kernel_size  = 1,
        )

        if variant in ("16s", "8s"):
            self.score_pool4 = nn.Conv2d(
                in_channels  = feature_sizes[-1],
                out_channels = config.out_channels,
                kernel_size  = 1,
            )

        if variant == "8s":
            self.score_pool3 = nn.Conv2d(
                in_channels  = feature_sizes[-2],
                out_channels = config.out_channels,
                kernel_size  = 1,
            )

        initialize_weights(module=self, mode=config.init_mode)

        # Zero-initialise skip score layers (Long et al., 2015) so the model
        # begins as FCN-32s and gradually learns to incorporate finer features.
        if variant in ("16s", "8s"):
            nn.init.zeros_(self.score_pool4.weight)
            if self.score_pool4.bias is not None:
                nn.init.zeros_(self.score_pool4.bias)
        if variant == "8s":
            nn.init.zeros_(self.score_pool3.weight)
            if self.score_pool3.bias is not None:
                nn.init.zeros_(self.score_pool3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_input = x

        # Encoder: extract multi-scale features (like VGG backbone)
        encoder_features = []
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = encoder_block(x)
            encoder_features.append(x)
            x = downsample(x)

        # Classifier: replaces fully connected layers with convolutions
        x = self.classifier(x)

        # Score from deepest features (FCN-32s baseline)
        score = self.score_final(x)

        # Fuse pool-4 score (adds finer detail for FCN-16s / FCN-8s)
        if self.config.variant in ("16s", "8s"):
            score = functional.interpolate(
                input         = score,
                scale_factor  = 2,
                mode          = "bilinear",
                align_corners = False,
            )
            score_pool4 = self.score_pool4(encoder_features[-1])
            score = match_spatial_size(source=score, reference=score_pool4)
            score = score + score_pool4

        # Fuse pool-3 score (finest detail, FCN-8s only)
        if self.config.variant == "8s":
            score = functional.interpolate(
                input         = score,
                scale_factor  = 2,
                mode          = "bilinear",
                align_corners = False,
            )
            score_pool3 = self.score_pool3(encoder_features[-2])
            score = match_spatial_size(source=score, reference=score_pool3)
            score = score + score_pool3

        # Upsample final score map to original input resolution
        if score.shape[2:] != original_input.shape[2:]:
            score = functional.interpolate(
                input         = score,
                size          = original_input.shape[2:],
                mode          = "bilinear",
                align_corners = False,
            )

        return score
