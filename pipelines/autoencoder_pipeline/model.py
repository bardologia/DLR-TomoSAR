from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from .config import AutoencoderConfig, BackboneType, DecoderConfig, EncoderConfig


class AutoencoderOutput(NamedTuple):
    reconstruction : torch.Tensor
    latent         : torch.Tensor
    projection     : torch.Tensor | None


class Layers:
    @staticmethod
    def activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu" : return nn.ReLU(inplace=True)
        if name == "gelu" : return nn.GELU()
        if name == "silu" : return nn.SiLU(inplace=True)
        if name == "tanh" : return nn.Tanh()
        raise ValueError(f"Unknown activation '{name}'.")

    @staticmethod
    def norm_1d(name: str, num_features: int) -> nn.Module:
        name = name.lower()
        if name == "batch" : return nn.BatchNorm1d(num_features)
        if name == "layer" : return nn.GroupNorm(1, num_features)
        if name == "group" : return nn.GroupNorm(min(8, num_features), num_features)
        if name == "none"  : return nn.Identity()
        raise ValueError(f"Unknown normalization '{name}'.")

    @staticmethod
    def output_activation(name: str | None) -> nn.Module:
        if name is None or name == "none": return nn.Identity()
        if name == "sigmoid"             : return nn.Sigmoid()
        if name == "tanh"                : return nn.Tanh()
        if name == "softplus"            : return nn.Softplus()
        raise ValueError(f"Unknown output activation '{name}'.")


class ProjectionHead(nn.Module):

    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, activation: str = "gelu") -> None:
        super().__init__()
        layers : list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(Layers.activation(activation))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Encoder(nn.Module):

    def __init__(self, profile_length: int, latent_dim: int, config: EncoderConfig) -> None:
        super().__init__()
        self.profile_length = profile_length
        self.latent_dim     = latent_dim
        self.config         = config

        if config.backbone == BackboneType.conv1d:
            self._build_conv()
        elif config.backbone == BackboneType.mlp:
            self._build_mlp()
        else:
            raise ValueError(f"Unknown encoder backbone: {config.backbone}")

    def _build_conv(self) -> None:
        cfg     = self.config
        blocks  : list[nn.Module] = []
        in_ch   = 1
        length  = self.profile_length
        for out_ch in cfg.channels:
            blocks.append(nn.Conv1d(in_ch, out_ch, kernel_size = cfg.kernel_size, stride = cfg.stride, padding = cfg.kernel_size // 2))
            blocks.append(Layers.norm_1d(cfg.normalization, out_ch))
            blocks.append(Layers.activation(cfg.activation))
            if cfg.dropout > 0:
                blocks.append(nn.Dropout(cfg.dropout))
            in_ch  = out_ch
            length = (length + 2 * (cfg.kernel_size // 2) - cfg.kernel_size) // cfg.stride + 1

        self.feature_extractor = nn.Sequential(*blocks)
        self.feature_length    = max(1, length)
        self.feature_channels  = in_ch
        self.bottleneck        = nn.Linear(self.feature_channels * self.feature_length, self.latent_dim)
        self.head              = None

    def _build_mlp(self) -> None:
        cfg     = self.config
        layers  : list[nn.Module] = []
        prev    = self.profile_length
        for h in cfg.mlp_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(Layers.norm_1d(cfg.normalization, h))
            layers.append(Layers.activation(cfg.activation))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev = h
        layers.append(nn.Linear(prev, self.latent_dim))
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_length    = 1
        self.feature_channels  = self.latent_dim
        self.bottleneck        = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.backbone == BackboneType.conv1d:
            if x.ndim == 2:
                x = x.unsqueeze(1)
            feats = self.feature_extractor(x)
            return self.bottleneck(feats.flatten(1))
        if x.ndim == 3:
            x = x.squeeze(1)
        return self.feature_extractor(x)


class Decoder(nn.Module):

    def __init__(self,
                 profile_length : int,
                 latent_dim     : int,
                 config         : DecoderConfig,
                 start_length   : int = 1,
                 start_channels : int = 1) -> None:
        
        super().__init__()
        self.profile_length = profile_length
        self.latent_dim     = latent_dim
        self.config         = config
        self.start_length   = start_length
        self.start_channels = start_channels

        if config.backbone == BackboneType.conv1d:
            self._build_conv()
        elif config.backbone == BackboneType.mlp:
            self._build_mlp()
        else:
            raise ValueError(f"Unknown decoder backbone: {config.backbone}")

    def _build_conv(self) -> None:
        cfg = self.config
        self.unbottleneck = nn.Linear(self.latent_dim, self.start_channels * self.start_length)

        blocks : list[nn.Module] = []
        in_ch  = self.start_channels
        for out_ch in cfg.channels:
            blocks.append(nn.ConvTranspose1d(
                in_ch, out_ch,
                kernel_size    = cfg.kernel_size,
                stride         = cfg.stride,
                padding        = cfg.kernel_size // 2,
                output_padding = cfg.stride - 1,
            ))
            blocks.append(Layers.norm_1d(cfg.normalization, out_ch))
            blocks.append(Layers.activation(cfg.activation))
            if cfg.dropout > 0:
                blocks.append(nn.Dropout(cfg.dropout))
            in_ch = out_ch

        self.upsampler  = nn.Sequential(*blocks)
        self.head_conv  = nn.Conv1d(in_ch, 1, kernel_size=1)
        self.out_act    = Layers.output_activation(cfg.output_activation)
        self.mlp_net    = None

    def _build_mlp(self) -> None:
        cfg = self.config
        layers : list[nn.Module] = []
        prev = self.latent_dim
        for h in cfg.mlp_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(Layers.norm_1d(cfg.normalization, h))
            layers.append(Layers.activation(cfg.activation))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev = h
        layers.append(nn.Linear(prev, self.profile_length))
        self.mlp_net      = nn.Sequential(*layers)
        self.out_act      = Layers.output_activation(cfg.output_activation)
        self.unbottleneck = None
        self.upsampler    = None
        self.head_conv    = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.config.backbone == BackboneType.conv1d:
            x = self.unbottleneck(z).view(-1, self.start_channels, self.start_length)
            x = self.upsampler(x)
            x = self.head_conv(x)
            if x.shape[-1] != self.profile_length:
                x = nn.functional.interpolate(x, size=self.profile_length, mode="linear", align_corners=False)
            return self.out_act(x.squeeze(1))
        return self.out_act(self.mlp_net(z))


class Autoencoder(nn.Module):

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = Encoder(
            profile_length = config.profile_length,
            latent_dim     = config.latent_dim,
            config         = config.encoder,
        )

        if config.encoder.backbone == BackboneType.conv1d and config.decoder.backbone == BackboneType.conv1d:
            start_length   = self.encoder.feature_length
            start_channels = self.encoder.feature_channels
        else:
            start_length   = 1
            start_channels = config.decoder.channels[0] if config.decoder.channels else 1

        self.decoder = Decoder(
            profile_length = config.profile_length,
            latent_dim     = config.latent_dim,
            config         = config.decoder,
            start_length   = start_length,
            start_channels = start_channels,
        )

        if config.encoder.use_projection_head:
            self.projection_head : nn.Module | None = ProjectionHead(
                in_dim     = config.latent_dim,
                hidden     = config.encoder.proj_hidden,
                out_dim    = config.encoder.proj_dim,
                activation = config.encoder.activation,
            )
        else:
            self.projection_head = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def project(self, z: torch.Tensor) -> torch.Tensor | None:
        if self.projection_head is None:
            return None
        return self.projection_head(z)

    def forward(self, x: torch.Tensor) -> AutoencoderOutput:
        z     = self.encode(x)
        recon = self.decode(z)
        proj  = self.project(z)
        return AutoencoderOutput(reconstruction=recon, latent=z, projection=proj)
