from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import SwinUNetConfig, build_activation, DropPath, initialize_weights


class WindowAttention(nn.Module):
    def __init__(self, embedding_dim: int, window_size: int, num_heads: int,
                 dropout: float = 0.0, attention_dropout: float = 0.0):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_key_value = nn.Linear(embedding_dim, embedding_dim * 3)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coordinates = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(coordinates, coordinates, indexing="ij"))
        grid_flat = grid.reshape(2, -1)
        relative_coords = grid_flat[:, :, None] - grid_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, attention_mask=None):
        batch_windows, sequence_length, embedding_dim = x.shape
        qkv = self.query_key_value(x).reshape(
            batch_windows, sequence_length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attention_weights = (query @ key.transpose(-2, -1)) * self.scale

        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        attention_weights = attention_weights + relative_bias.unsqueeze(0)

        if attention_mask is not None:
            num_windows = attention_mask.shape[0]
            attention_weights = attention_weights.view(
                batch_windows // num_windows, num_windows, self.num_heads, sequence_length, sequence_length
            )
            attention_weights = attention_weights + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_weights = attention_weights.view(-1, self.num_heads, sequence_length, sequence_length)

        attention_weights = attention_weights.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        attended = (attention_weights @ value).transpose(1, 2).reshape(
            batch_windows, sequence_length, embedding_dim
        )
        return self.output_projection(attended)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        ffn_activation: str = "gelu",
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.embedding_dim = embedding_dim

        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.attention = WindowAttention(embedding_dim, window_size, num_heads, dropout, attention_dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm_2 = nn.LayerNorm(embedding_dim)
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            build_activation(ffn_activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def _create_attention_mask(self, height, width, device):
        if self.shift_size == 0:
            return None

        mask = torch.zeros(1, height, width, 1, device=device)
        height_slices = [
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        ]
        width_slices = [
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        ]
        region_index = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                mask[:, height_slice, width_slice, :] = region_index
                region_index += 1

        window_count = (height // self.window_size) * (width // self.window_size)
        mask_windows = mask.view(
            1,
            height // self.window_size, self.window_size,
            width // self.window_size, self.window_size,
            1,
        )
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        mask_windows = mask_windows.view(window_count, self.window_size * self.window_size)

        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, -100.0)
        attention_mask = attention_mask.masked_fill(attention_mask == 0, 0.0)
        return attention_mask

    def forward(self, x, height, width):
        batch_size, sequence_length, channels = x.shape
        x_2d = x.view(batch_size, height, width, channels)

        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        if pad_bottom > 0 or pad_right > 0:
            x_2d = functional.pad(x_2d, (0, 0, 0, pad_right, 0, pad_bottom))
        padded_height, padded_width = x_2d.shape[1], x_2d.shape[2]

        if self.shift_size > 0:
            shifted = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x_2d

        windows = shifted.view(
            batch_size,
            padded_height // self.window_size, self.window_size,
            padded_width // self.window_size, self.window_size,
            channels,
        )
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, self.window_size * self.window_size, channels)

        attention_mask = self._create_attention_mask(padded_height, padded_width, x.device)
        shortcut = x
        attended_windows = self.attention(self.norm_1(windows), attention_mask)

        num_height_windows = padded_height // self.window_size
        num_width_windows = padded_width // self.window_size
        attended_windows = attended_windows.view(
            batch_size, num_height_windows, num_width_windows,
            self.window_size, self.window_size, channels,
        )
        attended_windows = attended_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        shifted_back = attended_windows.view(batch_size, padded_height, padded_width, channels)

        if self.shift_size > 0:
            x_2d = torch.roll(shifted_back, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_2d = shifted_back

        if pad_bottom > 0 or pad_right > 0:
            x_2d = x_2d[:, :height, :width, :]

        x = shortcut + self.drop_path(x_2d.view(batch_size, height * width, channels))
        x = x + self.drop_path(self.feed_forward(self.norm_2(x)))
        return x


def match_spatial_size(source, reference):
    if source.shape[2:] != reference.shape[2:]:
        return functional.interpolate(
            source,
            size=reference.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
    return source


def tokens_to_feature_map(tokens, height, width):
    batch_size, _, channels = tokens.shape
    return tokens.transpose(1, 2).view(batch_size, channels, height, width)


class PatchMerging(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * input_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(4 * input_dim)

    def forward(self, x, height, width):
        batch_size, _, channels = x.shape
        x = x.view(batch_size, height, width, channels)
        top_left = x[:, 0::2, 0::2, :]
        top_right = x[:, 0::2, 1::2, :]
        bottom_left = x[:, 1::2, 0::2, :]
        bottom_right = x[:, 1::2, 1::2, :]

        x = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=-1)
        x = x.view(batch_size, -1, 4 * channels)
        x = self.norm(x)
        x = self.reduction(x)
        return x, height // 2, width // 2


class PatchExpanding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.expand = nn.Linear(input_dim, 4 * output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, x, height, width):
        batch_size, _, _ = x.shape
        x = self.expand(x)
        x = x.view(batch_size, height, width, 4, self.output_dim)
        x = x.view(batch_size, height, width, 2, 2, self.output_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch_size, height * 2, width * 2, self.output_dim)
        x = self.norm(x)
        x = x.view(batch_size, -1, self.output_dim)
        return x, height * 2, width * 2


class SwinEncoderStage(nn.Module):
    def __init__(self, embedding_dim: int, depth: int, num_heads: int, window_size: int, mlp_ratio: float,
                 dropout: float, attention_dropout: float = 0.0, ffn_activation: str = "gelu",
                 drop_path_rates: list[float] | None = None):
        super().__init__()
        self.blocks = nn.ModuleList()
        for block_index in range(depth):
            shift = 0 if block_index % 2 == 0 else window_size // 2
            dpr = drop_path_rates[block_index] if drop_path_rates else 0.0
            self.blocks.append(
                SwinTransformerBlock(embedding_dim, num_heads, window_size, shift, mlp_ratio, dropout,
                                     attention_dropout, ffn_activation, dpr)
            )

    def forward(self, x, height, width):
        for block in self.blocks:
            x = block(x, height, width)
        return x


class SwinDecoderStage(nn.Module):
    def __init__(self, embedding_dim: int, depth: int, num_heads: int, window_size: int, mlp_ratio: float,
                 dropout: float, attention_dropout: float = 0.0, ffn_activation: str = "gelu",
                 drop_path_rates: list[float] | None = None):
        super().__init__()
        self.blocks = nn.ModuleList()
        for block_index in range(depth):
            shift = 0 if block_index % 2 == 0 else window_size // 2
            dpr = drop_path_rates[block_index] if drop_path_rates else 0.0
            self.blocks.append(
                SwinTransformerBlock(embedding_dim, num_heads, window_size, shift, mlp_ratio, dropout,
                                     attention_dropout, ffn_activation, dpr)
            )

    def forward(self, x, height, width):
        for block in self.blocks:
            x = block(x, height, width)
        return x


class SwinUNet(nn.Module):
    def __init__(self, config: SwinUNetConfig | None = None):
        super().__init__()
        if config is None:
            config = SwinUNetConfig()
        self.config = config

        self.patch_embed = nn.Sequential(
            nn.Conv2d(config.in_channels, config.embedding_dim, kernel_size=config.patch_size, stride=config.patch_size),
        )
        self.patch_norm = nn.LayerNorm(config.embedding_dim)

        num_stages = len(config.depths)
        dims = [config.embedding_dim * (2 ** i) for i in range(num_stages)]

        # Compute linearly increasing drop path rates per block
        total_depth = sum(config.depths)
        dpr = [x.item() for x in torch.linspace(0, config.stochastic_depth_rate, total_depth)]
        dpr_splits = []
        cursor = 0
        for d in config.depths:
            dpr_splits.append(dpr[cursor:cursor + d])
            cursor += d

        self.encoder_stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for stage_index in range(num_stages):
            self.encoder_stages.append(
                SwinEncoderStage(
                    dims[stage_index], config.depths[stage_index], config.num_heads[stage_index],
                    config.window_size, config.mlp_ratio, config.dropout,
                    config.attention_dropout, config.ffn_activation, dpr_splits[stage_index],
                )
            )
            if stage_index < num_stages - 1:
                self.downsample_layers.append(PatchMerging(dims[stage_index], dims[stage_index + 1]))
            else:
                self.downsample_layers.append(nn.Identity())

        self.bottleneck_norm = nn.LayerNorm(dims[-1])

        self.upsample_layers = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()

        for stage_index in range(num_stages - 1):
            decoder_index = num_stages - 2 - stage_index
            self.upsample_layers.append(PatchExpanding(dims[decoder_index + 1], dims[decoder_index]))
            self.skip_projections.append(nn.Linear(dims[decoder_index] * 2, dims[decoder_index]))
            self.decoder_stages.append(
                SwinDecoderStage(
                    dims[decoder_index], config.depths[decoder_index], config.num_heads[decoder_index],
                    config.window_size, config.mlp_ratio, config.dropout,
                    config.attention_dropout, config.ffn_activation, dpr_splits[decoder_index],
                )
            )

        self.final_upsample = nn.ConvTranspose2d(
            dims[0], dims[0], kernel_size=config.patch_size, stride=config.patch_size,
        )
        self.output_head = nn.Conv2d(dims[0], config.out_channels, kernel_size=1)

        initialize_weights(self, config.init_mode)

    def forward(self, x):
        x = self.patch_embed(x)
        batch_size, channels, grid_height, grid_width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_norm(x)

        encoder_outputs = []
        height, width = grid_height, grid_width
        for stage_index, (encoder_stage, downsample) in enumerate(
            zip(self.encoder_stages, self.downsample_layers)
        ):
            x = encoder_stage(x, height, width)
            encoder_outputs.append((x, height, width))
            if stage_index < len(self.encoder_stages) - 1:
                x, height, width = downsample(x, height, width)

        x = self.bottleneck_norm(x)

        for stage_index, (upsample, skip_projection, decoder_stage) in enumerate(
            zip(self.upsample_layers, self.skip_projections, self.decoder_stages)
        ):
            skip_tokens, skip_height, skip_width = encoder_outputs[-(stage_index + 2)]
            x, height, width = upsample(x, height, width)

            if x.shape[1] != skip_tokens.shape[1]:
                feature_map = tokens_to_feature_map(x, height, width)
                skip_map = tokens_to_feature_map(skip_tokens, skip_height, skip_width)
                feature_map = match_spatial_size(feature_map, skip_map)
                height, width = skip_height, skip_width
                x = feature_map.flatten(2).transpose(1, 2)

            x = torch.cat([x, skip_tokens], dim=-1)
            x = skip_projection(x)
            x = decoder_stage(x, height, width)

        feature_map = tokens_to_feature_map(x, height, width)
        feature_map = self.final_upsample(feature_map)
        return self.output_head(feature_map)
