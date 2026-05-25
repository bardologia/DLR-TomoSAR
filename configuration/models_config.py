from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Type

import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


def build_activation(name: str) -> nn.Module:
    factories = {
        "relu"        : lambda: nn.ReLU(inplace=True),
        "leaky_relu"  : lambda: nn.LeakyReLU(inplace=True),
        "gelu"        : lambda: nn.GELU(),
        "elu"         : lambda: nn.ELU(inplace=True),
        "silu"        : lambda: nn.SiLU(inplace=True),
    }
    if name not in factories:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(factories.keys())}")
    return factories[name]()


def build_norm2d(name: str, num_features: int) -> nn.Module:
    if name == "batch":
        return nn.BatchNorm2d(num_features)
    if name == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)
    if name == "group":
        num_groups = min(32, num_features)
        while num_features % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_features)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unknown normalization '{name}'. Available: batch, instance, group, none")


def build_upsample(mode: str, in_channels: int, out_channels: int, scale_factor: int = 2) -> nn.Module:
    if mode == "convtranspose":
        return nn.ConvTranspose2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = scale_factor,
            stride       = scale_factor,
        )
    if mode == "bilinear":
        return nn.Sequential(
            nn.Upsample(
                scale_factor  = scale_factor,
                mode          = "bilinear",
                align_corners = False,
            ),
            nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = 1,
                bias         = False,
            ),
        )
    raise ValueError(f"Unknown upsample mode '{mode}'. Available: convtranspose, bilinear")


def initialize_weights(module: nn.Module, mode: str) -> None:
    if mode == "default":
        return
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if mode == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif mode == "xavier":
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            if mode == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif mode == "xavier":
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)


@dataclass
class UNetConfig:
    in_channels         : int             = 1
    out_channels        : int             = 6
    params_per_gaussian : int             = 3
    features            : list[int]       = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int             = 2
    dropout             : float           = 0.15
    activation          : str             = "relu"
    normalization       : str             = "batch"
    upsample_mode       : str             = "convtranspose"
    conv_bias           : bool            = False
    init_mode           : str             = "default"

    encoder_lr          : float           = 3e-4
    bottleneck_lr       : float           = 3e-4
    decoder_lr          : float           = 3e-4
    output_head_lr      : float           = 1e-3

    encoder_wd          : float           = 5e-3
    bottleneck_wd       : float           = 5e-3
    decoder_wd          : float           = 5e-3
    output_head_wd      : float           = 5e-3
    
    shape_logger_types  : tuple           = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        return [g for g in [
            {'params': list(model.encoder.parameters()),     'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.bottleneck.parameters()),  'lr': self.bottleneck_lr,  'weight_decay': self.bottleneck_wd,  'name': 'bottleneck'},
            {'params': list(model.decoder.parameters()),     'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': list(model.output_head.parameters()), 'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class UNetMultiHeadConfig:
    in_channels         : int             = 1
    out_channels        : int             = 6
    params_per_gaussian : int             = 3
    features            : list[int]       = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int             = 2
    dropout             : float           = 0.15
    activation          : str             = "relu"
    normalization       : str             = "batch"
    upsample_mode       : str             = "convtranspose"
    conv_bias           : bool            = False
    init_mode           : str             = "default"

    encoder_lr          : float           = 3e-4
    bottleneck_lr       : float           = 3e-4
    decoder_lr          : float           = 3e-4
    heads_lr            : float           = 1e-3

    encoder_wd          : float           = 5e-3
    bottleneck_wd       : float           = 5e-3
    decoder_wd          : float           = 5e-3
    heads_wd            : float           = 5e-3

    shape_logger_types  : tuple           = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        return [g for g in [
            {'params': list(model.encoder.parameters()),                                                                      'lr': self.encoder_lr,    'weight_decay': self.encoder_wd,    'name': 'encoder'},
            {'params': list(model.bottleneck.parameters()),                                                                   'lr': self.bottleneck_lr, 'weight_decay': self.bottleneck_wd, 'name': 'bottleneck'},
            {'params': list(model.decoder.parameters()),                                                                      'lr': self.decoder_lr,    'weight_decay': self.decoder_wd,    'name': 'decoder'},
            {'params': list(model.head_amp.parameters()) + list(model.head_mu.parameters()) + list(model.head_sigma.parameters()), 'lr': self.heads_lr,      'weight_decay': self.heads_wd,      'name': 'heads'},
        ] if len(g['params']) > 0]


@dataclass
class ResUNetConfig:
    in_channels         : int             = 1
    out_channels        : int             = 6
    params_per_gaussian : int             = 3
    features            : list[int]       = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int             = 2
    dropout             : float           = 0.15
    activation          : str             = "relu"
    normalization       : str             = "batch"
    upsample_mode       : str             = "convtranspose"
    conv_bias           : bool            = False
    init_mode           : str             = "default"

    encoder_lr          : float           = 3e-4
    bottleneck_lr       : float           = 3e-4
    decoder_lr          : float           = 3e-4
    output_head_lr      : float           = 1e-3

    encoder_wd          : float           = 5e-3
    bottleneck_wd       : float           = 5e-3
    decoder_wd          : float           = 5e-3
    output_head_wd      : float           = 5e-3
    
    shape_logger_types  : tuple           = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        encoder_params = list(model.encoder_blocks.parameters()) + list(model.downsample_layers.parameters())
        decoder_params = list(model.upsample_layers.parameters()) + list(model.decoder_blocks.parameters())
        return [g for g in [
            {'params': encoder_params,                       'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.bottleneck.parameters()),  'lr': self.bottleneck_lr,  'weight_decay': self.bottleneck_wd,  'name': 'bottleneck'},
            {'params': decoder_params,                       'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': list(model.output_head.parameters()), 'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class AttentionUNetConfig:
    in_channels                  : int       = 1
    out_channels                 : int       = 6
    params_per_gaussian          : int       = 3
    features                     : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor            : int       = 2
    dropout                      : float     = 0.15
    attention_intermediate_ratio : float     = 0.5
    activation                   : str       = "relu"
    normalization                : str       = "batch"
    upsample_mode                : str       = "convtranspose"
    conv_bias                    : bool      = False
    init_mode                    : str       = "default"

    encoder_lr                   : float     = 3e-4
    bottleneck_lr                : float     = 3e-4
    decoder_lr                   : float     = 3e-4
    output_head_lr               : float     = 1e-3

    encoder_wd                   : float     = 5e-3
    bottleneck_wd                : float     = 5e-3
    decoder_wd                   : float     = 5e-3
    output_head_wd               : float     = 5e-3
    
    shape_logger_types           : tuple     = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU, nn.Sigmoid,
    ))

    def get_param_groups(self, model: nn.Module) -> list[dict]:
     
        encoder_params = (
            list(model.encoder_blocks.parameters()) +
            list(model.downsample_layers.parameters())
        )
     
        decoder_params = (
            list(model.upsample_layers.parameters()) +
            list(model.attention_gates.parameters()) +
            list(model.decoder_blocks.parameters())
        )
     
        return [g for g in [
            {'params': encoder_params,                        'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.bottleneck.parameters()),   'lr': self.bottleneck_lr,  'weight_decay': self.bottleneck_wd,  'name': 'bottleneck'},
            {'params': decoder_params,                        'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': list(model.output_head.parameters()),  'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class UNetPlusPlusConfig:
    in_channels           : int       = 1
    out_channels          : int       = 6
    params_per_gaussian   : int       = 3
    features              : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor     : int       = 2
    dropout               : float     = 0.15
    deep_supervision      : bool      = False
    activation            : str       = "relu"
    normalization         : str       = "batch"
    upsample_mode         : str       = "convtranspose"
    conv_bias             : bool      = False
    init_mode             : str       = "default"

    encoder_lr            : float     = 3e-4
    decoder_lr            : float     = 3e-4
    output_head_lr        : float     = 1e-3

    encoder_wd            : float     = 5e-3
    decoder_wd            : float     = 5e-3
    output_head_wd        : float     = 5e-3
    
    shape_logger_types    : tuple     = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Upsample, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        encoder_params, dense_params, upsample_params, head_params = [], [], [], []
        for name, param in model.named_parameters():
            if name.startswith("encoder_") or name.startswith("pool."):
                encoder_params.append(param)
            elif name.startswith("dense_"):
                dense_params.append(param)
            elif name.startswith("upsample.") or name.startswith("upsample_modules."):
                upsample_params.append(param)
            elif name.startswith("output_head"):
                head_params.append(param)
        
        return [g for g in [
            {'params': encoder_params,    'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': dense_params,      'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'dense_blocks'},
            {'params': upsample_params,   'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'upsample'},
            {'params': head_params,       'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'}
        ] if len(g['params']) > 0]


@dataclass
class FCNConfig:
    in_channels         : int       = 1
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int       = 2
    variant             : str       = "8s"
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr          : float     = 3e-4
    classifier_lr       : float     = 3e-4
    output_head_lr      : float     = 1e-3

    encoder_wd          : float     = 5e-3
    classifier_wd       : float     = 5e-3
    output_head_wd      : float     = 5e-3
    
    shape_logger_types  : tuple     = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        encoder_params = list(model.encoder_blocks.parameters()) + list(model.downsample_layers.parameters())
        head_params    = list(model.score_final.parameters())
       
        if hasattr(model, 'score_pool4'): head_params += list(model.score_pool4.parameters())
       
        if hasattr(model, 'score_pool3'): head_params += list(model.score_pool3.parameters())
       
        return [g for g in [
            {'params': encoder_params,                      'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.classifier.parameters()), 'lr': self.classifier_lr,  'weight_decay': self.classifier_wd,  'name': 'classifier'},
            {'params': head_params,                         'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class LinkNetConfig:
    in_channels              : int       = 1
    out_channels             : int       = 6
    params_per_gaussian      : int       = 3
    features                 : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    dropout                  : float     = 0.15
    initial_kernel_size      : int       = 7
    decoder_bottleneck_ratio : int       = 4
    activation               : str       = "relu"
    normalization            : str       = "batch"
    conv_bias                : bool      = False
    init_mode                : str       = "default"

    encoder_lr               : float     = 3e-4
    decoder_lr               : float     = 3e-4
    output_head_lr           : float     = 1e-3

    encoder_wd               : float     = 5e-3
    decoder_wd               : float     = 5e-3
    output_head_wd           : float     = 5e-3
    
    shape_logger_types       : tuple     = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        return [g for g in [
            {'params': list(model.initial_conv.parameters()),   'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'initial_conv'},
            {'params': list(model.encoder_stages.parameters()), 'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.decoder_stages.parameters()), 'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': list(model.output_head.parameters()),    'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class SwinUNetConfig:
    in_channels           : int       = 1
    out_channels          : int       = 6
    params_per_gaussian   : int       = 3
    image_size            : int       = 256
    patch_size            : int       = 4
    embedding_dim         : int       = 96
    depths                : list[int] = field(default_factory=lambda: [2, 2, 6, 2])
    num_heads             : list[int] = field(default_factory=lambda: [3, 6, 12, 24])
    window_size           : int       = 7
    mlp_ratio             : float     = 4.0
    dropout               : float     = 0.15
    attention_dropout     : float     = 0.0
    ffn_activation        : str       = "gelu"
    stochastic_depth_rate : float     = 0.0
    init_mode             : str       = "default"

    encoder_lr            : float     = 3e-4
    decoder_lr            : float     = 3e-4
    output_head_lr        : float     = 1e-3

    encoder_wd            : float     = 5e-3
    decoder_wd            : float     = 5e-3
    output_head_wd        : float     = 5e-3
    
    shape_logger_types    : tuple     = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.LayerNorm, nn.GELU, nn.Dropout,
    ))

    def get_param_groups(self, model: nn.Module) -> list[dict]:
     
        patch_embed_params = (
            list(model.patch_embed.parameters()) +
            list(model.patch_norm.parameters())
        )
     
        encoder_params = (
            list(model.encoder_stages.parameters()) +
            list(model.downsample_layers.parameters())
        )
     
        decoder_params = (
            list(model.upsample_layers.parameters()) +
            list(model.skip_projections.parameters()) +
            list(model.decoder_stages.parameters())
        )
     
        return [g for g in [
            {'params': patch_embed_params,                        'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'patch_embed'},
            {'params': encoder_params,                            'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.bottleneck_norm.parameters()),  'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'bottleneck'},
            {'params': decoder_params,                            'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': list(model.final_upsample.parameters()),   'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'final_upsample'},
            {'params': list(model.output_head.parameters()),      'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class TransUNetConfig:
    in_channels           : int       = 1
    out_channels          : int       = 6
    params_per_gaussian   : int       = 3
    image_size            : int       = 256
    cnn_features          : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor     : int       = 2
    transformer_layers    : int       = 12
    transformer_heads     : int       = 8
    transformer_mlp_ratio : float     = 4.0
    patch_size            : int       = 1
    dropout               : float     = 0.15
    activation            : str       = "relu"
    normalization         : str       = "batch"
    upsample_mode         : str       = "convtranspose"
    conv_bias             : bool      = False
    attention_dropout     : float     = 0.0
    ffn_activation        : str       = "gelu"
    stochastic_depth_rate : float     = 0.0
    init_mode             : str       = "default"

    encoder_lr            : float     = 3e-4
    decoder_lr            : float     = 3e-4
    output_head_lr        : float     = 1e-3

    encoder_wd            : float     = 5e-3
    decoder_wd            : float     = 5e-3
    output_head_wd        : float     = 5e-3
    
    shape_logger_types    : tuple     = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Linear, nn.LayerNorm,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
        nn.Dropout, nn.Dropout2d,
    ))

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        
        cnn_encoder_params = (
            list(model.encoder_blocks.parameters()) +
            list(model.downsample_layers.parameters()) +
            list(model.pre_transformer_conv.parameters())
        )
        
        patch_embed_params = (
            list(model.patch_embedding.parameters()) +
            [model.positional_embedding]
        )
        
        transformer_params = (
            list(model.transformer_blocks.parameters()) +
            list(model.transformer_norm.parameters())
        )
        
        decoder_params = (
            list(model.upsample_layers.parameters()) +
            list(model.decoder_blocks.parameters())
        )
        
        return [g for g in [
            {'params': cnn_encoder_params,                    'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'cnn_encoder'},
            {'params': patch_embed_params,                    'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'patch_embed'},
            {'params': transformer_params,                    'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'transformer'},
            {'params': decoder_params,                        'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': list(model.output_head.parameters()),  'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class UNETRConfig:
    in_channels           : int       = 1
    out_channels          : int       = 6
    params_per_gaussian   : int       = 3
    image_size            : int       = 256
    patch_size            : int       = 16
    embedding_dim         : int       = 768
    transformer_layers    : int       = 12
    transformer_heads     : int       = 12
    transformer_mlp_ratio : float     = 4.0
    decoder_features      : list[int] = field(default_factory=lambda: [512, 256, 128, 64])
    dropout               : float     = 0.15
    activation            : str       = "relu"
    normalization         : str       = "batch"
    conv_bias             : bool      = False
    attention_dropout     : float     = 0.0
    ffn_activation        : str       = "gelu"
    stochastic_depth_rate : float     = 0.0
    init_mode             : str       = "default"

    encoder_lr            : float     = 3e-4
    decoder_lr            : float     = 3e-4
    output_head_lr        : float     = 1e-3

    encoder_wd            : float     = 5e-3
    decoder_wd            : float     = 5e-3
    output_head_wd        : float     = 5e-3
    
    shape_logger_types    : tuple     = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.LayerNorm,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.GELU, nn.SiLU,
        nn.Dropout, nn.Dropout2d,
    ))

    def get_param_groups(self, model: nn.Module) -> list[dict]:
       
        patch_embed_params = (
            list(model.patch_embedding.parameters()) +
            [model.positional_embedding]
        )
       
        transformer_params = (
            list(model.transformer_blocks.parameters()) +
            list(model.transformer_norm.parameters())
        )
       
        decoder_params = (
            list(model.transformer_skip_heads.parameters()) +
            list(model.bottleneck_projection.parameters()) +
            list(model.input_skip_conv.parameters()) +
            list(model.upsample_layers.parameters()) +
            list(model.decoder_blocks.parameters()) +
            list(model.final_upsample.parameters())
        )
       
        return [g for g in [
            {'params': patch_embed_params,                    'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'patch_embed'},
            {'params': transformer_params,                    'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'transformer'},
            {'params': decoder_params,                        'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': list(model.output_head.parameters()),  'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


