from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import torch.nn as nn



@dataclass
class BackboneConfigBase:
    in_channels         : int = 1
    out_channels        : int = 6
    params_per_gaussian : int = 3

    PARAM_GROUP_SPEC : ClassVar[tuple] = ()

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {}

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        groups = []
        for name, attrs, lr_field, wd_field in self.PARAM_GROUP_SPEC:
            params = []
            for attr in attrs:
                target = getattr(model, attr)
                if isinstance(target, nn.Module):
                    params += list(target.parameters())
                else:
                    params.append(target)
            groups.append({'params': params, 'lr': getattr(self, lr_field), 'weight_decay': getattr(self, wd_field), 'name': name})
        return [g for g in groups if len(g['params']) > 0]


@dataclass
class UNetConfig(BackboneConfigBase):
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types  : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',     ('encoder',),     'encoder_lr',     'encoder_wd'),
        ('bottleneck',  ('bottleneck',),  'bottleneck_lr',  'bottleneck_wd'),
        ('decoder',     ('decoder',),     'decoder_lr',     'decoder_wd'),
        ('output_head', ('output_head',), 'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"          : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384], [64, 128, 256, 512, 1024]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }


@dataclass
class UNetMultiHeadConfig(BackboneConfigBase):
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr    : float = 3e-4
    bottleneck_lr : float = 3e-4
    decoder_lr    : float = 3e-4
    heads_lr      : float = 1e-3

    encoder_wd    : float = 1e-4
    bottleneck_wd : float = 1e-4
    decoder_wd    : float = 1e-4
    heads_wd      : float = 1e-4

    shape_logger_types  : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',    ('encoder',),                            'encoder_lr',    'encoder_wd'),
        ('bottleneck', ('bottleneck',),                         'bottleneck_lr', 'bottleneck_wd'),
        ('decoder',    ('decoder',),                            'decoder_lr',    'decoder_wd'),
        ('heads',      ('head_amp', 'head_mu', 'head_sigma'),   'heads_lr',      'heads_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"    : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"    : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "heads_lr"      : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"    : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"    : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "heads_wd"      : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"       : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"          : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }


@dataclass
class UNetPerGaussianConfig(BackboneConfigBase):
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr    : float = 3e-4
    bottleneck_lr : float = 3e-4
    decoder_lr    : float = 3e-4
    heads_lr      : float = 1e-3

    encoder_wd    : float = 1e-4
    bottleneck_wd : float = 1e-4
    decoder_wd    : float = 1e-4
    heads_wd      : float = 1e-4

    shape_logger_types  : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',    ('encoder',),         'encoder_lr',    'encoder_wd'),
        ('bottleneck', ('bottleneck',),      'bottleneck_lr', 'bottleneck_wd'),
        ('decoder',    ('decoder',),         'decoder_lr',    'decoder_wd'),
        ('heads',      ('gaussian_heads',),  'heads_lr',      'heads_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"    : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"    : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "heads_lr"      : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"    : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"    : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "heads_wd"      : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"       : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"          : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }


@dataclass
class ResUNetConfig(BackboneConfigBase):
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types  : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',     ('encoder_blocks',),                    'encoder_lr',     'encoder_wd'),
        ('bottleneck',  ('bottleneck',),                        'bottleneck_lr',  'bottleneck_wd'),
        ('decoder',     ('upsample_layers', 'decoder_blocks'),  'decoder_lr',     'decoder_wd'),
        ('output_head', ('output_head',),                       'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"          : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }


@dataclass
class UNetSkipConfig(BackboneConfigBase):
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types  : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',     ('encoder_blocks', 'downsample_layers'), 'encoder_lr',     'encoder_wd'),
        ('bottleneck',  ('bottleneck',),                         'bottleneck_lr',  'bottleneck_wd'),
        ('decoder',     ('upsample_layers', 'decoder_blocks'),   'decoder_lr',     'decoder_wd'),
        ('output_head', ('output_head',),                        'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"          : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }


@dataclass
class AttentionUNetConfig(BackboneConfigBase):
    features                     : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor            : int       = 2
    dropout                      : float     = 0.15
    attention_intermediate_ratio : float     = 0.5
    activation                   : str       = "relu"
    normalization                : str       = "batch"
    upsample_mode                : str       = "convtranspose"
    conv_bias                    : bool      = False
    init_mode                    : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types           : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU, nn.Sigmoid,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',     ('encoder_blocks', 'downsample_layers'),                  'encoder_lr',     'encoder_wd'),
        ('bottleneck',  ('bottleneck',),                                          'bottleneck_lr',  'bottleneck_wd'),
        ('decoder',     ('upsample_layers', 'attention_gates', 'decoder_blocks'), 'decoder_lr',     'decoder_wd'),
        ('output_head', ('output_head',),                                         'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"                   : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "bottleneck_factor"          : {"type": "categorical",         "choices": [1, 2, 4]},
            "attention_intermediate_ratio": {"type": "float",              "low": 0.25, "high": 0.75},
            "activation"                 : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"              : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"              : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }


@dataclass
class UNetPlusPlusConfig(BackboneConfigBase):
    features            : list[int] = field(default_factory=lambda: [56, 112, 216, 440])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types    : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Upsample, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"          : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }

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
class LinkNetConfig(BackboneConfigBase):
    features                 : list[int] = field(default_factory=lambda: [152, 312, 624, 1248])
    dropout                  : float     = 0.15
    initial_kernel_size      : int       = 7
    decoder_bottleneck_ratio : int       = 4
    activation               : str       = "relu"
    normalization            : str       = "batch"
    conv_bias                : bool      = False
    init_mode                : str       = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types       : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('initial_conv', ('initial_conv',),   'encoder_lr',     'encoder_wd'),
        ('encoder',      ('encoder_stages',), 'encoder_lr',     'encoder_wd'),
        ('decoder',      ('decoder_stages',), 'decoder_lr',     'decoder_wd'),
        ('output_head',  ('output_head',),    'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"                : {"type": "indexed_categorical", "choices": [[64, 128, 256, 512], [152, 312, 624, 1248], [96, 192, 384, 768]]},
            "initial_kernel_size"     : {"type": "categorical",         "choices": [3, 5, 7]},
            "decoder_bottleneck_ratio": {"type": "categorical",         "choices": [2, 4, 8]},
            "activation"              : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"           : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
        }


@dataclass
class SwinUNetConfig(BackboneConfigBase):
    image_size            : int       = 256
    patch_size            : int       = 4
    embedding_dim         : int       = 80
    depths                : list[int] = field(default_factory=lambda: [2, 2, 6, 2])
    num_heads             : list[int] = field(default_factory=lambda: [2, 5, 10, 20])
    window_size           : int       = 7
    mlp_ratio             : float     = 4.0
    dropout               : float     = 0.30
    attention_dropout     : float     = 0.10
    ffn_activation        : str       = "gelu"
    stochastic_depth_rate : float     = 0.10
    init_mode             : str       = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-2
    decoder_wd     : float = 1e-2
    output_head_wd : float = 1e-2

    shape_logger_types    : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.LayerNorm, nn.GELU, nn.Dropout,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('patch_embed',    ('patch_embed', 'patch_norm'),                          'encoder_lr',     'encoder_wd'),
        ('encoder',        ('encoder_stages', 'downsample_layers'),                'encoder_lr',     'encoder_wd'),
        ('bottleneck',     ('bottleneck_norm',),                                   'encoder_lr',     'encoder_wd'),
        ('decoder',        ('upsample_layers', 'skip_projections', 'decoder_stages'), 'decoder_lr',  'decoder_wd'),
        ('final_upsample', ('final_upsample',),                                    'decoder_lr',     'decoder_wd'),
        ('output_head',    ('output_head',),                                       'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "embedding_dim"         : {"type": "categorical", "choices": [48, 80, 96, 128]},
            "attention_dropout"     : {"type": "float",       "low": 0.0,  "high": 0.3},
            "stochastic_depth_rate" : {"type": "float",       "low": 0.0,  "high": 0.3},
            "ffn_activation"        : {"type": "categorical", "choices": ["gelu", "relu", "silu"]},
        }


@dataclass
class TransUNetConfig(BackboneConfigBase):
    image_size            : int       = 256
    cnn_features          : list[int] = field(default_factory=lambda: [32, 72, 136, 272])
    bottleneck_factor     : int       = 2
    transformer_layers    : int       = 6
    transformer_heads     : int       = 4
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

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 5e-3
    decoder_wd     : float = 5e-3
    output_head_wd : float = 5e-3

    shape_logger_types    : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Linear, nn.LayerNorm,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
        nn.Dropout, nn.Dropout2d,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('cnn_encoder', ('encoder_blocks', 'downsample_layers', 'pre_transformer_conv'), 'encoder_lr',     'encoder_wd'),
        ('patch_embed', ('patch_embedding', 'positional_embedding'),                     'encoder_lr',     'encoder_wd'),
        ('transformer', ('transformer_blocks', 'transformer_norm'),                      'encoder_lr',     'encoder_wd'),
        ('decoder',     ('upsample_layers', 'decoder_blocks'),                           'decoder_lr',     'decoder_wd'),
        ('output_head', ('output_head',),                                                'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.4},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "cnn_features"          : {"type": "indexed_categorical", "choices": [[16, 32, 64, 128], [32, 72, 136, 272], [32, 64, 128, 256]]},
            "transformer_layers"    : {"type": "categorical",         "choices": [2, 4, 6, 8]},
            "transformer_heads"     : {"type": "categorical",         "choices": [2, 4, 8]},
            "attention_dropout"     : {"type": "float",               "low": 0.0, "high": 0.3},
            "stochastic_depth_rate" : {"type": "float",               "low": 0.0, "high": 0.2},
            "activation"            : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"         : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
        }


@dataclass
class UNETRConfig(BackboneConfigBase):
    image_size            : int       = 256
    patch_size            : int       = 16
    embedding_dim         : int       = 544
    transformer_layers    : int       = 8
    transformer_heads     : int       = 8
    transformer_mlp_ratio : float     = 4.0
    decoder_features      : list[int] = field(default_factory=lambda: [360, 184, 88, 48])
    dropout               : float     = 0.15
    activation            : str       = "relu"
    normalization         : str       = "batch"
    conv_bias             : bool      = False
    attention_dropout     : float     = 0.0
    ffn_activation        : str       = "gelu"
    stochastic_depth_rate : float     = 0.0
    init_mode             : str       = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 5e-3
    decoder_wd     : float = 5e-3
    output_head_wd : float = 5e-3

    shape_logger_types    : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.LayerNorm,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.GELU, nn.SiLU,
        nn.Dropout, nn.Dropout2d,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('patch_embed', ('patch_embedding', 'positional_embedding'),                                                                                             'encoder_lr',     'encoder_wd'),
        ('transformer', ('transformer_blocks', 'transformer_norm'),                                                                                              'encoder_lr',     'encoder_wd'),
        ('decoder',     ('transformer_skip_heads', 'bottleneck_projection', 'input_skip_conv', 'upsample_layers', 'decoder_blocks', 'final_upsample'),           'decoder_lr',     'decoder_wd'),
        ('output_head', ('output_head',),                                                                                                                        'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.4},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "embedding_dim"         : {"type": "categorical",         "choices": [256, 384, 544, 768]},
            "transformer_layers"    : {"type": "categorical",         "choices": [4, 6, 8, 12]},
            "transformer_heads"     : {"type": "categorical",         "choices": [4, 8, 16]},
            "decoder_features"      : {"type": "indexed_categorical", "choices": [[180, 96, 48, 24], [360, 184, 88, 48], [256, 128, 64, 32]]},
            "attention_dropout"     : {"type": "float",               "low": 0.0, "high": 0.3},
            "stochastic_depth_rate" : {"type": "float",               "low": 0.0, "high": 0.2},
        }


@dataclass
class ResUNetMultiHeadConfig(BackboneConfigBase):
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr    : float = 3e-4
    bottleneck_lr : float = 3e-4
    decoder_lr    : float = 3e-4
    heads_lr      : float = 1e-3

    encoder_wd    : float = 1e-4
    bottleneck_wd : float = 1e-4
    decoder_wd    : float = 1e-4
    heads_wd      : float = 1e-4

    shape_logger_types  : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',    ('encoder_blocks',),                     'encoder_lr',    'encoder_wd'),
        ('bottleneck', ('bottleneck',),                         'bottleneck_lr', 'bottleneck_wd'),
        ('decoder',    ('upsample_layers', 'decoder_blocks'),   'decoder_lr',    'decoder_wd'),
        ('heads',      ('head_amp', 'head_mu', 'head_sigma'),   'heads_lr',      'heads_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"    : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"    : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "heads_lr"      : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"    : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"    : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "heads_wd"      : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"       : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"          : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }


@dataclass
class ResUNetPerGaussianConfig(BackboneConfigBase):
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr    : float = 3e-4
    bottleneck_lr : float = 3e-4
    decoder_lr    : float = 3e-4
    heads_lr      : float = 1e-3

    encoder_wd    : float = 1e-4
    bottleneck_wd : float = 1e-4
    decoder_wd    : float = 1e-4
    heads_wd      : float = 1e-4

    shape_logger_types  : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',    ('encoder_blocks',),                    'encoder_lr',    'encoder_wd'),
        ('bottleneck', ('bottleneck',),                        'bottleneck_lr', 'bottleneck_wd'),
        ('decoder',    ('upsample_layers', 'decoder_blocks'),  'decoder_lr',    'decoder_wd'),
        ('heads',      ('gaussian_heads',),                    'heads_lr',      'heads_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"    : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"    : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "heads_lr"      : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"    : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"    : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "heads_wd"      : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"       : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"          : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }


@dataclass
class DeepLabV3PlusConfig(BackboneConfigBase):
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    atrous_rates        : tuple     = (1, 2, 4)
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    aspp_lr        : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    aspp_wd        : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types  : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',     ('stem', 'encoder_stages'),                       'encoder_lr',     'encoder_wd'),
        ('aspp',        ('aspp',),                                        'aspp_lr',        'aspp_wd'),
        ('decoder',     ('low_level_projection', 'decoder_blocks'),       'decoder_lr',     'decoder_wd'),
        ('output_head', ('output_head',),                                 'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "aspp_lr"        : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "aspp_wd"        : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"      : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "atrous_rates"  : {"type": "categorical",         "choices": [(1, 2, 4), (2, 4, 8), (1, 2, 4, 8)]},
            "activation"    : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization" : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
        }


@dataclass
class SegFormerLiteConfig(BackboneConfigBase):
    embedding_dims        : list[int] = field(default_factory=lambda: [40, 80, 192, 320])
    depths                : list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_heads             : tuple     = (1, 2, 4, 8)
    sr_ratios             : tuple     = (4, 2, 2, 1)
    mlp_ratio             : float     = 4.0
    decoder_channels      : int       = 256
    dropout               : float     = 0.10
    attention_dropout     : float     = 0.0
    ffn_activation        : str       = "gelu"
    stochastic_depth_rate : float     = 0.10
    init_mode             : str       = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-2
    decoder_wd     : float = 1e-2
    output_head_wd : float = 1e-2

    shape_logger_types    : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.Linear, nn.LayerNorm, nn.BatchNorm2d,
        nn.GELU, nn.ReLU, nn.SiLU, nn.Dropout, nn.Dropout2d,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',     ('patch_embeddings', 'encoder_stages'),    'encoder_lr',     'encoder_wd'),
        ('decoder',     ('decode_projections', 'fuse'),            'decoder_lr',     'decoder_wd'),
        ('output_head', ('output_head',),                          'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.4},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "embedding_dims"        : {"type": "indexed_categorical", "choices": [[32, 64, 160, 256], [40, 80, 192, 320], [64, 128, 320, 512]]},
            "depths"                : {"type": "indexed_categorical", "choices": [[2, 2, 2, 2], [2, 2, 4, 2], [3, 3, 6, 3]]},
            "decoder_channels"      : {"type": "categorical",         "choices": [128, 256, 384]},
            "attention_dropout"     : {"type": "float",               "low": 0.0, "high": 0.3},
            "stochastic_depth_rate" : {"type": "float",               "low": 0.0, "high": 0.2},
        }


@dataclass
class ConvNeXtUNetConfig(BackboneConfigBase):
    features              : list[int] = field(default_factory=lambda: [48, 96, 192, 384])
    bottleneck_factor     : int       = 2
    blocks_per_stage      : int       = 2
    ffn_ratio             : float     = 4.0
    ffn_activation        : str       = "gelu"
    stochastic_depth_rate : float     = 0.10
    layer_scale_init      : float     = 1e-6
    conv_bias             : bool      = False
    init_mode             : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 5e-3
    bottleneck_wd  : float = 5e-3
    decoder_wd     : float = 5e-3
    output_head_wd : float = 5e-3

    shape_logger_types    : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.LayerNorm,
        nn.GELU, nn.ReLU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',     ('stem', 'encoder_stages', 'downsample_layers'),  'encoder_lr',     'encoder_wd'),
        ('bottleneck',  ('bottleneck',),                                  'bottleneck_lr',  'bottleneck_wd'),
        ('decoder',     ('upsample_layers', 'decoder_stages'),            'decoder_lr',     'decoder_wd'),
        ('output_head', ('output_head',),                                 'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"              : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [48, 96, 192, 384], [64, 128, 256, 512]]},
            "blocks_per_stage"      : {"type": "categorical",         "choices": [1, 2, 3]},
            "bottleneck_factor"     : {"type": "categorical",         "choices": [1, 2]},
            "ffn_ratio"             : {"type": "categorical",         "choices": [2.0, 4.0]},
            "stochastic_depth_rate" : {"type": "float",               "low": 0.0, "high": 0.3},
        }


@dataclass
class DenseUNetConfig(BackboneConfigBase):
    growth_rate         : int       = 16
    block_layers        : list[int] = field(default_factory=lambda: [4, 4, 4])
    bottleneck_layers   : int       = 4
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types  : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',     ('stem', 'dense_down', 'trans_down'),  'encoder_lr',     'encoder_wd'),
        ('bottleneck',  ('bottleneck',),                       'bottleneck_lr',  'bottleneck_wd'),
        ('decoder',     ('trans_up', 'dense_up'),              'decoder_lr',     'decoder_wd'),
        ('output_head', ('output_head',),                      'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "growth_rate"       : {"type": "categorical",         "choices": [12, 16, 24, 32]},
            "block_layers"      : {"type": "indexed_categorical", "choices": [[4, 4, 4], [4, 5, 7], [5, 5, 5, 5]]},
            "bottleneck_layers" : {"type": "categorical",         "choices": [4, 6, 8]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
        }


@dataclass
class HRNetLiteConfig(BackboneConfigBase):
    base_channels       : int   = 48
    n_branches          : int   = 3
    blocks_per_stage    : int   = 2
    dropout             : float = 0.15
    activation          : str   = "relu"
    normalization       : str   = "batch"
    conv_bias           : bool  = False
    init_mode           : str   = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types  : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',     ('stem', 'transition_modules', 'stage_modules', 'fuse_modules'), 'encoder_lr',     'encoder_wd'),
        ('decoder',     ('final_fuse',),                                                 'decoder_lr',     'decoder_wd'),
        ('output_head', ('output_head',),                                                'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "base_channels"    : {"type": "categorical", "choices": [32, 48, 64]},
            "n_branches"       : {"type": "categorical", "choices": [2, 3, 4]},
            "blocks_per_stage" : {"type": "categorical", "choices": [1, 2, 3]},
            "activation"       : {"type": "categorical", "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"    : {"type": "categorical", "choices": ["batch", "instance", "group"]},
        }


@dataclass
class MultiResUNetConfig(BackboneConfigBase):
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types  : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',     ('encoder_blocks', 'downsample_layers', 'res_paths'), 'encoder_lr',     'encoder_wd'),
        ('bottleneck',  ('bottleneck',),                                      'bottleneck_lr',  'bottleneck_wd'),
        ('decoder',     ('upsample_layers', 'decoder_blocks'),               'decoder_lr',     'decoder_wd'),
        ('output_head', ('output_head',),                                    'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"          : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }


@dataclass
class FPNNetConfig(BackboneConfigBase):
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    pyramid_channels    : int       = 128
    segmentation_convs  : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types  : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',     ('encoder_blocks', 'downsample_layers'),                                'encoder_lr',     'encoder_wd'),
        ('decoder',     ('lateral_convs', 'smooth_convs', 'segmentation_blocks', 'fuse_block'), 'decoder_lr',     'decoder_wd'),
        ('output_head', ('output_head',),                                                       'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"           : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "pyramid_channels"   : {"type": "categorical",         "choices": [64, 128, 256]},
            "segmentation_convs" : {"type": "categorical",         "choices": [1, 2, 3]},
            "activation"         : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"      : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
        }


@dataclass
class U2NetLiteConfig(BackboneConfigBase):
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    rsu_heights         : tuple     = (5, 4, 3)
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types  : tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    PARAM_GROUP_SPEC : ClassVar[tuple] = (
        ('encoder',     ('encoder_stages',), 'encoder_lr',     'encoder_wd'),
        ('bridge',      ('bridge',),         'bottleneck_lr',  'bottleneck_wd'),
        ('decoder',     ('decoder_stages',), 'decoder_lr',     'decoder_wd'),
        ('output_head', ('output_head',),    'output_head_lr', 'output_head_wd'),
    )

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"      : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "rsu_heights"   : {"type": "categorical",         "choices": [(4, 3, 2), (5, 4, 3), (6, 5, 4)]},
            "activation"    : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization" : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
        }
