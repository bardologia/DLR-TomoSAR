from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class DualResUNetConfig:
    in_channels         : int       = 9
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    head                : str       = "set_pred"
    ifg_channels        : tuple     = (5, 6, 7, 8)
    params_features     : list[int] = field(default_factory=lambda: [64, 128, 256])
    existence_features  : list[int] = field(default_factory=lambda: [64, 128])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    params_trunk_lr    : float = 3e-4
    existence_trunk_lr : float = 3e-4
    output_head_lr     : float = 1e-3

    params_trunk_wd    : float = 1e-4
    existence_trunk_wd : float = 1e-4
    output_head_wd     : float = 1e-4

    shape_logger_types  : tuple           = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "params_trunk_lr"    : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "existence_trunk_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "params_trunk_wd"    : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "existence_trunk_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"            : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "params_features"    : {"type": "indexed_categorical", "choices": [[32, 64, 128], [64, 128, 256], [48, 96, 192]]},
            "existence_features" : {"type": "indexed_categorical", "choices": [[32, 64], [64, 128], [48, 96]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        return [g for g in [
            {'params': list(model.trunk_params.parameters()),    'lr': self.params_trunk_lr,    'weight_decay': self.params_trunk_wd,    'name': 'params_trunk'},
            {'params': list(model.trunk_existence.parameters()), 'lr': self.existence_trunk_lr, 'weight_decay': self.existence_trunk_wd, 'name': 'existence_trunk'},
            {'params': model.head_parameters(),                  'lr': self.output_head_lr,     'weight_decay': self.output_head_wd,     'name': 'output_head'},
        ] if len(g['params']) > 0]
