from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class ImageAutoencoderBaseConfig:
    in_channels : int = 1

    embedding_dim  : int = 24
    embedding_norm : str = "none"

    downsample_factor : int = 1
    base_channels     : int = 32
    depth             : int = 2

    activation    : str   = "gelu"
    normalization : str   = "batch"
    dropout       : float = 0.0
    init_mode     : str   = "default"

    encoder_lr : float = 3e-4
    decoder_lr : float = 3e-4

    encoder_wd : float = 1e-4
    decoder_wd : float = 1e-4

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"    : {"type": "float",       "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"    : {"type": "float",       "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"    : {"type": "float",       "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"    : {"type": "float",       "low": 1e-6, "high": 1e-1, "log": True},
            "embedding_dim" : {"type": "categorical", "choices": [16, 24, 32, 48]},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {}

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        groups = [
            {"params": list(model.encoder.parameters()), "lr": self.encoder_lr, "weight_decay": self.encoder_wd, "name": "image_ae_encoder"},
            {"params": list(model.decoder.parameters()), "lr": self.decoder_lr, "weight_decay": self.decoder_wd, "name": "image_ae_decoder"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


@dataclass
class Conv2dImageAutoencoderConfig(ImageAutoencoderBaseConfig):
    upsample_mode : str = "convtranspose"

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "base_channels"     : {"type": "categorical", "choices": [16, 32, 64]},
            "depth"             : {"type": "categorical", "choices": [1, 2, 3]},
            "downsample_factor" : {"type": "categorical", "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical", "choices": ["relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical", "choices": ["batch", "instance", "group"]},
            "dropout"           : {"type": "float",       "low": 0.0, "high": 0.3},
        }
