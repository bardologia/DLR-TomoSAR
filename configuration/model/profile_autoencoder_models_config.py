from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class ProfileAutoencoderBaseConfig:
    profile_length : int   = 256

    embedding_dim  : int = 24
    embedding_norm : str = "l2"

    activation : str = "gelu"
    init_mode  : str = "default"

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
            {"params": list(model.encoder.parameters()), "lr": self.encoder_lr, "weight_decay": self.encoder_wd, "name": "ae_encoder"},
            {"params": list(model.decoder.parameters()), "lr": self.decoder_lr, "weight_decay": self.decoder_wd, "name": "ae_decoder"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


@dataclass
class MlpAutoencoderConfig(ProfileAutoencoderBaseConfig):
    hidden_dim : int   = 128
    depth      : int   = 2
    dropout    : float = 0.0

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "hidden_dim" : {"type": "categorical", "choices": [64, 128, 256]},
            "depth"      : {"type": "categorical", "choices": [1, 2, 3]},
            "activation" : {"type": "categorical", "choices": ["relu", "gelu", "silu"]},
            "dropout"    : {"type": "float",       "low": 0.0, "high": 0.3},
        }


@dataclass
class Conv1dAutoencoderConfig(ProfileAutoencoderBaseConfig):
    seq_channels    : int = 32
    seq_kernel_size : int = 5

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "seq_channels"    : {"type": "categorical", "choices": [16, 32, 64]},
            "seq_kernel_size" : {"type": "categorical", "choices": [3, 5, 7]},
            "activation"      : {"type": "categorical", "choices": ["relu", "gelu", "silu"]},
        }


@dataclass
class Transformer1dAutoencoderConfig(ProfileAutoencoderBaseConfig):
    hidden_dim : int   = 128
    depth      : int   = 2
    num_heads  : int   = 4
    dropout    : float = 0.0

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "hidden_dim" : {"type": "categorical", "choices": [64, 128, 256]},
            "depth"      : {"type": "categorical", "choices": [1, 2, 4]},
            "num_heads"  : {"type": "categorical", "choices": [2, 4, 8]},
            "dropout"    : {"type": "float",       "low": 0.0, "high": 0.3},
            "activation" : {"type": "categorical", "choices": ["relu", "gelu"]},
        }
