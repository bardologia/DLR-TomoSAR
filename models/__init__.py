from configuration.models_config import (
    AttentionUNetConfig,
    DropPath,
    LinkNetConfig,
    ResUNetConfig,
    SwinUNetConfig,
    TransUNetConfig,
    UNETRConfig,
    UNetConfig,
    UNetMultiHeadConfig,
    UNetPerGaussianConfig,
    UNetPlusPlusConfig,
    build_activation,
    build_norm2d,
    build_upsample,
    initialize_weights,
)
from .UNet import UNet
from .UNet_multihead import UNetMultiHead
from .UNet_pergaussian import UNetPerGaussian
from .ResUNet import ResUNet
from .AttentionUNet import AttentionUNet
from .UNetPlusPlus import UNetPlusPlus
from .LinkNet import LinkNet
from .SwinUNet import SwinUNet
from .TransUNet import TransUNet
from .UNETR import UNETR

MODEL_REGISTRY: dict[str, type] = {
    "unet"           : UNet,
    "unet_multihead"   : UNetMultiHead,
    "unet_pergaussian" : UNetPerGaussian,
    "resunet"          : ResUNet,
    "attention_unet" : AttentionUNet,
    "unetplusplus"   : UNetPlusPlus,
    "linknet"        : LinkNet,
    "swin_unet"      : SwinUNet,
    "transunet"      : TransUNet,
    "unetr"          : UNETR,
}

CONFIG_REGISTRY: dict[str, type] = {
    "unet"             : UNetConfig,
    "unet_multihead"   : UNetMultiHeadConfig,
    "unet_pergaussian" : UNetPerGaussianConfig,
    "resunet"          : ResUNetConfig,
    "attention_unet"   : AttentionUNetConfig,
    "unetplusplus"     : UNetPlusPlusConfig,
    "linknet"          : LinkNetConfig,
    "swin_unet"        : SwinUNetConfig,
    "transunet"        : TransUNetConfig,
    "unetr"            : UNETRConfig,
}


def get_model(name: str, config=None, **overrides):
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    if config is None:
        config = CONFIG_REGISTRY[key](**overrides)
    elif overrides:
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)
    return MODEL_REGISTRY[key](config), config


__all__ = [
    "UNet",
    "UNetMultiHead",
    "UNetPerGaussian",
    "ResUNet",
    "AttentionUNet",
    "UNetPlusPlus",
    "LinkNet",
    "SwinUNet",
    "TransUNet",
    "UNETR",
    "UNetConfig",
    "UNetMultiHeadConfig",
    "UNetPerGaussianConfig",
    "ResUNetConfig",
    "AttentionUNetConfig",
    "UNetPlusPlusConfig",
    "LinkNetConfig",
    "SwinUNetConfig",
    "TransUNetConfig",
    "UNETRConfig",
    "get_model",
    "MODEL_REGISTRY",
    "CONFIG_REGISTRY",
    "build_activation",
    "build_norm2d",
    "build_upsample",
    "initialize_weights",
    "DropPath",
]
