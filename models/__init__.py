from configuration.models_config import (
    AttentionUNetConfig,
    DropPath,
    FCNConfig,
    LinkNetConfig,
    ResUNetConfig,
    SwinUNetConfig,
    TransUNetConfig,
    UNETRConfig,
    UNetConfig,
    UNetMultiHeadConfig,
    UNetPlusPlusConfig,
    build_activation,
    build_norm2d,
    build_upsample,
    initialize_weights,
)
from .UNet import UNet
from .UNet_multihead import UNetMultiHead
from .ResUNet import ResUNet
from .AttentionUNet import AttentionUNet
from .UNetPlusPlus import UNetPlusPlus
from .FCN import FCN
from .LinkNet import LinkNet
from .SwinUNet import SwinUNet
from .TransUNet import TransUNet
from .UNETR import UNETR

MODEL_REGISTRY: dict[str, type] = {
    "unet"           : UNet,
    "unet_multihead" : UNetMultiHead,
    "resunet"        : ResUNet,
    "attention_unet" : AttentionUNet,
    "unetplusplus"   : UNetPlusPlus,
    "fcn"            : FCN,
    "linknet"        : LinkNet,
    "swin_unet"      : SwinUNet,
    "transunet"      : TransUNet,
    "unetr"          : UNETR,
}

CONFIG_REGISTRY: dict[str, type] = {
    "unet"           : UNetConfig,
    "unet_multihead" : UNetMultiHeadConfig,
    "resunet"        : ResUNetConfig,
    "attention_unet" : AttentionUNetConfig,
    "unetplusplus"   : UNetPlusPlusConfig,
    "fcn"            : FCNConfig,
    "linknet"        : LinkNetConfig,
    "swin_unet"      : SwinUNetConfig,
    "transunet"      : TransUNetConfig,
    "unetr"          : UNETRConfig,
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
    "ResUNet",
    "AttentionUNet",
    "UNetPlusPlus",
    "FCN",
    "LinkNet",
    "SwinUNet",
    "TransUNet",
    "UNETR",
    "UNetConfig",
    "UNetMultiHeadConfig",
    "ResUNetConfig",
    "AttentionUNetConfig",
    "UNetPlusPlusConfig",
    "FCNConfig",
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
