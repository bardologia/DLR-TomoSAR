from configuration.models_config import (
    AttentionUNetConfig,
    ConvNeXtUNetConfig,
    DeepLabV3PlusConfig,
    DenseUNetConfig,
    DropPath,
    FPNNetConfig,
    HRNetLiteConfig,
    LinkNetConfig,
    MultiResUNetConfig,
    ResUNetConfig,
    ResUNetMultiHeadConfig,
    ResUNetPerGaussianConfig,
    SegFormerLiteConfig,
    SwinUNetConfig,
    TransUNetConfig,
    U2NetLiteConfig,
    UNETRConfig,
    UNetConfig,
    UNetMultiHeadConfig,
    UNetPerGaussianConfig,
    UNetPlusPlusConfig,
    UNetSkipConfig,
    build_activation,
    build_norm2d,
    build_upsample,
    initialize_weights,
)
from .unet import UNet, UNetMultiHead, UNetPerGaussian
from .resunet import ResUNet, ResUNetMultiHead, ResUNetPerGaussian, UNetSkip
from .AttentionUNet import AttentionUNet
from .UNetPlusPlus import UNetPlusPlus
from .LinkNet import LinkNet
from .SwinUNet import SwinUNet
from .TransUNet import TransUNet
from .UNETR import UNETR
from .DeepLabV3Plus import DeepLabV3Plus
from .SegFormerLite import SegFormerLite
from .ConvNeXtUNet import ConvNeXtUNet
from .DenseUNet import DenseUNet
from .HRNetLite import HRNetLite
from .MultiResUNet import MultiResUNet
from .FPNNet import FPNNet
from .U2NetLite import U2NetLite

MODEL_REGISTRY: dict[str, type] = {
    "unet"                : UNet,
    "unet_multihead"      : UNetMultiHead,
    "unet_pergaussian"    : UNetPerGaussian,
    "unet_skip"           : UNetSkip,
    "resunet"             : ResUNet,
    "resunet_multihead"   : ResUNetMultiHead,
    "resunet_pergaussian" : ResUNetPerGaussian,
    "attention_unet"      : AttentionUNet,
    "unetplusplus"        : UNetPlusPlus,
    "linknet"             : LinkNet,
    "swin_unet"           : SwinUNet,
    "transunet"           : TransUNet,
    "unetr"               : UNETR,
    "deeplabv3plus"       : DeepLabV3Plus,
    "segformer"           : SegFormerLite,
    "convnext_unet"       : ConvNeXtUNet,
    "dense_unet"          : DenseUNet,
    "hrnet"               : HRNetLite,
    "multires_unet"       : MultiResUNet,
    "fpn"                 : FPNNet,
    "u2net"               : U2NetLite,
}

CONFIG_REGISTRY: dict[str, type] = {
    "unet"                : UNetConfig,
    "unet_multihead"      : UNetMultiHeadConfig,
    "unet_pergaussian"    : UNetPerGaussianConfig,
    "unet_skip"           : UNetSkipConfig,
    "resunet"             : ResUNetConfig,
    "resunet_multihead"   : ResUNetMultiHeadConfig,
    "resunet_pergaussian" : ResUNetPerGaussianConfig,
    "attention_unet"      : AttentionUNetConfig,
    "unetplusplus"        : UNetPlusPlusConfig,
    "linknet"             : LinkNetConfig,
    "swin_unet"           : SwinUNetConfig,
    "transunet"           : TransUNetConfig,
    "unetr"               : UNETRConfig,
    "deeplabv3plus"       : DeepLabV3PlusConfig,
    "segformer"           : SegFormerLiteConfig,
    "convnext_unet"       : ConvNeXtUNetConfig,
    "dense_unet"          : DenseUNetConfig,
    "hrnet"               : HRNetLiteConfig,
    "multires_unet"       : MultiResUNetConfig,
    "fpn"                 : FPNNetConfig,
    "u2net"               : U2NetLiteConfig,
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
    "UNetSkip",
    "ResUNet",
    "ResUNetMultiHead",
    "ResUNetPerGaussian",
    "AttentionUNet",
    "UNetPlusPlus",
    "LinkNet",
    "SwinUNet",
    "TransUNet",
    "UNETR",
    "DeepLabV3Plus",
    "SegFormerLite",
    "ConvNeXtUNet",
    "DenseUNet",
    "HRNetLite",
    "MultiResUNet",
    "FPNNet",
    "U2NetLite",
    "UNetConfig",
    "UNetMultiHeadConfig",
    "UNetPerGaussianConfig",
    "UNetSkipConfig",
    "ResUNetConfig",
    "ResUNetMultiHeadConfig",
    "ResUNetPerGaussianConfig",
    "AttentionUNetConfig",
    "UNetPlusPlusConfig",
    "LinkNetConfig",
    "SwinUNetConfig",
    "TransUNetConfig",
    "UNETRConfig",
    "DeepLabV3PlusConfig",
    "SegFormerLiteConfig",
    "ConvNeXtUNetConfig",
    "DenseUNetConfig",
    "HRNetLiteConfig",
    "MultiResUNetConfig",
    "FPNNetConfig",
    "U2NetLiteConfig",
    "get_model",
    "MODEL_REGISTRY",
    "CONFIG_REGISTRY",
    "build_activation",
    "build_norm2d",
    "build_upsample",
    "initialize_weights",
    "DropPath",
]
