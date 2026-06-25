from configuration.architectures import AttentionUNetConfig, ConvNeXtUNetConfig, DeepLabV3PlusConfig, DenseUNetConfig, FPNNetConfig, HRNetLiteConfig, LinkNetConfig, MultiResUNetConfig, ResUNetConfig, ResUNetMultiHeadConfig, ResUNetPerGaussianConfig, SegFormerLiteConfig, SwinUNetConfig, TransUNetConfig, U2NetLiteConfig, UNETRConfig, UNetConfig, UNetMultiHeadConfig, UNetPerGaussianConfig, UNetPlusPlusConfig, UNetSkipConfig
from ..blocks       import DropPath, build_activation, build_norm2d, build_upsample, initialize_weights
from .unet          import UNet, UNetMultiHead, UNetPerGaussian
from .resunet       import ResUNet, ResUNetMultiHead, ResUNetPerGaussian, UNetSkip
from .attention_unet import AttentionUNet
from .unet_plus_plus  import UNetPlusPlus
from .link_net        import LinkNet
from .swin_unet       import SwinUNet
from .trans_unet      import TransUNet
from .unetr           import UNETR
from .deeplab_v3_plus import DeepLabV3Plus
from .segformer_lite  import SegFormerLite
from .convnext_unet   import ConvNeXtUNet
from .dense_unet      import DenseUNet
from .hrnet_lite      import HRNetLite
from .multires_unet   import MultiResUNet
from .fpn_net         import FPNNet
from .u2net_lite      import U2NetLite

BACKBONE_MODEL_REGISTRY: dict[str, type] = {
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

BACKBONE_CONFIG_REGISTRY: dict[str, type] = {
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


BACKBONE_IMAGE_SIZE_MODELS: frozenset[str] = frozenset({"swin_unet", "transunet", "unetr"})


def get_backbone(name: str, config=None, **overrides):
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in BACKBONE_MODEL_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(BACKBONE_MODEL_REGISTRY.keys())}")
    if config is None:
        config = BACKBONE_CONFIG_REGISTRY[key](**overrides)
    elif overrides:
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)
    return BACKBONE_MODEL_REGISTRY[key](config), config


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
    "get_backbone",
    "BACKBONE_MODEL_REGISTRY",
    "BACKBONE_CONFIG_REGISTRY",
    "BACKBONE_IMAGE_SIZE_MODELS",
    "build_activation",
    "build_norm2d",
    "build_upsample",
    "initialize_weights",
    "DropPath",
]
