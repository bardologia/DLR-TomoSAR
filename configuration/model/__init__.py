import importlib

_EXPORTS = {
    "UNetConfig": "models_config",
    "UNetMultiHeadConfig": "models_config",
    "UNetPerGaussianConfig": "models_config",
    "ResUNetConfig": "models_config",
    "UNetSkipConfig": "models_config",
    "AttentionUNetConfig": "models_config",
    "UNetPlusPlusConfig": "models_config",
    "LinkNetConfig": "models_config",
    "SwinUNetConfig": "models_config",
    "TransUNetConfig": "models_config",
    "UNETRConfig": "models_config",
    "ResUNetMultiHeadConfig": "models_config",
    "ResUNetPerGaussianConfig": "models_config",
    "DeepLabV3PlusConfig": "models_config",
    "SegFormerLiteConfig": "models_config",
    "ConvNeXtUNetConfig": "models_config",
    "DenseUNetConfig": "models_config",
    "HRNetLiteConfig": "models_config",
    "MultiResUNetConfig": "models_config",
    "FPNNetConfig": "models_config",
    "U2NetLiteConfig": "models_config",
    "AutoencoderConfig": "models_config",
}

__all__ = [
    "UNetConfig",
    "UNetMultiHeadConfig",
    "UNetPerGaussianConfig",
    "ResUNetConfig",
    "UNetSkipConfig",
    "AttentionUNetConfig",
    "UNetPlusPlusConfig",
    "LinkNetConfig",
    "SwinUNetConfig",
    "TransUNetConfig",
    "UNETRConfig",
    "ResUNetMultiHeadConfig",
    "ResUNetPerGaussianConfig",
    "DeepLabV3PlusConfig",
    "SegFormerLiteConfig",
    "ConvNeXtUNetConfig",
    "DenseUNetConfig",
    "HRNetLiteConfig",
    "MultiResUNetConfig",
    "FPNNetConfig",
    "U2NetLiteConfig",
    "AutoencoderConfig",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
