import importlib

_EXPORTS = {
    "InputConfig": "dataset_config",
    "OutputConfig": "dataset_config",
    "PatchConfiguration": "dataset_config",
    "AugmentationConfig": "dataset_config",
    "DatasetConfiguration": "dataset_config",
    "NormMethod": "norm_config",
    "ChannelStrategy": "norm_config",
    "Presets": "norm_config",
    "ChannelStats": "norm_config",
}

__all__ = [
    "InputConfig",
    "OutputConfig",
    "PatchConfiguration",
    "AugmentationConfig",
    "DatasetConfiguration",
    "NormMethod",
    "ChannelStrategy",
    "Presets",
    "ChannelStats",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
