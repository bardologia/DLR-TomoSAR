import importlib

_EXPORTS = {
    "InferencePaths": "inference_config",
    "InferenceConfig": "inference_config",
    "InferenceEntryConfig": "inference_config",
}

__all__ = [
    "InferencePaths",
    "InferenceConfig",
    "InferenceEntryConfig",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
