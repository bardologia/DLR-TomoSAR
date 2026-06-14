import importlib

_EXPORTS = {
    "GaussianConfig": "gaussian_config",
    "GeometryConfig": "geometry_config",
    "TomogramConfiguration": "processing_config",
    "ParallelConfiguration": "processing_config",
    "PathConfiguration": "processing_config",
    "ProcessingConfiguration": "processing_config",
    "PreProcessEntryConfig": "processing_config",
}

__all__ = [
    "GaussianConfig",
    "GeometryConfig",
    "TomogramConfiguration",
    "ParallelConfiguration",
    "PathConfiguration",
    "ProcessingConfiguration",
    "PreProcessEntryConfig",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
