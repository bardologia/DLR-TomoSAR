from .backbone import *  # noqa: F401,F403
from .backbone import CONFIG_REGISTRY, IMAGE_SIZE_MODELS, MODEL_REGISTRY, get_model
from .blocks   import DropPath, build_activation, build_norm2d, build_upsample, initialize_weights
from .autoencoder import (
    AE_CONFIG_REGISTRY,
    AE_MODEL_REGISTRY,
    AutoencoderBase,
    Conv1dAutoencoder,
    MlpAutoencoder,
    Transformer1dAutoencoder,
    get_autoencoder,
)

def config_registry(training_type: str) -> dict:
    if training_type == "autoencoder":
        return AE_CONFIG_REGISTRY
    return CONFIG_REGISTRY


__all__ = [
    "CONFIG_REGISTRY",
    "config_registry",
    "MODEL_REGISTRY",
    "IMAGE_SIZE_MODELS",
    "get_model",
    "DropPath",
    "build_activation",
    "build_norm2d",
    "build_upsample",
    "initialize_weights",
    "AE_CONFIG_REGISTRY",
    "AE_MODEL_REGISTRY",
    "AutoencoderBase",
    "MlpAutoencoder",
    "Conv1dAutoencoder",
    "Transformer1dAutoencoder",
    "get_autoencoder",
]
