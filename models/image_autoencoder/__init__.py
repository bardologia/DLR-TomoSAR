from configuration.model.image_autoencoder_models_config import (
    Conv2dImageAutoencoderConfig,
    ImageAutoencoderBaseConfig,
)
from .base   import ImageAutoencoderBase
from .conv2d import Conv2dImageAutoencoder


IMAGE_AE_MODEL_REGISTRY: dict[str, type] = {
    "conv2d_ae" : Conv2dImageAutoencoder,
}

IMAGE_AE_CONFIG_REGISTRY: dict[str, type] = {
    "conv2d_ae" : Conv2dImageAutoencoderConfig,
}


def get_image_autoencoder(name: str, config=None, **overrides):
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in IMAGE_AE_MODEL_REGISTRY:
        raise ValueError(f"Unknown image autoencoder '{name}'. Available: {list(IMAGE_AE_MODEL_REGISTRY.keys())}")
    if config is None:
        config = IMAGE_AE_CONFIG_REGISTRY[key](**overrides)
    elif overrides:
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)
    return IMAGE_AE_MODEL_REGISTRY[key](config), config


__all__ = [
    "ImageAutoencoderBase",
    "Conv2dImageAutoencoder",
    "ImageAutoencoderBaseConfig",
    "Conv2dImageAutoencoderConfig",
    "IMAGE_AE_MODEL_REGISTRY",
    "IMAGE_AE_CONFIG_REGISTRY",
    "get_image_autoencoder",
]
