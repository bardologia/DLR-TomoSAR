from configuration.model.profile_autoencoder_models_config import (
    ProfileAutoencoderBaseConfig,
    Conv1dAutoencoderConfig,
    MlpAutoencoderConfig,
    Transformer1dAutoencoderConfig,
)
from .base          import ProfileAutoencoderBase, ProfileAutoencoderBlocks
from .mlp           import MlpAutoencoder
from .conv1d        import Conv1dAutoencoder
from .transformer1d import Transformer1dAutoencoder


PROFILE_AE_MODEL_REGISTRY: dict[str, type] = {
    "mlp_ae"           : MlpAutoencoder,
    "conv1d_ae"        : Conv1dAutoencoder,
    "transformer1d_ae" : Transformer1dAutoencoder,
}

PROFILE_AE_CONFIG_REGISTRY: dict[str, type] = {
    "mlp_ae"           : MlpAutoencoderConfig,
    "conv1d_ae"        : Conv1dAutoencoderConfig,
    "transformer1d_ae" : Transformer1dAutoencoderConfig,
}


def get_profile_autoencoder(name: str, config=None, **overrides):
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in PROFILE_AE_MODEL_REGISTRY:
        raise ValueError(f"Unknown autoencoder '{name}'. Available: {list(PROFILE_AE_MODEL_REGISTRY.keys())}")
    if config is None:
        config = PROFILE_AE_CONFIG_REGISTRY[key](**overrides)
    elif overrides:
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)
    return PROFILE_AE_MODEL_REGISTRY[key](config), config


__all__ = [
    "ProfileAutoencoderBase",
    "ProfileAutoencoderBlocks",
    "MlpAutoencoder",
    "Conv1dAutoencoder",
    "Transformer1dAutoencoder",
    "ProfileAutoencoderBaseConfig",
    "MlpAutoencoderConfig",
    "Conv1dAutoencoderConfig",
    "Transformer1dAutoencoderConfig",
    "PROFILE_AE_MODEL_REGISTRY",
    "PROFILE_AE_CONFIG_REGISTRY",
    "get_profile_autoencoder",
]
