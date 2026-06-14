from __future__ import annotations

import torch
import torch.nn as nn

from models                                              import IMAGE_SIZE_MODELS, get_model
from models.autoencoder                                  import get_autoencoder
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats
from pipelines.backbone.inference.loader                 import ModelWrapper, RunLoader
from pipelines.jepa.training.trainer                     import JepaModule
from tools.data.io                                       import AutoencoderConfigIO, ModelConfigIO



class JepaInferenceModel(nn.Module):
    def __init__(self, jepa_module: JepaModule, profile_normalizer: ProfileNormalizer) -> None:
        super().__init__()
        self.jepa               = jepa_module
        self.profile_normalizer = profile_normalizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_hat   = self.jepa.backbone(x)
        curve_n = self.jepa.autoencoder.decode(self.jepa.autoencoder.normalize_embedding(z_hat))
        return self.profile_normalizer.denormalize(curve_n)


class JepaRunLoader(RunLoader):
    def _build_model(self, model_name: str, in_channels: int, out_channels: int, image_size: int):
        ae_cfg, ae_name = AutoencoderConfigIO.load(self.meta_directory)
        model_config, _ = ModelConfigIO.load(self.meta_directory)

        overrides = {"in_channels": in_channels, "out_channels": ae_cfg.embedding_dim}
        if model_name in IMAGE_SIZE_MODELS:
            overrides["image_size"] = image_size
        backbone, _ = get_model(model_name, config=model_config, **overrides)

        autoencoder, _          = get_autoencoder(ae_name, ae_cfg)
        self.profile_normalizer = ProfileNormalizer(ProfileStats.load(self.meta_directory))

        return JepaModule(backbone, autoencoder)

    def _wrap_model(self, model, device: str, norm_stats, x_axis, amp_max: float) -> ModelWrapper:
        adapter = JepaInferenceModel(model, self.profile_normalizer).to(device)
        adapter.eval()
        return ModelWrapper(model=adapter, device=device, params_per_gaussian=3, normalizer=None, x_axis=None, amp_max=None)
