from __future__ import annotations

from dataclasses import replace
from pathlib     import Path

import torch
import torch.nn as nn

from models                                   import get_model
from models.profile_autoencoder               import ProfileAutoencoder
from pipelines.dataset_pipeline.normalization import Normalizer
from pipelines.inference_pipeline.loader      import ModelWrapper, RunLoader
from pipelines.inference_pipeline.pipeline    import InferencePipeline
from pipelines.jepa_pipeline.pipeline         import JepaPipelineSupport
from pipelines.jepa_pipeline.predictor_trainer import JepaModule

_IMAGE_SIZE_MODELS = {"swin_unet", "transunet", "unetr"}


class JepaInferenceModel(nn.Module):
    def __init__(self, jepa_module: JepaModule) -> None:
        super().__init__()
        self.jepa = jepa_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_hat = self.jepa.backbone(x)
        return self.jepa.autoencoder.heads(z_hat)


class JepaRunLoader(RunLoader):
    def _build_model(self, model_name: str, in_channels: int, out_channels: int, image_size: int):
        ae_cfg = JepaPipelineSupport.load_autoencoder_config(self.meta_directory)

        overrides = {"in_channels": in_channels, "out_channels": ae_cfg.embedding_dim}
        if model_name in _IMAGE_SIZE_MODELS:
            overrides["image_size"] = image_size
        backbone, _ = get_model(model_name, **overrides)

        autoencoder = ProfileAutoencoder(ae_cfg)
        return JepaModule(backbone, autoencoder)

    def _wrap_model(self, model, device: str, norm_stats, x_axis, amp_max: float) -> ModelWrapper:
        adapter = JepaInferenceModel(model).to(device)
        adapter.eval()
        return ModelWrapper(
            model               = adapter,
            device              = device,
            params_per_gaussian = 3,
            normalizer          = Normalizer(norm_stats),
            x_axis              = torch.from_numpy(x_axis),
            amp_max             = amp_max,
        )


class JepaInferencePipeline(InferencePipeline):
    def __init__(self, config, run_directory: Path) -> None:
        super().__init__(replace(config, run_directory=Path(run_directory), output_subdir=None))

    def _load_run(self, cfg, logger):
        loader = JepaRunLoader(cfg.run_directory, logger=logger)
        return loader.load(
            split           = cfg.split,
            batch_size      = cfg.batch_size,
            num_workers     = cfg.num_workers,
            device          = cfg.device,
            checkpoint_name = cfg.checkpoint_name,
        )
