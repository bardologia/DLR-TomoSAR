from __future__ import annotations

from pathlib import Path

import torch

from configuration.training.jepa_config                  import JepaTrainerConfig
from models                                              import IMAGE_SIZE_MODELS, get_model
from models.autoencoder                                  import get_autoencoder
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats
from pipelines.shared.config_factory                     import ConfigFactory
from pipelines.shared.run_metadata                       import TrainingRunMetadata
from pipelines.jepa.training.dataset_prep                import ProfileDatasetPreparation
from pipelines.jepa.training.trainer                     import JepaModule, Trainer
from tools.data.io                                       import AutoencoderConfigIO
from tools.runtime.reproducibility                       import Reproducibility



class TrainingPipeline:
    def __init__(self, entry_config, split_regions=None) -> None:
        self.entry   = entry_config
        self.factory = ConfigFactory(entry_config)
        Reproducibility.seed_everything(entry_config.seed)

        base = self.factory.training_trainer_config(logdir=entry_config.logdir)

        if not entry_config.profile_autoencoder_run:
            raise ValueError("JEPA training requires a pretrained profile autoencoder; set profile_autoencoder_run to a completed autoencoder run under profile_autoencoder_logdir. Training the autoencoder jointly with the backbone is not supported.")

        profile_autoencoder_dir = Path(entry_config.profile_autoencoder_logdir) / entry_config.profile_autoencoder_run
        if not profile_autoencoder_dir.is_dir():
            raise FileNotFoundError(f"Profile autoencoder run '{entry_config.profile_autoencoder_run}' not found under {entry_config.profile_autoencoder_logdir}")

        self.profile_autoencoder_meta = profile_autoencoder_dir / "meta"
        self.autoencoder_cfg, self.ae_model_name = AutoencoderConfigIO.load(self.profile_autoencoder_meta)

        self.trainer_config = JepaTrainerConfig(
            gaussian                       = base.gaussian,
            autoencoder                    = self.autoencoder_cfg,
            embedding_loss                 = entry_config.embedding_loss,
            profile_autoencoder_mode       = entry_config.profile_autoencoder_mode,
            target_provider                = entry_config.target_provider,
            profile_autoencoder_checkpoint = str(profile_autoencoder_dir / "best_model.pt"),
            overfit                        = entry_config.overfit,
        )
        self.trainer_config.inherit_shared_from(base)
        self.trainer_config.geometry = entry_config.geometry.resolved(entry_config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        self.dataset_config = self.factory.training_dataset_config()
        if split_regions is not None:
            self.dataset_config.split_regions = split_regions

        self.model_name     = entry_config.model_name

    def _build_module(self, datasets, x_len: int):
        in_channels   = datasets["train"].input_channels

        self.autoencoder_cfg.profile_length = x_len
        embedding_dim                       = self.autoencoder_cfg.embedding_dim

        backbone, backbone_cfg = self._build_backbone(in_channels, embedding_dim, x_len)
        autoencoder            = self._load_autoencoder()
        return JepaModule(backbone, autoencoder), backbone_cfg

    def _build_backbone(self, in_channels: int, embedding_dim: int, image_size: int):
        overrides = {"in_channels": in_channels, "out_channels": embedding_dim}
        if self.model_name in IMAGE_SIZE_MODELS:
            overrides["image_size"] = image_size
        for k, v in self.entry.model_overrides.items():
            overrides[k] = v
        return get_model(self.model_name, **overrides)

    def _load_autoencoder(self):
        autoencoder, _ = get_autoencoder(self.ae_model_name, self.autoencoder_cfg)
        ckpt_path      = self.trainer_config.profile_autoencoder_checkpoint
        self.validate_profile_autoencoder_checkpoint(ckpt_path)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        autoencoder.load_state_dict(ckpt["params"])
        return autoencoder

    @staticmethod
    def validate_profile_autoencoder_checkpoint(ckpt_path) -> None:
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Profile autoencoder checkpoint '{ckpt_path}' does not exist; expected 'best_model.pt' under the selected profile_autoencoder_run directory.")

    def _save_metadata(self, run_meta, backbone_cfg, datasets, x_len: int) -> None:
        gaussian_cfg = self.trainer_config.gaussian
        in_channels  = datasets["train"].input_channels

        run_meta.save_trainer_config()
        run_meta.save_model_config(backbone_cfg, self.model_name)
        AutoencoderConfigIO.save(self.autoencoder_cfg, self.ae_model_name, run_meta.metadata_directory)
        run_meta.save_run_summary(self.model_name, in_channels=in_channels, out_channels=gaussian_cfg.params_per_gaussian * gaussian_cfg.n_default_gaussians, x_axis_length=x_len)

    def _profile_normalizer(self, run_meta, logger):
        stats = ProfileStats.load(self.profile_autoencoder_meta, logger=logger)
        stats.save(run_meta.metadata_directory)

        return ProfileNormalizer(stats)

    def _make_trainer(self, model, backbone_cfg, x_axis, run_dir, logger, norm_stats, profile_normalizer):
        return Trainer(model, backbone_cfg, x_axis, self.trainer_config, run_dir, logger, norm_stats, profile_normalizer)

    def _train(self, run_meta, logger, model, backbone_cfg, x_axis, datasets, loaders, profile_normalizer):
        train_loader, val_loader, test_loader = loaders
        norm_stats                            = datasets["train"].normalizer

        trainer = self._make_trainer(model, backbone_cfg, x_axis, run_meta.run_directory, logger, norm_stats, profile_normalizer)
        try:
            results = trainer.train(train_loader, val_loader, test_loader)
        finally:
            run_meta.close()
        return results, run_meta.run_directory

    def run(self):
        run_meta = TrainingRunMetadata(self.trainer_config, self.model_name, Path(self.trainer_config.io.logdir), self.entry.run_name)
        logger   = run_meta.logger

        loaders, datasets, x_axis, x_len = ProfileDatasetPreparation(self.dataset_config, self.trainer_config, run_meta, logger, self.entry.seed).run()

        model, backbone_cfg = self._build_module(datasets, x_len)

        self._save_metadata(run_meta, backbone_cfg, datasets, x_len)

        profile_normalizer = self._profile_normalizer(run_meta, logger)

        return self._train(run_meta, logger, model, backbone_cfg, x_axis, datasets, loaders, profile_normalizer)


class SingleTrainRunner:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        results, run_directory = TrainingPipeline(self.config).run()
        if self.config.infer_after:
            from dataclasses                           import replace
            from pipelines.backbone.inference.pipeline import InferencePipeline
            from pipelines.jepa.inference.pipeline     import JEPA_INFERENCE_COMPONENTS

            inference_config = replace(self.config.inference, run_directory=Path(run_directory), output_subdir=None)
            InferencePipeline(inference_config, components=JEPA_INFERENCE_COMPONENTS).run()
