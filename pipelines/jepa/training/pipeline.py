from __future__ import annotations

from pathlib import Path

import torch

from configuration.training.jepa_config                  import JepaTrainerConfig
from models                                              import IMAGE_SIZE_MODELS, get_model
from models.autoencoder                                  import get_autoencoder
from models.image_autoencoder                            import get_image_autoencoder
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats
from pipelines.shared.config_factory                     import ConfigFactory
from pipelines.shared.run_metadata                       import TrainingRunMetadata
from pipelines.shared.dataset_prep                        import BackboneDatasetPreparation
from pipelines.jepa.training.trainer                     import JepaModule, Trainer
from tools.data.io                                       import AutoencoderConfigIO, ImageAutoencoderConfigIO
from tools.runtime.reproducibility                       import Reproducibility



class TrainingPipeline:
    def __init__(self, entry_config, split_regions=None) -> None:
        self.entry   = entry_config
        self.factory = ConfigFactory(entry_config)
        Reproducibility.seed_everything(entry_config.seed)

        base = self.factory.training_trainer_config(logdir=entry_config.logdir)

        profile_dir = self._resolve_ae_run(entry_config.profile_autoencoder_logdir, entry_config.profile_autoencoder_run, "profile")
        image_dir   = self._resolve_ae_run(entry_config.image_autoencoder_logdir,   entry_config.image_autoencoder_run,   "image")

        if profile_dir is None and image_dir is None:
            raise ValueError("JEPA requires at least one of profile_autoencoder_run or image_autoencoder_run; with neither, train the plain backbone with '--mode backbone'.")

        self.profile_autoencoder_meta = None
        self.autoencoder_cfg          = None
        self.ae_model_name            = None
        profile_checkpoint            = None
        if profile_dir is not None:
            self.profile_autoencoder_meta            = profile_dir / "meta"
            self.autoencoder_cfg, self.ae_model_name = AutoencoderConfigIO.load(self.profile_autoencoder_meta)
            profile_checkpoint                       = str(profile_dir / "best_model.pt")

        self.image_autoencoder_meta = None
        self.image_ae_cfg           = None
        self.image_ae_model_name    = None
        image_checkpoint            = None
        if image_dir is not None:
            self.image_autoencoder_meta                 = image_dir / "meta"
            self.image_ae_cfg, self.image_ae_model_name = ImageAutoencoderConfigIO.load(self.image_autoencoder_meta)
            image_checkpoint                            = str(image_dir / "best_model.pt")

        self.trainer_config = JepaTrainerConfig(
            gaussian                       = base.gaussian,
            autoencoder                    = self.autoencoder_cfg,
            embedding_loss                 = entry_config.embedding_loss,
            profile_autoencoder_mode       = entry_config.profile_autoencoder_mode,
            target_provider                = entry_config.target_provider,
            profile_autoencoder_checkpoint = profile_checkpoint,
            image_autoencoder              = self.image_ae_cfg,
            image_autoencoder_mode         = entry_config.image_autoencoder_mode,
            image_autoencoder_checkpoint   = image_checkpoint,
            param_loss                     = entry_config.param_loss,
            overfit                        = entry_config.overfit,
        )
        self.trainer_config.inherit_shared_from(base)
        self.trainer_config.geometry = entry_config.geometry.resolved(entry_config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        self.dataset_config = self.factory.training_dataset_config()
        if split_regions is not None:
            self.dataset_config.split_regions = split_regions

        self.model_name = entry_config.model_name

    @staticmethod
    def _resolve_ae_run(logdir, run, label):
        if not run:
            return None
        directory = Path(logdir) / run
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} autoencoder run '{run}' not found under {logdir}")
        return directory

    def _gaussian_out_channels(self) -> int:
        g = self.trainer_config.gaussian
        return g.params_per_gaussian * g.n_default_gaussians

    def _build_module(self, datasets, x_len: int):
        dataset_in_channels = datasets["train"].input_channels

        image_autoencoder = None
        backbone_in       = dataset_in_channels
        if self.image_ae_cfg is not None:
            if self.image_ae_cfg.in_channels != dataset_in_channels:
                raise ValueError(f"Image autoencoder was trained with in_channels={self.image_ae_cfg.in_channels} but the JEPA dataset produces {dataset_in_channels} input channels; the input representation and secondaries must match the image autoencoder run.")
            image_autoencoder = self._load_image_autoencoder()
            backbone_in       = self.image_ae_cfg.embedding_dim

        profile_autoencoder = None
        if self.autoencoder_cfg is not None:
            self.autoencoder_cfg.profile_length = x_len
            backbone_out                        = self.autoencoder_cfg.embedding_dim
            profile_autoencoder                 = self._load_profile_autoencoder()
        else:
            backbone_out = self._gaussian_out_channels()

        backbone, backbone_cfg = self._build_backbone(backbone_in, backbone_out, x_len)
        return JepaModule(backbone, profile_autoencoder=profile_autoencoder, image_autoencoder=image_autoencoder), backbone_cfg

    def _build_backbone(self, in_channels: int, out_channels: int, image_size: int):
        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        if self.model_name in IMAGE_SIZE_MODELS:
            overrides["image_size"] = image_size
        for k, v in self.entry.model_overrides.items():
            overrides[k] = v
        return get_model(self.model_name, **overrides)

    def _load_profile_autoencoder(self):
        autoencoder, _ = get_autoencoder(self.ae_model_name, self.autoencoder_cfg)
        ckpt_path      = self.trainer_config.profile_autoencoder_checkpoint
        self.validate_checkpoint(ckpt_path, "profile")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        autoencoder.load_state_dict(ckpt["params"])
        return autoencoder

    def _load_image_autoencoder(self):
        autoencoder, _ = get_image_autoencoder(self.image_ae_model_name, self.image_ae_cfg)
        ckpt_path      = self.trainer_config.image_autoencoder_checkpoint
        self.validate_checkpoint(ckpt_path, "image")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        autoencoder.load_state_dict(ckpt["params"])
        return autoencoder

    @staticmethod
    def validate_checkpoint(ckpt_path, label: str) -> None:
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"{label} autoencoder checkpoint '{ckpt_path}' does not exist; expected 'best_model.pt' under the selected {label} autoencoder run directory.")

    def _save_metadata(self, run_meta, backbone_cfg, datasets, x_len: int) -> None:
        gaussian_cfg = self.trainer_config.gaussian
        in_channels  = datasets["train"].input_channels

        run_meta.save_trainer_config()
        run_meta.save_model_config(backbone_cfg, self.model_name)

        if self.autoencoder_cfg is not None:
            AutoencoderConfigIO.save(self.autoencoder_cfg, self.ae_model_name, run_meta.metadata_directory)
        if self.image_ae_cfg is not None:
            ImageAutoencoderConfigIO.save(self.image_ae_cfg, self.image_ae_model_name, run_meta.metadata_directory)

        run_meta.save_run_summary(self.model_name, in_channels=in_channels, out_channels=gaussian_cfg.params_per_gaussian * gaussian_cfg.n_default_gaussians, x_axis_length=x_len)

    def _profile_normalizer(self, run_meta, logger):
        if self.autoencoder_cfg is None:
            return None

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

        loaders, datasets, x_axis, x_len = BackboneDatasetPreparation(self.dataset_config, self.trainer_config, run_meta, logger, self.entry.seed).run()

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

            inference_config = replace(self.config.inference, run_directory=Path(run_directory), output_subdir=None)
            InferencePipeline(inference_config).run()
