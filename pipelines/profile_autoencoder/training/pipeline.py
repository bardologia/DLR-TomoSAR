from __future__ import annotations

from pathlib import Path

from configuration.dataset import ProfileDatasetConfig
from configuration.training import ProfileAeTrainerConfig
from models.profile_autoencoder                             import PROFILE_AE_CONFIG_REGISTRY, get_profile_autoencoder
from pipelines.autoencoder_common.training          import AutoencoderTrainingPipeline
from pipelines.profile_autoencoder.dataset.pipeline import ProfileDatasetPipeline
from pipelines.shared.model.model_builder                    import ModelBuilder
from pipelines.shared.training.training_runner               import EntryConfigTrainRunner
from pipelines.profile_autoencoder.training.trainer import Trainer
from pipelines.shared.config.config_persistence            import ProfileAutoencoderConfigIO, ProfileDatasetConfigIO


class TrainingPipeline(AutoencoderTrainingPipeline):
    run_label     = "profile_ae"
    trainer_class = Trainer

    def _autoencoder_config(self, entry_config):
        return ModelBuilder.config_from_registry(entry_config.ae_model_name, entry_config.model_overrides, registry=PROFILE_AE_CONFIG_REGISTRY)

    def _build_trainer_config(self, base, entry_config):
        trainer_config = ProfileAeTrainerConfig(
            gaussian    = base.gaussian,
            autoencoder = self.autoencoder_cfg,
            ae_loss     = entry_config.ae_loss,
            overfit     = self.overfit,
        )
        trainer_config.inherit_shared_from(base)
        return trainer_config

    def _build_model(self, x_len: int):
        self.autoencoder_cfg.profile_length = x_len
        model, _ = get_profile_autoencoder(self.ae_model_name, self.autoencoder_cfg)
        return model

    def _profile_dataset_config(self) -> ProfileDatasetConfig:
        gaussian_cfg = self.trainer_config.gaussian
        ds           = self.dataset_config

        return ProfileDatasetConfig(
            preprocessing_run_directory = ds.preprocessing_run_directory,
            split_regions               = ds.split_regions,
            parameters_path             = ds.parameters_path,
            n_gaussians                 = gaussian_cfg.n_default_gaussians,
            x_min                       = gaussian_cfg.x_min,
            x_max                       = gaussian_cfg.x_max,
            pixel_subsample             = self.entry.pixel_subsample,
            keep_empty_frac             = self.entry.keep_empty_frac,
            batch_size                  = ds.batch_size,
            num_workers                 = ds.num_workers,
            prefetch_factor             = ds.prefetch_factor,
            pin_memory                  = ds.pin_memory,
            shuffle_train               = ds.shuffle_train,
            augmentation                = self.entry.profile_augmentation,
        )

    def _prepare_data(self, run_meta, logger):
        profile_config   = self._profile_dataset_config()
        dataset_pipeline = ProfileDatasetPipeline(profile_config, run_meta.run_directory, logger=logger, seed=self.entry.seed)

        (train_loader, val_loader, _test_loader), _datasets, x_axis, x_len, _normalizer = dataset_pipeline.run()

        return train_loader, val_loader, x_axis, x_len, (profile_config, x_len)

    def _save_metadata(self, run_meta, profile_config: ProfileDatasetConfig, x_len: int) -> None:
        run_meta.save_trainer_config()
        ProfileAutoencoderConfigIO.save(self.autoencoder_cfg, self.ae_model_name, run_meta.metadata_directory)
        ProfileDatasetConfigIO.save(profile_config, run_meta.metadata_directory)
        run_meta.save_run_summary("profile_ae", in_channels=x_len, out_channels=self.autoencoder_cfg.embedding_dim, x_axis_length=x_len)


class SingleTrainRunner(EntryConfigTrainRunner):
    pipeline_class = TrainingPipeline

    @property
    def label(self) -> str:
        return self.config.ae_model_name

    def _build_pretrain_trainer(self, logger):
        work_dir = Path(self.config.logdir) / "pretrain" / "context"
        pipeline = TrainingPipeline(self.config)

        profile_config   = pipeline._profile_dataset_config()
        dataset_pipeline = ProfileDatasetPipeline(profile_config, work_dir, logger=logger, seed=self.config.seed)

        (_loaders, datasets, x_axis, x_len, _normalizer) = dataset_pipeline.run()

        model   = pipeline._build_model(x_len)
        trainer = Trainer(model, pipeline.autoencoder_cfg, x_axis, pipeline.trainer_config, work_dir, logger)

        return trainer, datasets["train"], model
