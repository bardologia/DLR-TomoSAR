from __future__ import annotations

from pathlib import Path

from configuration.data.profile_config               import ProfileDatasetConfig
from configuration.training.autoencoder_config       import ProfileAeTrainerConfig
from models.autoencoder             import get_autoencoder
from pipelines.profile_autoencoder.dataset.pipeline import ProfileDatasetPipeline
from pipelines.shared.config_factory import ConfigFactory
from pipelines.shared.run_metadata import TrainingRunMetadata
from pipelines.profile_autoencoder.training.trainer import Trainer
from tools.data.io                               import AutoencoderConfigIO
from tools.runtime.reproducibility                  import Reproducibility


class TrainingPipeline:
    def __init__(self, entry_config, split_regions=None) -> None:
        self.entry   = entry_config
        self.factory = ConfigFactory(entry_config)
        Reproducibility.seed_everything(entry_config.seed)

        base = self.factory.training_trainer_config(logdir=entry_config.logdir)

        self.autoencoder_cfg = entry_config.autoencoder
        self.ae_model_name   = entry_config.ae_model_name

        self.trainer_config = ProfileAeTrainerConfig(
            gaussian    = base.gaussian,
            autoencoder = self.autoencoder_cfg,
            ae_loss     = entry_config.ae_loss,
            overfit     = entry_config.overfit,
        )
        self.trainer_config.inherit_shared_from(base)
        self.trainer_config.geometry = entry_config.geometry.resolved(entry_config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        self.dataset_config = self.factory.training_dataset_config()
        if split_regions is not None:
            self.dataset_config.split_regions = split_regions

    def _build_model(self, x_len: int):
        self.autoencoder_cfg.profile_length = x_len
        model, _ = get_autoencoder(self.ae_model_name, self.autoencoder_cfg)
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
            pin_memory                  = ds.pin_memory,
            shuffle_train               = ds.shuffle_train,
            augmentation                = self.entry.profile_augmentation,
        )

    def _save_metadata(self, run_meta, x_len: int) -> None:
        run_meta.save_trainer_config()
        AutoencoderConfigIO.save(self.autoencoder_cfg, self.ae_model_name, run_meta.metadata_directory)
        run_meta.save_run_summary("profile_ae", in_channels=x_len, out_channels=self.autoencoder_cfg.embedding_dim, x_axis_length=x_len)

    def _make_trainer(self, run_meta, logger, model, x_axis) -> Trainer:
        return Trainer(model, self.autoencoder_cfg, x_axis, self.trainer_config, run_meta.run_directory, logger)

    def _train(self, run_meta, logger, model, x_axis, train_loader, val_loader):
        trainer = self._make_trainer(run_meta, logger, model, x_axis)
        try:
            results = trainer.train(train_loader, val_loader, val_loader)
        finally:
            run_meta.close()
        return results, run_meta.run_directory

    def run(self):
        run_meta = TrainingRunMetadata(self.trainer_config, "profile_ae", Path(self.trainer_config.io.logdir), self.entry.run_name)
        logger   = run_meta.logger

        profile_config   = self._profile_dataset_config()
        dataset_pipeline = ProfileDatasetPipeline(profile_config, run_meta.run_directory, logger=logger, seed=self.entry.seed)

        (train_loader, val_loader, _test_loader), _datasets, x_axis, x_len, _normalizer = dataset_pipeline.run()

        model = self._build_model(x_len)

        self._save_metadata(run_meta, x_len)

        return self._train(run_meta, logger, model, x_axis, train_loader, val_loader)


class SingleTrainRunner:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        return TrainingPipeline(self.config).run()
