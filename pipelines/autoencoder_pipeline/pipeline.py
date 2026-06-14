from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from configuration.autoencoder_config       import ProfileAeTrainerConfig
from models.autoencoder             import Autoencoder
from pipelines.benchmark_pipeline.config_factory import ConfigFactory
from pipelines.dataset_pipeline.profile_preparation import ProfileDatasetPreparation
from pipelines.backbone_pipeline.pipeline   import TrainingRunMetadata
from pipelines.autoencoder_pipeline.autoencoder_trainer import ProfileAeTrainer
from pipelines.autoencoder_pipeline.profile_dataset     import ProfileDataset
from tools.data.io                               import AutoencoderConfigIO
from tools.reproducibility                  import Reproducibility


class ProfileAePipeline:
    def __init__(self, entry_config, split_regions=None) -> None:
        self.entry   = entry_config
        self.factory = ConfigFactory(entry_config)
        Reproducibility.seed_everything(entry_config.seed)

        base = self.factory.training_trainer_config(logdir=entry_config.logdir)

        self.autoencoder_cfg = entry_config.autoencoder

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

    def _build_model(self, x_len: int) -> Autoencoder:
        self.autoencoder_cfg.profile_length = x_len
        return Autoencoder(self.autoencoder_cfg)

    def _profile_loaders(self, datasets, x_axis):
        gaussian_cfg = self.trainer_config.gaussian
        train_loader = self._profile_loader(datasets["train"], x_axis, gaussian_cfg, shuffle=True)
        val_loader   = self._profile_loader(datasets["val"],   x_axis, gaussian_cfg, shuffle=False)
        return train_loader, val_loader

    def _profile_loader(self, patch_ds, x_axis, gaussian_cfg, shuffle: bool) -> DataLoader:
        profile_ds = ProfileDataset.from_patch_dataset(
            patch_ds, x_axis, gaussian_cfg.n_default_gaussians,
            pixel_subsample = self.entry.pixel_subsample,
            keep_empty_frac = self.entry.keep_empty_frac,
            seed            = self.entry.seed,
        )
        return DataLoader(profile_ds, batch_size=self.dataset_config.batch_size, shuffle=shuffle,
                          num_workers=self.dataset_config.num_workers, pin_memory=self.dataset_config.pin_memory, drop_last=False)

    def _save_metadata(self, run_meta, x_len: int) -> None:
        run_meta.save_trainer_config()
        AutoencoderConfigIO.save(self.autoencoder_cfg, run_meta.metadata_directory)
        run_meta.save_run_summary("profile_ae", in_channels=x_len, out_channels=self.autoencoder_cfg.embedding_dim, x_axis_length=x_len)

    def _train(self, run_meta, logger, model, x_axis, train_loader, val_loader):
        trainer = ProfileAeTrainer(model, self.autoencoder_cfg, x_axis, self.trainer_config, run_meta.run_directory, logger)
        try:
            results = trainer.train(train_loader, val_loader, val_loader)
        finally:
            run_meta.close()
            logger.close()
        return results, run_meta.run_directory

    def run(self):
        run_meta = TrainingRunMetadata(self.trainer_config, "profile_ae", Path(self.trainer_config.io.logdir), self.entry.run_name)
        logger   = run_meta.logger

        _, datasets, x_axis, x_len = ProfileDatasetPreparation(self.dataset_config, self.trainer_config, run_meta, logger, self.entry.seed).run()

        model                    = self._build_model(x_len)
        train_loader, val_loader = self._profile_loaders(datasets, x_axis)

        self._save_metadata(run_meta, x_len)

        return self._train(run_meta, logger, model, x_axis, train_loader, val_loader)


class SingleProfileAeRunner:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        return ProfileAePipeline(self.config).run()
