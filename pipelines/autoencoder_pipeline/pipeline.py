from __future__ import annotations

import json
from dataclasses import asdict
from pathlib     import Path

import numpy as np
from torch.utils.data import DataLoader

from configuration.autoencoder_config       import ProfileAeTrainerConfig, ProfileAutoencoderConfig
from models.profile_autoencoder             import ProfileAutoencoder
from pipelines.benchmark_pipeline.config_factory import ConfigFactory
from pipelines.dataset_pipeline.pipeline    import DatasetPipeline
from pipelines.training_pipeline.pipeline   import TrainingRunMetadata
from pipelines.autoencoder_pipeline.autoencoder_trainer import ProfileAeTrainer
from pipelines.autoencoder_pipeline.profile_dataset     import ProfileDataset
from tools.reproducibility                  import Reproducibility

_SHARED_SUBCONFIGS = (
    "geometry", "early_stopping", "warmup", "scheduler", "io", "optimizer",
    "training", "resources", "gradient_clipper",
)


class AutoencoderPipelineSupport:
    @staticmethod
    def x_axis_length(dataset_pipeline: DatasetPipeline) -> int:
        tomo_path = dataset_pipeline.layout.artifact_path("tomogram_full")
        tomo_mmap = np.load(str(tomo_path), mmap_mode="r", allow_pickle=False)
        return int(tomo_mmap.shape[0])

    @staticmethod
    def copy_base_subconfigs(target, base) -> None:
        for name in _SHARED_SUBCONFIGS:
            setattr(target, name, getattr(base, name))

    @staticmethod
    def save_autoencoder_config(cfg: ProfileAutoencoderConfig, meta_dir: Path) -> Path:
        out = Path(meta_dir) / "autoencoder_config.json"
        out.write_text(json.dumps(asdict(cfg), indent=2) + "\n")
        return out

    @staticmethod
    def load_autoencoder_config(meta_dir: Path) -> ProfileAutoencoderConfig:
        payload = json.loads((Path(meta_dir) / "autoencoder_config.json").read_text())
        return ProfileAutoencoderConfig(**payload)


class ProfileAePipeline:
    def __init__(self, entry_config) -> None:
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
        AutoencoderPipelineSupport.copy_base_subconfigs(self.trainer_config, base)
        self.trainer_config.geometry = entry_config.geometry.resolved(entry_config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        self.dataset_config = self.factory.training_dataset_config()

    def run(self):
        run_meta = TrainingRunMetadata(self.trainer_config, "profile_ae", Path(self.trainer_config.io.logdir), self.entry.run_name)
        logger   = run_meta.logger

        gaussian_cfg = self.trainer_config.gaussian
        self.dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        dataset_pipeline = DatasetPipeline(self.dataset_config, run_meta.run_directory, logger=logger, seed=self.entry.seed)
        x_len            = AutoencoderPipelineSupport.x_axis_length(dataset_pipeline)
        x_axis           = np.linspace(gaussian_cfg.x_min, gaussian_cfg.x_max, x_len, dtype=np.float32)
        self.dataset_config.x_axis = x_axis

        _, _, _, datasets = dataset_pipeline.run()

        self.autoencoder_cfg.profile_length = x_len
        model = ProfileAutoencoder(self.autoencoder_cfg)

        train_loader = self._profile_loader(datasets["train"], x_axis, gaussian_cfg, shuffle=True)
        val_loader   = self._profile_loader(datasets["val"],   x_axis, gaussian_cfg, shuffle=False)

        run_meta.save_trainer_config()
        AutoencoderPipelineSupport.save_autoencoder_config(self.autoencoder_cfg, run_meta.metadata_directory)
        run_meta.save_run_summary("profile_ae", in_channels=x_len, out_channels=self.autoencoder_cfg.embedding_dim, x_axis_length=x_len)

        trainer = ProfileAeTrainer(model, self.autoencoder_cfg, x_axis, self.trainer_config, run_meta.run_directory, logger)
        try:
            results = trainer.train(train_loader, val_loader, val_loader)
        finally:
            run_meta.close()
            logger.close()
        return results, run_meta.run_directory

    def _profile_loader(self, patch_ds, x_axis, gaussian_cfg, shuffle: bool) -> DataLoader:
        profile_ds = ProfileDataset.from_patch_dataset(
            patch_ds, x_axis, gaussian_cfg.n_default_gaussians,
            pixel_subsample = self.entry.pixel_subsample,
            keep_empty_frac = self.entry.keep_empty_frac,
            seed            = self.entry.seed,
        )
        return DataLoader(profile_ds, batch_size=self.dataset_config.batch_size, shuffle=shuffle,
                          num_workers=self.dataset_config.num_workers, pin_memory=self.dataset_config.pin_memory, drop_last=False)


class SingleProfileAeRunner:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        return ProfileAePipeline(self.config).run()
