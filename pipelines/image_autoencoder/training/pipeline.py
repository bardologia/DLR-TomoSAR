from __future__ import annotations

from pathlib import Path

from configuration.training.image_autoencoder_config import ImageAeTrainerConfig
from models.image_autoencoder                        import get_image_autoencoder
from pipelines.shared.config_factory                 import ConfigFactory
from pipelines.shared.dataset_prep                   import BackboneDatasetPreparation
from pipelines.shared.run_metadata                   import TrainingRunMetadata
from pipelines.image_autoencoder.training.trainer    import Trainer
from tools.data.io                                   import ImageAutoencoderConfigIO
from tools.runtime.reproducibility                   import Reproducibility


class TrainingPipeline:
    def __init__(self, entry_config, split_regions=None) -> None:
        self.entry   = entry_config
        self.factory = ConfigFactory(entry_config)
        Reproducibility.seed_everything(entry_config.seed)

        base = self.factory.training_trainer_config(logdir=entry_config.logdir)

        self.autoencoder_cfg = entry_config.image_autoencoder
        self.ae_model_name   = entry_config.ae_model_name

        self.trainer_config = ImageAeTrainerConfig(
            gaussian          = base.gaussian,
            image_autoencoder = self.autoencoder_cfg,
            ae_loss           = entry_config.ae_loss,
            overfit           = entry_config.overfit,
        )
        self.trainer_config.inherit_shared_from(base)
        self.trainer_config.geometry = entry_config.geometry.resolved(entry_config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        self.dataset_config = self.factory.training_dataset_config()
        if split_regions is not None:
            self.dataset_config.split_regions = split_regions

    def _build_model(self, in_channels: int):
        self.autoencoder_cfg.in_channels = in_channels
        model, _ = get_image_autoencoder(self.ae_model_name, self.autoencoder_cfg)
        return model

    def _save_metadata(self, run_meta, in_channels: int, x_len: int) -> None:
        run_meta.save_trainer_config()
        ImageAutoencoderConfigIO.save(self.autoencoder_cfg, self.ae_model_name, run_meta.metadata_directory)
        run_meta.save_run_summary("image_ae", in_channels=in_channels, out_channels=self.autoencoder_cfg.embedding_dim, x_axis_length=x_len)

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
        run_meta = TrainingRunMetadata(self.trainer_config, "image_ae", Path(self.trainer_config.io.logdir), self.entry.run_name)
        logger   = run_meta.logger

        loaders, datasets, x_axis, x_len       = BackboneDatasetPreparation(self.dataset_config, self.trainer_config, run_meta, logger, self.entry.seed).run()
        train_loader, val_loader, _test_loader = loaders

        in_channels = datasets["train"].input_channels
        model       = self._build_model(in_channels)

        self._save_metadata(run_meta, in_channels, x_len)

        return self._train(run_meta, logger, model, x_axis, train_loader, val_loader)


class SingleTrainRunner:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        return TrainingPipeline(self.config).run()
