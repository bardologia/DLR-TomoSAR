from __future__ import annotations

from pathlib import Path

from configuration.training              import OverfitConfig
from pipelines.shared.config.config_factory import ConfigFactory
from pipelines.shared.config.run_metadata   import TrainingRunMetadata
from tools.runtime.reproducibility   import Reproducibility


class AutoencoderTrainingPipeline:
    run_label     = None
    trainer_class = None

    def __init__(self, entry_config, split_regions=None, overfit=None) -> None:
        self.entry   = entry_config
        self.overfit = overfit if overfit is not None else OverfitConfig(enabled=False)
        self.factory = ConfigFactory(entry_config)
        Reproducibility.seed_everything(entry_config.seed)

        base = self.factory.training_trainer_config(logdir=entry_config.logdir)

        self.autoencoder_cfg = self._autoencoder_config(entry_config)
        self.ae_model_name   = entry_config.ae_model_name

        self.trainer_config          = self._build_trainer_config(base, entry_config)
        self.trainer_config.geometry = entry_config.geometry.resolved(entry_config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        self.dataset_config = self.factory.training_dataset_config()
        if split_regions is not None:
            self.dataset_config.split_regions = split_regions

    def _autoencoder_config(self, entry_config):
        raise NotImplementedError

    def _build_trainer_config(self, base, entry_config):
        raise NotImplementedError

    def _build_model(self, model_dim: int):
        raise NotImplementedError

    def _prepare_data(self, run_meta, logger):
        raise NotImplementedError

    def _save_metadata(self, run_meta, *args) -> None:
        raise NotImplementedError

    def _make_trainer(self, run_meta, logger, model, x_axis):
        return self.trainer_class(model, self.autoencoder_cfg, x_axis, self.trainer_config, run_meta.run_directory, logger)

    def _train(self, run_meta, logger, model, x_axis, train_loader, val_loader):
        trainer = self._make_trainer(run_meta, logger, model, x_axis)
        try:
            results = trainer.train(train_loader, val_loader, val_loader)
        finally:
            run_meta.close()
        return results, run_meta.run_directory

    def run(self):
        run_meta = TrainingRunMetadata(self.trainer_config, self.run_label, Path(self.trainer_config.io.logdir), self.entry.run_name)
        logger   = run_meta.logger

        train_loader, val_loader, x_axis, model_dim, metadata_args = self._prepare_data(run_meta, logger)

        model = self._build_model(model_dim)

        self._save_metadata(run_meta, *metadata_args)

        return self._train(run_meta, logger, model, x_axis, train_loader, val_loader)
