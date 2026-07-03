from __future__ import annotations

from pathlib import Path

from configuration.training import ImageAeTrainerConfig
from models.image_autoencoder                        import IMAGE_AE_CONFIG_REGISTRY, get_image_autoencoder
from pipelines.autoencoder_common.training           import AutoencoderTrainingPipeline
from pipelines.backbone.dataset.pipeline             import DatasetPipeline
from pipelines.shared.dataset.dataset_prep                   import BackboneDatasetPreparation
from pipelines.shared.model.model_builder                     import ModelBuilder
from pipelines.shared.training.training_runner                import EntryConfigTrainRunner
from pipelines.image_autoencoder.training.trainer    import Trainer
from pipelines.shared.config.config_persistence             import ImageAutoencoderConfigIO
from tools.data.gaussians                            import GaussianAxis


class TrainingPipeline(AutoencoderTrainingPipeline):
    run_label       = "image_ae"
    trainer_class   = Trainer
    model_dim_label = "In Channels"

    def _autoencoder_config(self, entry_config):
        return ModelBuilder.config_from_registry(entry_config.ae_model_name, entry_config.model_overrides, registry=IMAGE_AE_CONFIG_REGISTRY)

    def _build_trainer_config(self, base, entry_config):
        trainer_config = ImageAeTrainerConfig(
            gaussian          = base.gaussian,
            image_autoencoder = self.autoencoder_cfg,
            ae_loss           = entry_config.ae_loss,
            overfit           = self.overfit,
        )
        trainer_config.inherit_shared_from(base)
        return trainer_config

    def _build_model(self, in_channels: int, config):
        config.in_channels = in_channels
        model, _ = get_image_autoencoder(self.ae_model_name, config)
        return model

    def _prepare_data(self, run_meta, logger):
        loaders, datasets, x_axis, x_len      = BackboneDatasetPreparation(self.dataset_config, self.trainer_config, run_meta, logger, self.entry.seed).run()
        train_loader, val_loader, test_loader = loaders

        in_channels = datasets["train"].input_channels

        return train_loader, val_loader, test_loader, x_axis, in_channels, datasets, (in_channels, x_len)

    def _save_metadata(self, run_meta, in_channels: int, x_len: int) -> None:
        run_meta.save_trainer_config()
        ImageAutoencoderConfigIO.save(self.autoencoder_cfg, self.ae_model_name, run_meta.metadata_directory)
        run_meta.save_run_summary("image_ae", in_channels=in_channels, out_channels=self.autoencoder_cfg.embedding_dim, x_axis_length=x_len, seed=self.entry.seed)


class SingleTrainRunner(EntryConfigTrainRunner):
    pipeline_class = TrainingPipeline

    @property
    def label(self) -> str:
        return self.config.ae_model_name

    def _build_pretrain_trainer(self, logger):
        work_dir = Path(self.config.logdir) / "pretrain" / "context"
        pipeline = TrainingPipeline(self.config)

        dataset_config             = pipeline.dataset_config
        gaussian_config            = pipeline.trainer_config.gaussian
        dataset_config.n_gaussians = gaussian_config.n_default_gaussians

        dataset_pipeline      = DatasetPipeline(dataset_config, work_dir, logger=logger, seed=self.config.seed)
        profile_length        = dataset_pipeline.profile_length
        dataset_config.x_axis = GaussianAxis.build(gaussian_config.x_min, gaussian_config.x_max, profile_length)

        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()

        in_channels = datasets["train"].input_channels
        model       = pipeline._build_model(in_channels, pipeline.autoencoder_cfg)
        trainer     = Trainer(model, pipeline.autoencoder_cfg, dataset_config.x_axis, pipeline.trainer_config, work_dir, logger)

        return trainer, datasets["train"], model
