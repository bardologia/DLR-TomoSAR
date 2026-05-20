from __future__ import annotations

import json
from dataclasses import asdict
from datetime    import datetime
from pathlib     import Path

from torch.utils.tensorboard import SummaryWriter

from configuration.dataset_config import DatasetCreationConfiguration
from pipelines.dataset_pipeline.pipeline    import DatasetCreationPipeline
from tools.logger                                    import Logger

from .config    import AutoencoderConfig
from .data      import LoaderBuilder, ProfileDataset
from .inference import Inference
from .model     import Autoencoder
from .plotter   import Plotter
from .reporter  import Reporter
from .trainer   import Trainer


class AutoencoderPipeline:

    def __init__(
        self,
        ae_config      : AutoencoderConfig,
        dataset_config : DatasetCreationConfiguration,
        run_name       : str | None = None,
    ) -> None:
        self.ae_config      = ae_config
        self.dataset_config = dataset_config

        self.run_directory = self._resolve_run_directory(run_name)
        self._provision_directories()

        self.logger = Logger(
            log_dir = str(self.logs_directory),
            name    = "autoencoder",
            level   = "INFO",
        )
        self.writer  = SummaryWriter(log_dir=str(self.tensorboard_directory))
        self.plotter = Plotter(
            images_dir = self.images_directory,
            embed_dir  = self.embeddings_directory,
            recon_dir  = self.recon_directory,
        )

        ae_config.io.logdir         = str(self.run_directory)
        ae_config.io.tb_dir         = str(self.tensorboard_directory)
        ae_config.io.docs_dir       = str(self.docs_directory)
        ae_config.io.logs_dir       = str(self.logs_directory)
        ae_config.io.images_dir     = str(self.images_directory)
        ae_config.io.embed_dir      = str(self.embeddings_directory)
        ae_config.io.recon_dir      = str(self.recon_directory)
        ae_config.io.checkpoint_dir = str(self.checkpoint_directory)
        ae_config.io.report_path    = str(self.run_directory / "report.md")

        self.logger.section("[AutoencoderPipeline Initialized]")
        self.logger.kv_table(
            {
                "Run Directory" : str(self.run_directory),
                "Latent Dim"    : ae_config.latent_dim,
                "Encoder"       : ae_config.encoder.backbone.value,
                "Decoder"       : ae_config.decoder.backbone.value,
            },
            title="Autoencoder Pipeline",
        )

        self.dataset_pipeline = DatasetCreationPipeline(
            config                 = dataset_config,
            training_run_directory = self.run_directory,
            logger                 = self.logger,
        )

    def _resolve_run_directory(self, run_name: str | None) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name      = run_name or self.ae_config.io.run_name or f"run_autoencoder_{timestamp}"
        return Path(self.ae_config.io.logdir) / name

    def _provision_directories(self) -> None:
        self.tensorboard_directory = self.run_directory / "tensorboard"
        self.docs_directory        = self.run_directory / "docs"
        self.logs_directory        = self.run_directory / "logs"
        self.images_directory      = self.run_directory / "images"
        self.metadata_directory    = self.run_directory / "meta"
        self.embeddings_directory  = self.run_directory / "embeddings"
        self.recon_directory       = self.run_directory / "reconstructions"
        self.checkpoint_directory  = self.run_directory / "checkpoints"

        for directory in (
            self.run_directory, self.tensorboard_directory, self.docs_directory,
            self.logs_directory, self.images_directory, self.metadata_directory,
            self.embeddings_directory, self.recon_directory, self.checkpoint_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _save_configs(self) -> None:
        with open(self.docs_directory / "ae_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.ae_config), f, indent=4, default=str)
        self.logger.subsection("-> AE config saved: docs/ae_config.json")

    def _save_run_summary(self,
                          profile_length : int,
                          num_train      : int,
                          num_val        : int,
                          num_test       : int) -> None:
        payload = {
            "run_directory"      : str(self.run_directory),
            "profile_length"     : profile_length,
            "latent_dim"         : self.ae_config.latent_dim,
            "encoder"            : self.ae_config.encoder.backbone.value,
            "decoder"            : self.ae_config.decoder.backbone.value,
            "num_train_profiles" : num_train,
            "num_val_profiles"   : num_val,
            "num_test_profiles"  : num_test,
        }
        with open(self.metadata_directory / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        self.logger.subsection("-> Run summary saved: meta/run_summary.json")

    def run(self) -> dict:
        self.logger.section("[Autoencoder Pipeline Execution]")
        _, _, _, datasets = self.dataset_pipeline.run()

        profile_train = ProfileDataset(datasets["train"], self.ae_config.data, "train", self.logger)
        profile_val   = ProfileDataset(datasets["val"],   self.ae_config.data, "val",   self.logger)
        profile_test  = ProfileDataset(datasets["test"],  self.ae_config.data, "test",  self.logger)

        profile_length = profile_train.profile_length
        self.ae_config.profile_length = profile_length

        num_workers = self.dataset_config.num_workers
        loader_builder = LoaderBuilder(
            batch_size         = self.dataset_config.batch_size,
            num_workers        = num_workers,
            logger             = self.logger,
            shuffle_train      = self.dataset_config.shuffle_train,
            pin_memory         = self.dataset_config.pin_memory,
            persistent_workers = num_workers > 0,
            prefetch_factor    = 4 if num_workers > 0 else None,
        )
        train_loader, val_loader, test_loader = loader_builder.build(
            train_dataset = profile_train,
            val_dataset   = profile_val,
            test_dataset  = profile_test,
        )

        self._save_configs()
        self._save_run_summary(profile_length, len(profile_train), len(profile_val), len(profile_test))

        model = Autoencoder(self.ae_config)
        self.logger.section("[Model Built]")
        self.logger.kv_table(
            {
                "Profile length" : profile_length,
                "Latent dim"     : self.ae_config.latent_dim,
                "Parameters"     : f"{sum(p.numel() for p in model.parameters()):,}",
            },
            title="Autoencoder Model",
        )

        trainer = Trainer(
            model         = model,
            ae_config     = self.ae_config,
            train_loader  = train_loader,
            val_loader    = val_loader,
            run_directory = self.run_directory,
            logger        = self.logger,
            plotter       = self.plotter,
            writer        = self.writer,
        )

        try:
            history = trainer.fit()
        finally:
            self.writer.flush()
            self.writer.close()

        inference_results: dict[str, dict] = {}
        for split_name, loader in (("val", val_loader), ("test", test_loader)):
            inference = Inference(
                model         = trainer.model,
                ae_config     = self.ae_config,
                loader        = loader,
                run_directory = self.run_directory,
                logger        = self.logger,
                plotter       = self.plotter,
                split_name    = split_name,
            )
            inference_results[split_name] = inference.run()

        checkpoint_paths = {
            "encoder_best"     : str(self.checkpoint_directory / "encoder_best.pt"),
            "decoder_best"     : str(self.checkpoint_directory / "decoder_best.pt"),
            "autoencoder_best" : str(self.checkpoint_directory / "autoencoder_best.pt"),
            "encoder_final"    : str(self.checkpoint_directory / "encoder_final.pt"),
            "decoder_final"    : str(self.checkpoint_directory / "decoder_final.pt"),
            "autoencoder_final": str(self.checkpoint_directory / "autoencoder_final.pt"),
        }

        reporter = Reporter(self.ae_config, self.run_directory)
        report_path = reporter.write(
            history           = history,
            best_epoch        = trainer.best_epoch,
            best_val_total    = trainer.best_val_total,
            inference_results = inference_results,
            checkpoint_paths  = checkpoint_paths,
        )
        self.logger.section("[Report Written]")
        self.logger.subsection(f"-> {report_path}")

        self.logger.close()
        return {
            "history"           : history,
            "inference_results" : inference_results,
            "checkpoint_paths"  : checkpoint_paths,
            "report_path"       : str(report_path),
        }
