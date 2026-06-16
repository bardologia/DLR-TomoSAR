from __future__ import annotations

import gc
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from configuration.benchmark            import BenchmarkConfig
from models                             import BACKBONE_CONFIG_REGISTRY, BACKBONE_IMAGE_SIZE_MODELS, get_backbone
from pipelines.backbone.dataset.pipeline import DatasetPipeline
from pipelines.backbone.training.trainer import Trainer
from pipelines.shared.config_factory     import ConfigFactory
from tools.monitoring.logger             import Logger
from tools.runtime.reproducibility       import Reproducibility


class MaxBatchProbe:
    def __init__(self, config: BenchmarkConfig, model_name: str, overrides: dict) -> None:
        self.config     = config
        self.model_name = model_name
        self.overrides  = overrides

        self.budget_gb     = config.max_batch.vram_budget_gb
        self.ceiling       = config.max_batch.max_batch
        self.measure_steps = config.max_batch.measure_steps
        self.seed          = config.max_batch.seed

        self.device     = torch.device("cuda")
        self.context_gb = 0.0
        self.work_dir   = Path(config.paths.log_base_dir) / "max_batch_probe" / model_name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger     = Logger(log_dir=str(self.work_dir / "logs"), name="max_batch", level="INFO")

    def _measure_context(self) -> float:
        warm = torch.zeros(1, device=self.device)
        del warm
        torch.cuda.empty_cache()

        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)

        return (total_bytes - free_bytes) / (1024.0 ** 3)

    def _build_context(self):
        factory        = ConfigFactory(self.config)
        dataset_config = factory.training_dataset_config()
        trainer_config = factory.training_trainer_config(logdir=self.work_dir)

        gaussian_cfg               = trainer_config.gaussian
        dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        dataset_pipeline = DatasetPipeline(config=dataset_config, training_run_directory=self.work_dir, logger=self.logger, seed=self.seed)

        profile_length        = dataset_pipeline.layout.profile_length
        dataset_config.x_axis = np.linspace(gaussian_cfg.x_min, gaussian_cfg.x_max, profile_length, dtype=np.float32)

        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()

        return trainer_config, dataset_config, datasets["train"], gaussian_cfg

    def _build_model(self, dataset_config, dataset, gaussian_cfg):
        model_config = BACKBONE_CONFIG_REGISTRY[self.model_name]()

        for attribute, value in self.overrides.items():
            setattr(model_config, attribute, value)

        in_channels  = dataset.input_channels
        out_channels = gaussian_cfg.params_per_gaussian * gaussian_cfg.n_default_gaussians

        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        if self.model_name in BACKBONE_IMAGE_SIZE_MODELS:
            overrides["image_size"] = dataset_config.patch.size[0]

        return get_backbone(self.model_name, config=model_config, **overrides)

    def _build_trainer(self, trainer_config, dataset_config, model, model_cfg, dataset):
        trainer = Trainer(
            model      = model,
            model_cfg  = model_cfg,
            x_axis     = dataset_config.x_axis,
            config     = trainer_config,
            run_dir    = self.work_dir,
            logger     = self.logger,
            norm_stats = dataset.normalizer,
            emit_docs  = False,
        )

        trainer.criterion.set_curriculum(trainer_config.curriculum.complete)
        trainer.model.train()

        return trainer

    def _candidates(self) -> list[int]:
        sizes = []
        size  = 1

        while size <= self.ceiling:
            sizes.append(size)
            size *= 2

        return sizes

    def _trial(self, trainer, dataset, batch_size: int) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
        iterator = iter(loader)

        for _ in range(self.measure_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch    = next(iterator)

            trainer.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=trainer.use_amp):
                loss = trainer._compute_loss(batch)["total_loss"]

            loss.backward()
            trainer.optimizer.step()

        peak_reserved = torch.cuda.max_memory_reserved(self.device)

        del loss, batch, iterator, loader

        return self.context_gb + peak_reserved / (1024.0 ** 3)

    def run(self) -> dict:
        result = {
            "model"      : self.model_name,
            "status"     : None,
            "batch_size" : None,
            "peak_gb"    : None,
            "budget_gb"  : self.budget_gb,
            "ceiling"    : self.ceiling,
            "trials"     : [],
            "error"      : None,
        }

        try:
            Reproducibility.seed_everything(self.seed)

            self.context_gb = self._measure_context()
            self.logger.subsection(f"CUDA context: {self.context_gb:.2f} GB")

            trainer_config, dataset_config, dataset, gaussian_cfg = self._build_context()

            model, model_cfg = self._build_model(dataset_config, dataset, gaussian_cfg)
            trainer          = self._build_trainer(trainer_config, dataset_config, model, model_cfg, dataset)

            best = None

            for batch_size in self._candidates():
                try:
                    peak_gb = self._trial(trainer, dataset, batch_size)
                except torch.cuda.OutOfMemoryError:
                    trainer.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    result["trials"].append({"batch_size": batch_size, "peak_gb": None, "status": "OOM"})
                    self.logger.subsection(f"batch {batch_size:>5} -> OOM")
                    break

                fits = peak_gb <= self.budget_gb
                result["trials"].append({"batch_size": batch_size, "peak_gb": peak_gb, "status": "FIT" if fits else "OVER"})
                self.logger.subsection(f"batch {batch_size:>5} -> {peak_gb:.2f} GB {'FIT' if fits else 'OVER'}")

                if not fits:
                    break

                best = (batch_size, peak_gb)

            if best is None:
                result["status"] = "FAIL"
                result["error"]  = f"batch size 1 exceeds the {self.budget_gb:g} GB budget"
            else:
                result["batch_size"], result["peak_gb"] = best
                result["status"]                        = "PASS"

            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            result["status"] = "FAIL"
            result["error"]  = traceback.format_exc()

        self.logger.close()

        return result
