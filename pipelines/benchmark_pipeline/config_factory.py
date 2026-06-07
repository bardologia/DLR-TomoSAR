from __future__ import annotations

import json
from pathlib import Path

from configuration.benchmark_config import BenchmarkConfig
from configuration.inference_config import InferenceConfig
from tools.regions              import CropRegion

from configuration.dataset_config import (
    AugmentationConfig,
    DatasetConfiguration,
    InputConfig,
    PatchConfiguration,
    Representation,
    SplitRegions,
)

from configuration.training_config import (
    LossCurriculumConfig,
    EarlyStoppingConfig,
    EMAConfig,
    GaussianConfig,
    GeometryConfig,
    GradientClipperConfig,
    IOConfig,
    LossConfig,
    OptimizerConfig,
    OverfitConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfigInner,
    WarmupConfig,
)


class ConfigFactory:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def global_crop(self) -> CropRegion:
        layout_path = Path(self.config.paths.dataset_path) / "data" / "dataset.json"

        with open(layout_path, "r", encoding="utf-8") as f:
            layout = json.load(f)

        return CropRegion(*layout["global_crop"])

    def benchmark_input_config(self) -> InputConfig:
        return InputConfig(
            use_primary        = True,  primary_representation        = Representation.MAG_ONLY,
            use_secondaries    = True,  secondaries_representation    = Representation.MAG_ONLY,
            use_interferograms = True,  interferograms_representation = Representation.ANGLE_ONLY,
        )

    def training_dataset_config(self) -> DatasetConfiguration:
        crop     = self.global_crop()
        training = self.config.training

        split_regions = SplitRegions(
            train = CropRegion(training.train_azimuth[0], training.train_azimuth[1], crop.range_start, crop.range_end),
            val   = CropRegion(training.val_azimuth[0],   training.val_azimuth[1],   crop.range_start, crop.range_end),
            test  = CropRegion(training.test_azimuth[0],  training.test_azimuth[1],  crop.range_start, crop.range_end),
        )

        return DatasetConfiguration(
            preprocessing_run_directory = self.config.paths.dataset_path,
            parameters_path             = self.config.paths.parameters_path,
            split_regions               = split_regions,
            patch         = PatchConfiguration(size=training.patch_size, stride=training.patch_stride, use_reflective_padding=True),
            input_config  = self.benchmark_input_config(),
            batch_size    = training.batch_size,
            num_workers   = training.num_workers,
            shuffle_train = True,
            pin_memory    = True,
        )

    def overfit_dataset_config(self) -> DatasetConfiguration:
        crop    = self.global_crop()
        overfit = self.config.overfit

        azimuth_end  = overfit.azimuth_start + overfit.azimuth_lines
        range_end    = crop.range_start + overfit.range_lines
        overfit_crop = CropRegion(overfit.azimuth_start, azimuth_end, crop.range_start, range_end)

        return DatasetConfiguration(
            preprocessing_run_directory = self.config.paths.dataset_path,
            parameters_path             = self.config.paths.parameters_path,

            split_regions = SplitRegions(
                train = overfit_crop,
                val   = overfit_crop,
                test  = overfit_crop,
            ),

            input_config = self.benchmark_input_config(),

            augmentation = AugmentationConfig(
                p_flip_h    = 0.0,
                p_flip_v    = 0.0,
                p_rot90     = 0.0,
                p_amp_scale = 0.0,
                p_noise     = 0.0,
            ),
        )

    def training_trainer_config(self, logdir: Path) -> TrainerConfig:
        training         = self.config.training
        scheduler_epochs = training.scheduler_epochs if training.scheduler_epochs is not None else training.epochs

        return TrainerConfig(
            gaussian         = GaussianConfig.from_dataset(self.config.paths.dataset_path, n_gaussians=self.config.n_gaussians),
            geometry         = GeometryConfig().resolved(self.config.paths.dataset_path),
            early_stopping   = EarlyStoppingConfig(patience=training.early_stop_patience, min_delta=training.early_stop_min_delta, restore_best=True),
            warmup           = WarmupConfig(warmup_steps=training.warmup_steps, warmup_start_factor=0.1, warmup_enabled=True, warmup_mode="linear"),
            scheduler        = SchedulerConfig(type="cosine_annealing", epochs=scheduler_epochs, eta_min=training.eta_min),
            ema              = EMAConfig(use_ema=False, ema_decay=0.999),
            optimizer        = OptimizerConfig(betas=(0.9, 0.999), eps=1e-8),
            gradient_clipper = GradientClipperConfig(clip_mode="fixed", max_grad_norm=1.0),

            io = IOConfig(logdir=str(logdir)),

            training = TrainingConfigInner(
                device                      = "gpu",
                epochs                      = training.epochs,
                validation_frequency        = training.validation_frequency,
                use_amp                     = False,
                gradient_accumulation_steps = 1,
                max_grad_norm               = None,
                verbose                     = True,
                log_all_losses              = training.log_all_losses,
            ),

            overfit = OverfitConfig(enabled=False),

            curriculum = LossCurriculumConfig(
                enabled  = False,
                warmup   = LossConfig(use_param_l1=True, weight_param_l1=1.0, param_weights=(1.0, 1.0, 1.0)),
                complete = LossConfig(use_param_l1=True, weight_param_l1=1.0),
            ),
        )

    def overfit_trainer_config(self, logdir: Path) -> TrainerConfig:
        overfit = self.config.overfit

        return TrainerConfig(
            gaussian         = GaussianConfig.from_dataset(self.config.paths.dataset_path, n_gaussians=self.config.n_gaussians),
            geometry         = GeometryConfig().resolved(self.config.paths.dataset_path),
            early_stopping   = EarlyStoppingConfig(patience=9999, min_delta=0.0, restore_best=False),
            warmup           = WarmupConfig(warmup_enabled=False),
            scheduler        = SchedulerConfig(type="constant"),
            ema              = EMAConfig(use_ema=False, ema_decay=0.999),
            optimizer        = OptimizerConfig(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
            gradient_clipper = GradientClipperConfig(clip_mode="fixed", max_grad_norm=1.0),

            io = IOConfig(logdir=str(logdir)),

            training = TrainingConfigInner(
                device               = "gpu",
                epochs               = 10000,
                validation_frequency = 9999,
            ),

            overfit = OverfitConfig(
                enabled        = True,
                max_steps      = overfit.max_steps,
                stop_threshold = overfit.stop_threshold,
                batch_size     = overfit.batch_size,
            ),

            curriculum = LossCurriculumConfig(
                enabled  = False,
                warmup   = LossConfig(use_param_l1=True, weight_param_l1=1.0),
                complete = LossConfig(use_param_l1=True, weight_param_l1=1.0),
            ),
        )

    def prepare_overfit_model_config(self, model_config):
        for attribute in ("dropout", "attention_dropout", "stochastic_depth_rate"):
            if hasattr(model_config, attribute):
                setattr(model_config, attribute, 0.0)

        for attribute in vars(model_config):
            if attribute.endswith("_wd"):
                setattr(model_config, attribute, 0.0)

            if attribute.endswith("_lr"):
                setattr(model_config, attribute, getattr(model_config, attribute) * 10.0)

        return model_config

    def inference_config(self, run_directory: Path) -> InferenceConfig:
        inference = self.config.inference

        return InferenceConfig(
            run_directory      = run_directory,
            output_subdir      = None,
            device             = "cuda",
            use_ema            = inference.use_ema,
            checkpoint_name    = inference.checkpoint_name,
            split              = inference.split,
            batch_size         = inference.batch_size,
            num_workers        = inference.num_workers,
            stitch_window      = inference.stitch_window,
            save_cubes         = inference.save_cubes,
            n_best_profiles    = inference.n_best_profiles,
            n_worst_profiles   = inference.n_worst_profiles,
            n_random_profiles  = inference.n_random_profiles,
            n_range_slices     = inference.n_range_slices,
            n_azimuth_slices   = inference.n_azimuth_slices,
            n_elevation_slices = inference.n_elevation_slices,
            gif_axes           = list(inference.gif_axes),
            gif_fps            = inference.gif_fps,
            gif_max_frames     = inference.gif_max_frames,
            cpu_workers        = inference.cpu_workers,
        )
