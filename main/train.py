from __future__ import annotations

import os
os.environ["MKL_NUM_THREADS"]     = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"]     = "4" 
 
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuration.dataset_config       import (
    DatasetCreationConfiguration,
    InputConfig,
    PatchConfiguration,
    Representation,
    SplitRegions,
    InputNormalizationMode,
    OutputNormalizationMode,
)
from tools.crop_region import CropRegion
from configuration.training_config      import (
    EarlyStoppingConfig,
    EMAConfig,
    GaussianConfig,
    IOConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfigInner,
    WarmupConfig,
)
from pipelines.training_pipeline.pipeline import TrainingPipeline


def main() -> None:
    model_name   = "unet"
    dataset_path = Path("/ste/rnd/User/vice_vi/Dataset/base_dataset")

    with open(dataset_path / "data" / "dataset.json", "r", encoding="utf-8") as f:
        layout = json.load(f)
    global_crop = CropRegion(*layout["global_crop"])

    split_regions = SplitRegions(
        train = CropRegion(1000,  9120,  global_crop.range_start, global_crop.range_end),
        val   = CropRegion(9120,  12400, global_crop.range_start, global_crop.range_end),
        test  = CropRegion(12400, 16000, global_crop.range_start, global_crop.range_end),
    )

    dataset_config = DatasetCreationConfiguration(
        preprocessing_run_directory = dataset_path,
        parameters_path             = dataset_path / "params" / "params_Ng2_adapt" / "parameters_Ng2_adapt.npy",
        split_regions               = split_regions,
        patch                       = PatchConfiguration(size=(64, 64), stride=32, use_reflective_padding=True),

        input_config = InputConfig(
            use_master         = False, master_representation         = Representation.MAG_ONLY,
            use_slaves         = True,  slaves_representation         = Representation.MAG_ONLY,
            use_interferograms = True,  interferograms_representation = Representation.ANGLE_ONLY,
        ),

        batch_size                  = 400,
        num_workers                 = 8,

        shuffle_train               = True,
        pin_memory                  = False,

        input_normalization_mode    = InputNormalizationMode.GROUPED,
        output_normalization_mode   = OutputNormalizationMode.GROUPED,
    )

    trainer_config = TrainerConfig(
        gaussian       = GaussianConfig.from_dataset(dataset_path),
        early_stopping = EarlyStoppingConfig(patience=30, min_delta=0.0001, restore_best=True),
        warmup         = WarmupConfig(warmup_steps=50, warmup_start_factor=0.1, warmup_enabled=True),
        scheduler      = SchedulerConfig(epochs=200, eta_min=1e-6),
        ema            = EMAConfig(use_ema=False, ema_decay=0.999),
        optimizer      = OptimizerConfig(lr=2e-3, betas=(0.9, 0.999), eps=1e-8),
        io             = IOConfig(logdir="/ste/rnd/User/vice_vi/DLR-TomoSAR/logs"),

        training = TrainingConfigInner(
            device                      = "gpu",
            epochs                      = 200,
            validation_frequency        = 5,
            use_amp                     = False,
            gradient_accumulation_steps = 1,
            max_grad_norm               = None,
            verbose                     = True,
            overfit_enabled             = False,
            deep_validation             = False,
            eval_train_split            = False,
        ),

        loss = LossConfig(
            use_ssim_curve    = True,
            weight_ssim_curve = 1.0,
            ssim_window_size  = 11,
            ssim_sigma        = 1.5,
            ssim_data_range   = 1.0,
            ssim_k1           = 0.01,
            ssim_k2           = 0.03,
            ssim_axis         = "elevation",

            use_mse_curve    = True,
            weight_mse_curve = 1.0,
        ),
    )

    pipeline = TrainingPipeline(
        trainer_config = trainer_config,
        dataset_config = dataset_config,
        model_name     = model_name,
        seed           = 0,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
