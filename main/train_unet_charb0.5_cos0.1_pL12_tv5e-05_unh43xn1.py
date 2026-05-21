from __future__ import annotations

import os
GPU_ID = 3
os.environ["CUDA_VISIBLE_DEVICES"]    = str(GPU_ID)
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
    DatasetConfiguration,
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
    OverfitConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfigInner,
    WarmupConfig,
)

from models import UNetConfig
from pipelines.training_pipeline.pipeline import TrainingPipeline


model_name   = "unet"
n_gaussians  = 5
seed         = 0

dataset_path  = Path("/ste/rnd/User/vice_vi/Dataset/clean_dataset")
logdir        = "/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/unet_trials/unet_charb0.5_cos0.1_pL12_tv5e-05"
params_path   = Path("/ste/rnd/User/vice_vi/Dataset/clean_dataset/params/params_sig_k5/parameters_sig_k5.npy")

train_az  = (1000,  9120)
val_az    = (9120,  12400)
test_az   = (12400, 16000)

patch_size   = (64, 64)
patch_stride = 32

batch_size  = 256
num_workers = 8

betas = (0.9, 0.999)
eps   = 1e-8

scheduler_epochs = 200
eta_min          = 1e-6

warmup_enabled      = True
warmup_steps        = 50
warmup_start_factor = 0.1

use_ema    = False
ema_decay  = 0.999

es_patience  = 30
es_min_delta = 0.0001
es_restore   = True

epochs               = 200
validation_frequency = 1
use_amp              = False
grad_accum_steps     = 1

use_charbonnier_curve        = True
weight_charbonnier_curve     = 0.5

use_ssim_curve               = False
weight_ssim_curve            = 0.0
ssim_axis                = "elevation" 

use_cosine_curve             = True
weight_cosine_curve          = 0.1

use_param_l1                 = True
weight_param_l1              = 2.0
param_match              = "sort_gt_by_mu"

use_smoothness_tv            = True
weight_smoothness_tv         = 5e-05

overfit_enabled        = False
overfit_max_steps      = 5
overfit_stop_threshold = 1e-6
overfit_batch_size     = 1

primary_representation        = Representation.MAG_ONLY
secondaries_representation    = Representation.MAG_ONLY
interferograms_representation = Representation.ANGLE_ONLY

input_normalization_mode  = InputNormalizationMode.GROUPED
output_normalization_mode = OutputNormalizationMode.GROUPED

encoder_lr     = 3e-4
bottleneck_lr  = 3e-4
decoder_lr     = 3e-4
output_head_lr = 1e-3

encoder_wd     = 5e-3
bottleneck_wd  = 5e-3
decoder_wd     = 5e-3
output_head_wd = 5e-3


def main() -> None:
    model_config = UNetConfig(
        encoder_lr     = encoder_lr,
        bottleneck_lr  = bottleneck_lr,
        decoder_lr     = decoder_lr,
        output_head_lr = output_head_lr,
        encoder_wd     = encoder_wd,
        bottleneck_wd  = bottleneck_wd,
        decoder_wd     = decoder_wd,
        output_head_wd = output_head_wd,
    )

    with open(dataset_path / "data" / "dataset.json", "r", encoding="utf-8") as f:
        layout = json.load(f)
    
    global_crop = CropRegion(*layout["global_crop"])

    split_regions = SplitRegions(
        train = CropRegion(train_az[0], train_az[1], global_crop.range_start, global_crop.range_end),
        val   = CropRegion(val_az[0],   val_az[1],   global_crop.range_start, global_crop.range_end),
        test  = CropRegion(test_az[0],  test_az[1],  global_crop.range_start, global_crop.range_end),
    )
   
    dataset_config = DatasetConfiguration(
        preprocessing_run_directory = dataset_path,
        parameters_path             = params_path,
        split_regions               = split_regions,
        patch                       = PatchConfiguration(size=patch_size, stride=patch_stride, use_reflective_padding=True),

        input_config = InputConfig(
            use_primary        = True,  primary_representation        = primary_representation,
            use_secondaries    = True,  secondaries_representation    = secondaries_representation,
            use_interferograms = True,  interferograms_representation = interferograms_representation,
        ),

        batch_size                  = batch_size,
        num_workers                 = num_workers,

        shuffle_train               = True,
        pin_memory                  = True,

        input_normalization_mode    = input_normalization_mode,
        output_normalization_mode   = output_normalization_mode,
    )

    trainer_config = TrainerConfig(
        gaussian       = GaussianConfig.from_dataset(dataset_path, n_gaussians=n_gaussians),
        early_stopping = EarlyStoppingConfig(patience=es_patience, min_delta=es_min_delta, restore_best=es_restore),
        warmup         = WarmupConfig(warmup_steps=warmup_steps, warmup_start_factor=warmup_start_factor, warmup_enabled=warmup_enabled),
        scheduler      = SchedulerConfig(epochs=scheduler_epochs, eta_min=eta_min),
        ema            = EMAConfig(use_ema=use_ema, ema_decay=ema_decay),
        optimizer      = OptimizerConfig(betas=betas, eps=eps),
        io             = IOConfig(logdir=logdir),

        training = TrainingConfigInner(
            device                      = "gpu",
            epochs                      = epochs,
            validation_frequency        = validation_frequency,
            use_amp                     = use_amp,
            gradient_accumulation_steps = grad_accum_steps,
            max_grad_norm               = None,
            verbose                     = True,
        ),

        overfit = OverfitConfig(
            enabled        = overfit_enabled,
            max_steps      = overfit_max_steps,
            stop_threshold = overfit_stop_threshold,
            batch_size     = overfit_batch_size,
        ),

        loss = LossConfig(
            use_ssim_curve           = use_ssim_curve,
            weight_ssim_curve        = weight_ssim_curve,
            ssim_axis                = ssim_axis,

            use_charbonnier_curve    = use_charbonnier_curve,
            weight_charbonnier_curve = weight_charbonnier_curve,

            use_cosine_curve         = use_cosine_curve,
            weight_cosine_curve      = weight_cosine_curve,

            use_param_l1             = use_param_l1,
            weight_param_l1          = weight_param_l1,
            param_match              = param_match,

            use_smoothness_tv        = use_smoothness_tv,
            weight_smoothness_tv     = weight_smoothness_tv,
        ),
    )

    pipeline = TrainingPipeline(
        trainer_config = trainer_config,
        dataset_config = dataset_config,
        model_name     = model_name,
        model_config   = model_config,
        seed           = seed,
    )

    pipeline.run()


if __name__ == "__main__":
    main()