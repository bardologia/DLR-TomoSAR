from __future__ import annotations

import os
GPU_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"]          = str(GPU_ID)
os.environ["MKL_NUM_THREADS"]               = "4"
os.environ["NUMEXPR_NUM_THREADS"]           = "4"
os.environ["OMP_NUM_THREADS"]               = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]       = "expandable_segments:True"
 
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuration.dataset_config       import (
    DatasetConfiguration,
    InputConfig,
    SplitRegions,
)

from tools.crop_region      import CropRegion
from tools.loss_scale_probe import LossScaleProbeConfig

from configuration.training_config      import (
    EarlyStoppingConfig,
    EMAConfig,
    GaussianConfig,
    GradientClipperConfig,
    IOConfig,
    LossConfig,
    LossCurriculumConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfigInner,
)

from pipelines.training_pipeline.pipeline import TrainingPipeline


run_name                     = "resunet_w-pL11_c-mse2-pL11"
model_name    = "resunet"
seed          = 0

dataset_path  = Path("/ste/rnd/User/vice_vi/Dataset/clean_dataset")
params_path   = Path("/ste/rnd/User/vice_vi/Dataset/clean_dataset/params/params_sig_k5/parameters_sig_k5.npy")
logdir        = "/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/curriculum/"

batch_size    = 256
num_workers   = 8

n_gaussians              = 5
epochs                   = 150
validation_frequency     = 1
early_stopping_patience  = 50
early_stopping_min_delta = 0.0001

probe_enabled    = False
probe_n_batches  = 1000
probe_reference  = "param_l1"
probe_exit_after = True

curriculum_enabled           = True
curriculum_swap_epoch           = 50
curriculum_reset_early_stopping = True
curriculum_reset_lr             = True
curriculum_reset_warmup         = True
curriculum_reset_optimizer      = True

warmup_use_mse_curve              = False
warmup_weight_mse_curve           = 0.0
warmup_use_l1_curve               = False
warmup_weight_l1_curve            = 0.0
warmup_use_huber_curve            = False
warmup_weight_huber_curve         = 0.0
warmup_use_charbonnier_curve      = False
warmup_weight_charbonnier_curve   = 0.0
warmup_use_cosine_curve           = False
warmup_weight_cosine_curve        = 0.0
warmup_use_spectral_coherence     = False
warmup_weight_spectral_coh        = 0.0
warmup_use_ssim_curve             = False
warmup_weight_ssim_curve          = 0.0
warmup_use_param_l1          = True
warmup_weight_param_l1       = 1.0
warmup_use_param_huber            = False
warmup_weight_param_huber         = 0.0
warmup_use_smoothness_tv          = False
warmup_weight_smoothness_tv       = 0.0

complete_use_mse_curve       = True
complete_weight_mse_curve         = 0.0
complete_use_l1_curve             = False
complete_weight_l1_curve          = 0.0
complete_use_huber_curve          = False
complete_weight_huber_curve       = 0.0
complete_use_charbonnier_curve    = False
complete_weight_charbonnier_curve = 0.0
complete_use_cosine_curve         = False
complete_weight_cosine_curve      = 0.0
complete_use_spectral_coherence   = False
complete_weight_spectral_coh      = 0.0
complete_use_ssim_curve           = False
complete_weight_ssim_curve        = 0.0
complete_use_param_l1        = True
complete_weight_param_l1     = 1.0
complete_use_param_huber          = False
complete_weight_param_huber       = 0.0
complete_use_smoothness_tv        = False
complete_weight_smoothness_tv     = 0.0


def main() -> None:
    with open(dataset_path / "data" / "dataset.json", "r", encoding="utf-8") as f:
        layout = json.load(f)

    global_crop = CropRegion(*layout["global_crop"])

    split_regions = SplitRegions(
        train = CropRegion(1000,  9120,  global_crop.range_start, global_crop.range_end),
        val   = CropRegion(9120,  12400, global_crop.range_start, global_crop.range_end),
        test  = CropRegion(12400, 16000, global_crop.range_start, global_crop.range_end),
    )

    dataset_config = DatasetConfiguration(
        preprocessing_run_directory = dataset_path,
        parameters_path             = params_path,
        split_regions               = split_regions,
        input_config                = InputConfig(use_secondaries=True),
        batch_size                  = batch_size,
        num_workers                 = num_workers,
    )

    warmup_loss_config = LossConfig(
        use_mse_curve          = warmup_use_mse_curve,          weight_mse_curve         = warmup_weight_mse_curve,
        use_l1_curve           = warmup_use_l1_curve,           weight_l1_curve          = warmup_weight_l1_curve,
        use_huber_curve        = warmup_use_huber_curve,        weight_huber_curve       = warmup_weight_huber_curve,
        use_charbonnier_curve  = warmup_use_charbonnier_curve,  weight_charbonnier_curve = warmup_weight_charbonnier_curve,
        use_cosine_curve       = warmup_use_cosine_curve,       weight_cosine_curve      = warmup_weight_cosine_curve,
        use_spectral_coherence = warmup_use_spectral_coherence, weight_spectral_coh      = warmup_weight_spectral_coh,
        use_ssim_curve         = warmup_use_ssim_curve,         weight_ssim_curve        = warmup_weight_ssim_curve,
        use_param_l1           = warmup_use_param_l1,           weight_param_l1          = warmup_weight_param_l1,
        use_param_huber        = warmup_use_param_huber,        weight_param_huber       = warmup_weight_param_huber,
        use_smoothness_tv      = warmup_use_smoothness_tv,      weight_smoothness_tv     = warmup_weight_smoothness_tv,
    )

    complete_loss_config = LossConfig(
        use_mse_curve          = complete_use_mse_curve,          weight_mse_curve         = complete_weight_mse_curve,
        use_l1_curve           = complete_use_l1_curve,           weight_l1_curve          = complete_weight_l1_curve,
        use_huber_curve        = complete_use_huber_curve,        weight_huber_curve       = complete_weight_huber_curve,
        use_charbonnier_curve  = complete_use_charbonnier_curve,  weight_charbonnier_curve = complete_weight_charbonnier_curve,
        use_cosine_curve       = complete_use_cosine_curve,       weight_cosine_curve      = complete_weight_cosine_curve,
        use_spectral_coherence = complete_use_spectral_coherence, weight_spectral_coh      = complete_weight_spectral_coh,
        use_ssim_curve         = complete_use_ssim_curve,         weight_ssim_curve        = complete_weight_ssim_curve,
        use_param_l1           = complete_use_param_l1,           weight_param_l1          = complete_weight_param_l1,
        use_param_huber        = complete_use_param_huber,        weight_param_huber       = complete_weight_param_huber,
        use_smoothness_tv      = complete_use_smoothness_tv,      weight_smoothness_tv     = complete_weight_smoothness_tv,
    )

    trainer_config = TrainerConfig(
        gaussian         = GaussianConfig.from_dataset(dataset_path, n_gaussians=n_gaussians),
        
        early_stopping   = EarlyStoppingConfig(patience=early_stopping_patience, min_delta=early_stopping_min_delta),
        
        scheduler        = SchedulerConfig(epochs=epochs),
        
        ema              = EMAConfig(use_ema=False),
        
        io               = IOConfig(logdir=logdir),
        
        gradient_clipper = GradientClipperConfig(clip_mode="fixed"),
        
        training         = TrainingConfigInner(device="gpu", epochs=epochs, validation_frequency=validation_frequency),
        
        curriculum       = LossCurriculumConfig(
            enabled              = curriculum_enabled,
            swap_epoch           = curriculum_swap_epoch,
            reset_early_stopping = curriculum_reset_early_stopping,
            reset_lr             = curriculum_reset_lr,
            reset_warmup         = curriculum_reset_warmup,
            reset_optimizer      = curriculum_reset_optimizer,
            warmup               = warmup_loss_config,
            complete             = complete_loss_config,
        ),
    )

    pipeline = TrainingPipeline(
        trainer_config = trainer_config,
        dataset_config = dataset_config,
        model_name     = model_name,
        seed           = seed,
        run_name       = run_name,
    )

    pipeline.run(probe_config=LossScaleProbeConfig(
        enabled        = probe_enabled,
        n_batches      = probe_n_batches,
        reference      = probe_reference,
        exit_after     = probe_exit_after,
        enabled_losses = {},
    ))


if __name__ == "__main__":
    main()