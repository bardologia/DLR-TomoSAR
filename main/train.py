from __future__ import annotations

import os
GPU_ID = 0  
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


def main() -> None:

    # ── Experiment ────────────────────────────────────────────────────────────
    model_name   = "unet"
    n_gaussians  = 2
    seed         = 0

    # ── Paths ─────────────────────────────────────────────────────────────────
    dataset_path  = Path("/ste/rnd/User/vice_vi/Dataset/base_dataset")
    logdir        = "/ste/rnd/User/vice_vi/DLR-TomoSAR/logs"
    params_subdir = f"params_Ng{n_gaussians}_adapt"

    # ── Dataset splits ────────────────────────────────────────────────────────
    train_az  = (1000,  9120)
    val_az    = (9120,  12400)
    test_az   = (12400, 16000)

    # ── Patch ─────────────────────────────────────────────────────────────────
    patch_size   = (64, 64)
    patch_stride = 32

    # ── Dataloader ────────────────────────────────────────────────────────────
    batch_size  = 256
    num_workers = 8

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr    = 1e-3
    betas = (0.9, 0.999)
    eps   = 1e-8

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler_epochs = 150
    eta_min          = 1e-6

    # ── Warmup ────────────────────────────────────────────────────────────────
    warmup_enabled      = True
    warmup_steps        = 50
    warmup_start_factor = 0.1

    # ── EMA ───────────────────────────────────────────────────────────────────
    use_ema    = False
    ema_decay  = 0.999

    # ── Early stopping ────────────────────────────────────────────────────────
    es_patience  = 15
    es_min_delta = 0.0001
    es_restore   = True

    # ── Training ──────────────────────────────────────────────────────────────
    epochs               = 15
    validation_frequency = 5
    use_amp              = False
    grad_accum_steps     = 1

    # ── Loss ──────────────────────────────────────────────────────────────────
    use_mse_curve     = True
    weight_mse_curve  = 1.0

    use_ssim_curve    = False
    weight_ssim_curve = 1.0
    ssim_window_size  = 11
    ssim_sigma        = 1.5
    ssim_data_range   = 1.0
    ssim_k1           = 0.01
    ssim_k2           = 0.03
    ssim_axis         = "elevation"

    # ── Overfit sanity check ──────────────────────────────────────────────────
    overfit_enabled        = False
    overfit_max_steps      = 50
    overfit_stop_threshold = 1e-6
    overfit_batch_size     = 1

    # ── Model architecture ────────────────────────────────────────────────────
    features           = [64, 128, 256, 512]
    bottleneck_factor  = 2
    dropout            = 0.0
    activation         = "relu"
    normalization      = "batch"
    upsample_mode      = "convtranspose"
    init_mode          = "default"

    # ── Per-layer LR / weight decay ───────────────────────────────────────────
    lr_encoder      = 1e-4;  wd_encoder      = 1e-4
    lr_bottleneck   = 1e-4;  wd_bottleneck   = 1e-4
    lr_decoder      = 1e-4;  wd_decoder      = 1e-4
    lr_output_head  = 1e-3;  wd_output_head  = 1e-4

    # ─────────────────────────────────────────────────────────────────────────

    model_config = UNetConfig(
        features          = features,
        bottleneck_factor = bottleneck_factor,
        dropout           = dropout,
        activation        = activation,
        normalization     = normalization,
        upsample_mode     = upsample_mode,
        init_mode         = init_mode,
    )

    _pg_overrides = {
        "encoder"     : {"lr": lr_encoder,     "weight_decay": wd_encoder},
        "bottleneck"  : {"lr": lr_bottleneck,  "weight_decay": wd_bottleneck},
        "decoder"     : {"lr": lr_decoder,     "weight_decay": wd_decoder},
        "output_head" : {"lr": lr_output_head, "weight_decay": wd_output_head},
    }
    _orig_get_param_groups = model_config.get_param_groups
    def _patched_get_param_groups(model):
        groups = _orig_get_param_groups(model)
        for g in groups:
            overrides = _pg_overrides.get(g.get("name", ""), {})
            g.update(overrides)
        return groups
    model_config.get_param_groups = _patched_get_param_groups

    params_path   = dataset_path / "params" / params_subdir / f"parameters_Ng{n_gaussians}_adapt.npy"

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
            use_primary        = True, primary_representation         = Representation.MAG_ONLY,
            use_secondaries    = True,  secondaries_representation    = Representation.MAG_ONLY,
            use_interferograms = True,  interferograms_representation = Representation.ANGLE_ONLY,
        ),

        batch_size                  = batch_size,
        num_workers                 = num_workers,

        shuffle_train               = True,
        pin_memory                  = True,

        input_normalization_mode    = InputNormalizationMode.GROUPED,
        output_normalization_mode   = OutputNormalizationMode.GROUPED,
    )

    trainer_config = TrainerConfig(
        gaussian       = GaussianConfig.from_dataset(dataset_path, params_subdir=params_subdir),
        early_stopping = EarlyStoppingConfig(patience=es_patience, min_delta=es_min_delta, restore_best=es_restore),
        warmup         = WarmupConfig(warmup_steps=warmup_steps, warmup_start_factor=warmup_start_factor, warmup_enabled=warmup_enabled),
        scheduler      = SchedulerConfig(epochs=scheduler_epochs, eta_min=eta_min),
        ema            = EMAConfig(use_ema=use_ema, ema_decay=ema_decay),
        optimizer      = OptimizerConfig(lr=lr, betas=betas, eps=eps),
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
            use_mse_curve     = use_mse_curve,
            weight_mse_curve  = weight_mse_curve,

            use_ssim_curve    = use_ssim_curve,
            weight_ssim_curve = weight_ssim_curve,
            ssim_window_size  = ssim_window_size,
            ssim_sigma        = ssim_sigma,
            ssim_data_range   = ssim_data_range,
            ssim_k1           = ssim_k1,
            ssim_k2           = ssim_k2,
            ssim_axis         = ssim_axis,
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
