from __future__ import annotations

import os
GPU_ID = "0"
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
)

from tools.crop_region      import CropRegion
from tools.loss_scale_probe import LossScaleProbeConfig

from configuration.training_config      import (
    CurriculumConfig,
    LossCurriculumConfig,
    MatchingCurriculumConfig,
    EarlyStoppingConfig,
    EMAConfig,
    GaussianConfig,
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

from models import UNetConfig
from pipelines.training_pipeline.pipeline import TrainingPipeline


def main() -> None:
    dataset_path = Path("/ste/rnd/User/vice_vi/Dataset/clean_dataset")
    params_path  = Path("/ste/rnd/User/vice_vi/Dataset/clean_dataset/params/params_sig_k5/parameters_sig_k5.npy")

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
        patch                       = PatchConfiguration(size=(64, 64), stride=32, use_reflective_padding=True),

        input_config = InputConfig(
            use_primary        = True,  primary_representation        = Representation.MAG_ONLY,
            use_secondaries    = True,  secondaries_representation    = Representation.MAG_ONLY,
            use_interferograms = True,  interferograms_representation = Representation.ANGLE_ONLY,
        ),

        batch_size    = 256,
        num_workers   = 8,
        shuffle_train = True,
        pin_memory    = True,
    )

    model_config = UNetConfig(
        encoder_lr    = 3e-4,  encoder_wd    = 5e-3,
        bottleneck_lr = 3e-4,  bottleneck_wd = 5e-3,
        decoder_lr    = 3e-4,  decoder_wd    = 5e-3,
        output_head_lr      = 1e-3,        output_head_wd = 5e-3,
    )

    trainer_config = TrainerConfig(
        gaussian         = GaussianConfig.from_dataset(dataset_path, n_gaussians=5),
        early_stopping   = EarlyStoppingConfig(patience=30, min_delta=0.0001, restore_best=True),
        warmup           = WarmupConfig(warmup_steps=200, warmup_start_factor=0.1, warmup_enabled=True, warmup_mode="linear"),
        scheduler        = SchedulerConfig(type="cosine_annealing", epochs=200, eta_min=1e-6),
        ema              = EMAConfig(use_ema=False, ema_decay=0.999),
        optimizer        = OptimizerConfig(betas=(0.9, 0.999), eps=1e-8),
        io               = IOConfig(logdir="/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/help"),
        gradient_clipper = GradientClipperConfig(clip_mode="fixed", max_grad_norm=1.0),

        training = TrainingConfigInner(
            device                      = "gpu",
            epochs                      = 200,
            validation_frequency        = 1,
            use_amp                     = False,
            gradient_accumulation_steps = 1,
            max_grad_norm               = None,
            verbose                     = True,
        ),

        overfit = OverfitConfig(
            enabled        = False,
            max_steps      = 5,
            stop_threshold = 1e-6,
            batch_size     = 1,
        ),

        curriculum = CurriculumConfig(
            
            matching = MatchingCurriculumConfig(
                enabled              = False,
                warmup_strategy      = "sort_gt_by_mu",
                graduation_strategy  = "sort_gt_by_mu",
                swap_epoch           = 1000,
                
                reset_early_stopping = True,
                reset_lr             = False,
                reset_warmup         = True,
            ),
            
            loss = LossCurriculumConfig(
                enabled              = False,
                swap_epoch           = 100,
                
                reset_early_stopping = False,
                reset_lr             = False,
                reset_warmup         = False,
                
                warmup = LossConfig(
                    use_param_l1             = True,
                    weight_param_l1          = 1.0,
                    param_weights            = (1.0, 1.0, 1.0)
                ),
                
                complete = LossConfig(
                    use_param_l1             = True,
                    weight_param_l1          = 1.0,
                    use_charbonnier_curve    = True,
                    weight_charbonnier_curve = 0.3,
                ),
            ),
        ),
    )

    pipeline = TrainingPipeline(
        trainer_config = trainer_config,
        dataset_config = dataset_config,
        model_name     = "unet",
        model_config   = model_config,
        seed           = 0,
        run_name       = "unet_baseline_hard_new",
    )

    probe_config = LossScaleProbeConfig(
        enabled        = False,
        n_batches      = 100,
        reference      = "param_l1",
        exit_after     = True,
        enabled_losses = {},
    )

    pipeline.run(probe_config=probe_config)


if __name__ == "__main__":
    main()
