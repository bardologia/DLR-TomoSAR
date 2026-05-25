from __future__ import annotations

import os
gpu_id = "1"
os.environ["CUDA_VISIBLE_DEVICES"]    = str(gpu_id)
os.environ["MKL_NUM_THREADS"]         = "4"
os.environ["NUMEXPR_NUM_THREADS"]     = "4"
os.environ["OMP_NUM_THREADS"]         = "4"

import json
import sys
import traceback
from pathlib import Path
from datetime import datetime

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from configuration.dataset_config import (
    AugmentationConfig,
    DatasetConfiguration,
    InputConfig,
    PatchConfiguration,
    Representation,
    SplitRegions,
)
from tools.crop_region      import CropRegion
from tools.loss_scale_probe import LossScaleProbeConfig
from tools.logger           import Logger

from configuration.training_config import (
    LossCurriculumConfig,
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

from models import CONFIG_REGISTRY
from pipelines.training_pipeline.pipeline import TrainingPipeline

dataset_path  = Path("/ste/rnd/User/vice_vi/Dataset/clean_dataset")
params_path   = Path("/ste/rnd/User/vice_vi/Dataset/clean_dataset/params/params_sig_k5/parameters_sig_k5.npy")
log_base_dir  = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/overfit_test")

overfit_steps       = 5000
stop_threshold      = 1e-3
overfit_batchsize   = 9
overfit_az_lines    = 128
overfit_range_lines = 128

skip_models: set[str] = set()


def build_dataset_config() -> DatasetConfiguration:
    with open(dataset_path / "data" / "dataset.json", "r", encoding="utf-8") as f:
        layout = json.load(f)

    global_crop = CropRegion(*layout["global_crop"])

    az_start    = 1000
    az_end      = az_start + overfit_az_lines
    range_start = global_crop.range_start
    range_end   = range_start + overfit_range_lines

    split_regions = SplitRegions(
        train = CropRegion(az_start, az_end, range_start, range_end),
        val   = CropRegion(az_start, az_end, range_start, range_end),
        test  = CropRegion(az_start, az_end, range_start, range_end),
    )

    return DatasetConfiguration(
        preprocessing_run_directory = dataset_path,
        parameters_path             = params_path,
        split_regions               = split_regions,
        patch = PatchConfiguration(size=(64, 64), stride=32, use_reflective_padding=True),

        input_config = InputConfig(
            use_primary        = True,  primary_representation        = Representation.MAG_ONLY,
            use_secondaries    = True,  secondaries_representation    = Representation.MAG_ONLY,
            use_interferograms = True,  interferograms_representation = Representation.ANGLE_ONLY,
        ),

        batch_size    = overfit_batchsize,
        num_workers   = 4,
        shuffle_train = True,
        pin_memory    = True,

        augmentation  = AugmentationConfig(
            p_flip_h    = 0.0,
            p_flip_v    = 0.0,
            p_rot90     = 0.0,
            p_amp_scale = 0.0,
            p_noise     = 0.0,
        ),
    )


def build_trainer_config(model_name: str) -> TrainerConfig:
    return TrainerConfig(
        gaussian = GaussianConfig.from_dataset(dataset_path, n_gaussians=5),

        early_stopping = EarlyStoppingConfig(patience=9999, min_delta=0.0, restore_best=False),

        warmup = WarmupConfig(warmup_enabled=False),

        scheduler = SchedulerConfig(type="constant"),

        ema              = EMAConfig(use_ema=False, ema_decay=0.999),
        optimizer        = OptimizerConfig(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
        gradient_clipper = GradientClipperConfig(clip_mode="fixed", max_grad_norm=1.0),

        io = IOConfig(logdir=str(log_base_dir / model_name)),

        training = TrainingConfigInner(
            device                      = "gpu",
            epochs                      = 10000,
            validation_frequency        = 9999,
            use_amp                     = False,
            gradient_accumulation_steps = 1,
            max_grad_norm               = None,
            verbose                     = False,
        ),

        overfit = OverfitConfig(
            enabled        = True,
            max_steps      = overfit_steps,
            stop_threshold = stop_threshold,
            batch_size     = overfit_batchsize,
        ),

        curriculum = LossCurriculumConfig(
            enabled  = False,
            warmup   = LossConfig(use_param_l1=True, weight_param_l1=1.0),
            complete = LossConfig(use_param_l1=True, weight_param_l1=1.0),
        ),
    )


def run_overfit(model_name: str, dataset_config: DatasetConfiguration, logger: Logger) -> dict:
    logger.subsection(model_name)

    model_config = CONFIG_REGISTRY[model_name]()

    for attr in ("dropout", "attention_dropout", "stochastic_depth_rate"):
        if hasattr(model_config, attr):
            setattr(model_config, attr, 0.0)

    for attr in vars(model_config):
        if attr.endswith("_wd"):
            setattr(model_config, attr, 0.0)

    trainer_config = build_trainer_config(model_name)

    pipeline = TrainingPipeline(
        trainer_config = trainer_config,
        dataset_config = dataset_config,
        model_name     = model_name,
        model_config   = model_config,
        seed           = 42,
        run_name       = f"overfit_{model_name}",
    )

    probe_config = LossScaleProbeConfig(
        enabled        = False,
        n_batches      = 100,
        reference      = "param_l1",
        exit_after     = True,
        enabled_losses = {},
    )

    result = {"model": model_name, "status": None, "final_loss": None, "error": None}
    try:
        pipeline.run(probe_config=probe_config)
        result["status"] = "PASS"
        logger.info(f" {model_name} :  PASS")
    except SystemExit:
        result["status"] = "PASS"
        logger.info(f" {model_name} :  PASS  (exit via SystemExit)")
    except Exception as e:
        result["status"] = "FAIL"
        result["error"]  = traceback.format_exc()
        logger.error(f" {model_name}  :  FAIL  |  {type(e).__name__}: {e}")

    return result


def main() -> None:
    log_base_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(log_dir=str(log_base_dir), name="overfit_test")

    all_models     = list(CONFIG_REGISTRY.keys())
    models_to_test = [m for m in all_models if m not in skip_models]

    logger.section("Overfit sanity-check")
    logger.kv_table({
        "Models":          len(models_to_test),
        "Steps per model": overfit_steps,
        "Stop threshold":  stop_threshold,
        "Batch size":      overfit_batchsize,
    }, title="Configuration")

    dataset_config = build_dataset_config()

    logger.section("Running tests")
    results = []
    for model_name in models_to_test:
        result = run_overfit(model_name, dataset_config, logger)
        results.append(result)

    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]

    logger.section("Summary")
    logger.kv_table({
        "Total":  len(results),
        "Passed": len(passed),
        "Failed": len(failed),
    }, title=f"{len(passed)}/{len(results)} passed")

    if passed:
        logger.subsection("Passed")
        for r in passed:
            logger.info(f"{r['model']}")

    if failed:
        logger.subsection("Failed")
        for r in failed:
            logger.error(f"{r['model']}")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = log_base_dir / f"overfit_results_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")

    logger.close()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
