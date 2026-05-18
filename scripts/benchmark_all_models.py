"""
Benchmark all models across all available GPUs.

One model is trained per GPU in a queue — as soon as a GPU is freed
the next pending model is dispatched.  Each model gets its own
subdirectory under BenchmarkConfig.benchmark_dir.

Usage
-----
    python scripts/benchmark_all_models.py
"""
from __future__ import annotations

import os
os.environ["MKL_NUM_THREADS"]     = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"]     = "4"

import copy
import json
import logging
import multiprocessing as mp
import queue
import sys
import time
from dataclasses import dataclass, field
from datetime    import timedelta
from pathlib     import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax

from configuration.dataset_config import (
    DatasetCreationConfiguration,
    InputConfig,
    InputNormalizationMode,
    OutputNormalizationMode,
    PassDropConfig,
    PatchConfiguration,
    Representation,
    SplitRegions,
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
from models            import MODEL_REGISTRY
from pipelines.training_pipeline.pipeline import TrainingPipeline


# ══════════════════════════════════════════════════════════════════════
#  BENCHMARK CONFIGURATION — edit these to taste
# ══════════════════════════════════════════════════════════════════════
@dataclass
class BenchmarkConfig:
    # ── I/O ────────────────────────────────────────────────────────────
    preprocessing_run_dir : Path      = Path("/ste/rnd/User/vice_vi/Dataset/new_good")
    benchmark_dir         : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/benchmark")

    # ── Hardware ───────────────────────────────────────────────────────
    # Leave empty to auto-detect all available CUDA devices.
    gpus                  : list[int] = field(default_factory=list)
    # Cap the number of GPUs used (0 = no cap).
    max_gpus              : int       = 3

    # ── Dataset ────────────────────────────────────────────────────────
    batch_size            : int             = 400
    num_workers           : int             = 8
    patch_size            : tuple[int, int] = (64, 64)
    patch_stride          : int             = 32

    input_config          : InputConfig = field(default_factory=lambda: InputConfig(
        use_master=False, master_representation=Representation.MAG_ONLY,
        use_slaves=True,  slaves_representation=Representation.MAG_ONLY,
        use_interferograms=True, interferograms_representation=Representation.ANGLE_ONLY,
    ))

    input_normalization_mode  : InputNormalizationMode  = InputNormalizationMode.GROUPED
    output_normalization_mode : OutputNormalizationMode = OutputNormalizationMode.GROUPED

    # ── Train / val / test split ───────────────────────────────────────
    split_mode    : str             = "manual"
    train_ratio   : float           = 0.70
    val_ratio     : float           = 0.15
    train_azimuth : tuple[int, int] = (1000,  9120)
    val_azimuth   : tuple[int, int] = (9120,  12400)
    test_azimuth  : tuple[int, int] = (12400, 16000)

    # ── Per-model overrides ────────────────────────────────────────────
    # Any key present here is merged on top of the shared config above
    # before that model's training job is dispatched.
    # Supported override keys: batch_size, patch_size, patch_stride,
    #                          num_workers, epochs, validation_frequency.
    model_overrides : dict[str, dict] = field(default_factory=lambda: {
        "swin_unet" : {"batch_size": 64,  "patch_size": (64, 64)},
        "transunet" : {"batch_size": 64,  "patch_size": (64, 64)},
        "unetr"     : {"batch_size": 32,  "patch_size": (64, 64)},
    })

    # ── Shared trainer settings ────────────────────────────────────────
    epochs               : int   = 100
    validation_frequency : int   = 5


# ══════════════════════════════════════════════════════════════════════


# ── Helpers ────────────────────────────────────────────────────────────

def _effective(bench: BenchmarkConfig, model_name: str) -> BenchmarkConfig:
    """Return a shallow-copied BenchmarkConfig with per-model overrides applied."""
    overrides = bench.model_overrides.get(model_name, {})
    if not overrides:
        return bench
    patched = copy.copy(bench)
    for key, value in overrides.items():
        if not hasattr(patched, key):
            raise ValueError(f"[benchmark] Unknown override key '{key}' for model '{model_name}'.")
        setattr(patched, key, value)
    return patched


def _build_split_regions(bench: BenchmarkConfig, global_crop: CropRegion) -> SplitRegions:
    if bench.split_mode == "ratios":
        return SplitRegions.from_ratios(
            global_crop = global_crop,
            train_ratio = bench.train_ratio,
            val_ratio   = bench.val_ratio,
        )
    if bench.split_mode == "manual":
        return SplitRegions(
            train = CropRegion(bench.train_azimuth[0], bench.train_azimuth[1], global_crop.range_start, global_crop.range_end),
            val   = CropRegion(bench.val_azimuth[0],   bench.val_azimuth[1],   global_crop.range_start, global_crop.range_end),
            test  = CropRegion(bench.test_azimuth[0],  bench.test_azimuth[1],  global_crop.range_start, global_crop.range_end),
        )
    raise ValueError(f"Unknown split_mode '{bench.split_mode}'. Use 'ratios' or 'manual'.")


def _build_dataset_config(bench: BenchmarkConfig) -> DatasetCreationConfiguration:
    with open(bench.preprocessing_run_dir / "data" / "dataset.json", "r", encoding="utf-8") as f:
        layout = json.load(f)
    global_crop   = CropRegion(*layout["global_crop"])
    split_regions = _build_split_regions(bench, global_crop)
    split_regions.validate_against(global_crop)

    return DatasetCreationConfiguration(
        preprocessing_run_directory = bench.preprocessing_run_dir,
        split_regions               = split_regions,
        patch                       = PatchConfiguration(
            size                 = bench.patch_size,
            stride               = bench.patch_stride,
            use_reflective_padding = True,
        ),
        input_config              = bench.input_config,
        pass_drop_train           = PassDropConfig(drop_probs=0.0, min_kept_passes=1, seed=0),
        pass_drop_val             = PassDropConfig(drop_probs=0.0, min_kept_passes=1),
        pass_drop_test            = PassDropConfig(drop_probs=0.0, min_kept_passes=1),
        batch_size                = bench.batch_size,
        num_workers               = bench.num_workers,
        shuffle_train             = True,
        pin_memory                = True,
        input_normalization_mode  = bench.input_normalization_mode,
        output_normalization_mode = bench.output_normalization_mode,
    )


def _build_trainer_config(bench: BenchmarkConfig, model_name: str, gpu_id: int) -> TrainerConfig:
    logdir = str(bench.benchmark_dir / model_name)
    return TrainerConfig(
        gaussian       = GaussianConfig.from_dataset(bench.preprocessing_run_dir),
        early_stopping = EarlyStoppingConfig(patience=30, min_delta=0.0001, restore_best=True),
        warmup         = WarmupConfig(warmup_steps=100, warmup_start_factor=0.1, warmup_enabled=True),
        scheduler      = SchedulerConfig(epochs=bench.epochs, eta_min=1e-6),
        ema            = EMAConfig(use_ema=False, ema_decay=0.999),
        optimizer      = OptimizerConfig(betas=(0.9, 0.999), eps=1e-8),
        io             = IOConfig(logdir=logdir),
        training       = TrainingConfigInner(
            device                      = "gpu" if jax.devices("gpu") else "cpu",
            epochs                      = bench.epochs,
            validation_frequency        = bench.validation_frequency,
            use_amp                     = False,
            gradient_accumulation_steps = 1,
            max_grad_norm               = None,
            verbose                     = True,
            overfit_enabled             = False,
            deep_validation             = True,
            eval_train_split            = True,
        ),
        loss = LossConfig(
            use_ssim_curve    = True,
            weight_ssim_curve = 1.0,
            ssim_window_size  = 11,
            ssim_sigma        = 1.5,
            ssim_data_range   = 1.0,
            ssim_k1           = 0.01,
            ssim_k2           = 0.03,
        ),
    )


# ── Worker (runs in a child process) ──────────────────────────────────

def _worker(model_name: str, gpu_id: int, bench: BenchmarkConfig) -> None:
    """Entry point executed in a dedicated child process for one model."""
    # Isolate to a single GPU so CUDA sees only one device at the OS level.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logging.basicConfig(
        level  = logging.INFO,
        format = f"[%(asctime)s  gpu={gpu_id}  {model_name}]  %(message)s",
        datefmt = "%H:%M:%S",
    )
    log = logging.getLogger(model_name)

    effective_bench = _effective(bench, model_name)

    try:
        log.info(f"Starting  (batch_size={effective_bench.batch_size}  patch={effective_bench.patch_size})")
        dataset_config = _build_dataset_config(effective_bench)
        trainer_config = _build_trainer_config(effective_bench, model_name, gpu_id=0)  # visible device is always 0

        pipeline = TrainingPipeline(
            trainer_config = trainer_config,
            dataset_config = dataset_config,
            model_name     = model_name,
        )
        pipeline.run()
        log.info("Finished successfully.")

    except Exception as exc:  # noqa: BLE001
        log.error(f"FAILED — {exc}", exc_info=True)
        sys.exit(1)


# ── GPU queue dispatcher ───────────────────────────────────────────────

class _GPUQueue:
    """Dispatches model jobs to GPUs; never assigns more than one job per GPU."""

    def __init__(self, gpu_ids: list[int]) -> None:
        self._gpu_pool : queue.Queue[int] = queue.Queue()
        for gid in gpu_ids:
            self._gpu_pool.put(gid)

    def run(self, model_names: list[str], bench: BenchmarkConfig) -> dict[str, str]:
        """
        Dispatch all models, blocking until all have finished.
        Returns a status dict: model_name -> "ok" | "failed".
        """
        pending    : list[str]                             = list(model_names)
        running    : dict[str, tuple[mp.Process, int, float]] = {}  # name -> (proc, gpu, t0)
        statuses   : dict[str, str]                        = {}

        while pending or running:
            # ── poll running processes ──────────────────────────────────
            for name in list(running):
                proc, gpu_id, t0 = running[name]
                if not proc.is_alive():
                    elapsed       = timedelta(seconds=int(time.time() - t0))
                    status        = "ok" if proc.exitcode == 0 else f"failed (exit={proc.exitcode})"
                    statuses[name] = status
                    del running[name]
                    self._gpu_pool.put(gpu_id)
                    _log.info(
                        f"  {'✓' if proc.exitcode == 0 else '✗'}  {name:<20}  gpu={gpu_id}  "
                        f"elapsed={elapsed}  status={status}"
                    )

            # ── dispatch pending jobs if a GPU is free ──────────────────
            while pending:
                try:
                    gpu_id = self._gpu_pool.get_nowait()
                except queue.Empty:
                    break
                model_name = pending.pop(0)
                t0         = time.time()
                proc       = mp.Process(
                    target = _worker,
                    args   = (model_name, gpu_id, bench),
                    name   = f"bench-{model_name}",
                    daemon = False,
                )
                proc.start()
                running[model_name] = (proc, gpu_id, t0)
                _log.info(f"  →  dispatched {model_name:<20}  on gpu={gpu_id}")

            time.sleep(2.0)

        return statuses


# ── Main ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "[%(asctime)s]  %(message)s",
    datefmt = "%H:%M:%S",
)
_log = logging.getLogger("benchmark")


def main() -> None:
    bench = BenchmarkConfig()

    # ── Resolve GPU list ────────────────────────────────────────────────
    if bench.gpus:
        gpu_ids = bench.gpus
    elif jax.devices("gpu"):
        gpu_ids = list(range(len(jax.devices("gpu"))))
    else:
        _log.warning("No GPU devices found — falling back to device 0.")
        gpu_ids = [0]

    if bench.max_gpus > 0:
        gpu_ids = gpu_ids[:bench.max_gpus]

    bench.benchmark_dir.mkdir(parents=True, exist_ok=True)

    model_names = sorted(MODEL_REGISTRY.keys())

    # ── Print plan ──────────────────────────────────────────────────────
    col_w = max(len(n) for n in model_names) + 2
    _log.info("")
    _log.info("══════════════════════════════════════════════════════════")
    _log.info("  Benchmark plan")
    _log.info(f"  Models      : {len(model_names)}")
    _log.info(f"  GPUs        : {gpu_ids}  (max_gpus={bench.max_gpus if bench.max_gpus > 0 else 'unlimited'})")
    _log.info(f"  Output dir  : {bench.benchmark_dir}")
    _log.info("──────────────────────────────────────────────────────────")
    _log.info(f"  {'Model':<{col_w}}  {'Batch':>6}  {'Patch':>10}  {'Overrides'}")
    _log.info(f"  {'-'*col_w}  {'------':>6}  {'----------':>10}  {'---------'}")
    for name in model_names:
        eff  = _effective(bench, name)
        ovrd = str(bench.model_overrides.get(name, {})) if name in bench.model_overrides else "—"
        _log.info(f"  {name:<{col_w}}  {eff.batch_size:>6}  {str(eff.patch_size):>10}  {ovrd}")
    _log.info("══════════════════════════════════════════════════════════")
    _log.info("")

    # ── Run ─────────────────────────────────────────────────────────────
    t_start  = time.time()
    gpu_queue = _GPUQueue(gpu_ids)
    statuses  = gpu_queue.run(model_names, bench)
    elapsed   = timedelta(seconds=int(time.time() - t_start))

    # ── Summary table ───────────────────────────────────────────────────
    _log.info("")
    _log.info("══════════════════════════════════════════════════════════")
    _log.info("  Benchmark summary")
    _log.info(f"  {'Model':<{col_w}}  Status")
    _log.info(f"  {'-'*col_w}  ------")
    for name in model_names:
        status = statuses.get(name, "not run")
        marker = "✓" if status == "ok" else "✗"
        _log.info(f"  {marker}  {name:<{col_w}}  {status}")
    _log.info(f"  Total elapsed : {elapsed}")
    _log.info("══════════════════════════════════════════════════════════")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
