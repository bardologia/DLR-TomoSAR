from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


NOTEBOOK_DIR = Path(__file__).resolve().parent


def md(text):
    return new_markdown_cell(text.strip("\n"))


def code(text):
    return new_code_cell(text.strip("\n"))


STYLE_CELL = '''
import sys
from pathlib import Path

_REPO_ROOT = Path.cwd()
while _REPO_ROOT != _REPO_ROOT.parent and not (_REPO_ROOT / "pipelines").is_dir():
    _REPO_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import Image, display

mpl.rcParams.update({
    "font.family"       : "serif",
    "font.size"         : 11,
    "axes.labelsize"    : 12,
    "axes.titlesize"    : 13,
    "legend.fontsize"   : 9,
    "xtick.labelsize"   : 10,
    "ytick.labelsize"   : 10,
    "figure.dpi"        : 150,
    "savefig.dpi"       : 300,
    "savefig.bbox"      : "tight",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

from tools.benchmarking import LoaderSpec, GpuFeedBenchmark, DataLoaderSweep, SweepReport

print("Repository root:", _REPO_ROOT)
print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
'''


HARDWARE_CELL = '''
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Selected device :", device)
print("CPU count        :", os.cpu_count())

if torch.cuda.is_available():
    for gpu_id in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(gpu_id)
        total_gb   = properties.total_memory / (1024.0 ** 3)
        print(f"GPU {gpu_id}            : {properties.name}  ({total_gb:.1f} GB, {properties.multi_processor_count} SMs)")

print("CUDA_VISIBLE_DEVICES :", os.environ.get("CUDA_VISIBLE_DEVICES", "(unset, all visible)"))

import psutil
memory = psutil.virtual_memory()
print(f"System RAM        : {memory.total / (1024.0 ** 3):.1f} GB total, {memory.available / (1024.0 ** 3):.1f} GB available")
'''


SWEEP_CELL = '''
benchmark = GpuFeedBenchmark(
    dataset        = train_dataset,
    model          = model,
    to_model_input = to_model_input,
    device         = device,
    use_amp        = USE_AMP,
    seed           = SEED,
    warmup_batches = WARMUP_BATCHES,
    timed_batches  = TIMED_BATCHES,
    gpu_index      = GPU_INDEX,
)

main_specs = [
    LoaderSpec(batch_size=batch_size, num_workers=workers, prefetch_factor=REFERENCE_PREFETCH_FACTOR, pin_memory=True, persistent_workers=True)
    for batch_size in BATCH_SIZES
    for workers in WORKER_COUNTS
]

print(f"Sweeping {len(main_specs)} configurations "
      f"({WARMUP_BATCHES} warmup + {TIMED_BATCHES} timed batches each)\\n")

def _report_progress(record):
    if record.get("status") != "ok":
        print(f"  [{record['status'].upper():4s}] bs={record['batch_size']:>5} workers={record['num_workers']}")
        return
    print(f"  bs={record['batch_size']:>5}  workers={record['num_workers']}  "
          f"throughput={record['end_to_end_samples_per_s']:>10.0f} samp/s  "
          f"data_wait={100.0 * record['data_wait_fraction']:>5.1f}%  "
          f"gpu_util={record['gpu_util_mean']:>5.1f}%  "
          f"feed_ratio={record['feed_ratio']:.2f}")

main_results = DataLoaderSweep(benchmark, main_specs, on_result=_report_progress).run()
print("\\nSweep complete.")
'''


REPORT_CELL = '''
report  = SweepReport(main_results, wait_threshold=DATA_WAIT_TARGET)
columns = [
    "batch_size", "num_workers", "loader_only_samples_per_s",
    "compute_ceiling_samples_per_s", "end_to_end_samples_per_s",
    "compute_efficiency", "data_wait_fraction", "feed_ratio",
    "gpu_util_mean", "gpu_util_max", "vram_peak_gb",
]
ordered = report.dataframe.sort_values("end_to_end_samples_per_s", ascending=False)
display(ordered[[column for column in columns if column in ordered]].round(3))
'''


FIGURES_CELL = '''
saved_figures = report.save_all(FIGURE_DIR)
for figure_path in saved_figures:
    print(figure_path)
    display(Image(filename=str(figure_path)))
'''


REFINE_CELL = '''
recommendation = report.recommendation
reference_batch_size  = recommendation["batch_size"]
reference_num_workers = max(1, recommendation["num_workers"])

refine_specs = [
    LoaderSpec(batch_size=reference_batch_size, num_workers=reference_num_workers, prefetch_factor=prefetch, pin_memory=pin_memory, persistent_workers=True)
    for prefetch in PREFETCH_FACTORS
    for pin_memory in (True, False)
]

print(f"Refining prefetch_factor and pin_memory at batch_size={reference_batch_size}, num_workers={reference_num_workers}\\n")

refine_results = DataLoaderSweep(benchmark, refine_specs, on_result=_report_progress).run()

refine_report = SweepReport(refine_results, wait_threshold=DATA_WAIT_TARGET)
refine_table  = refine_report.dataframe.sort_values("end_to_end_samples_per_s", ascending=False)
display(refine_table[["prefetch_factor", "pin_memory", "end_to_end_samples_per_s", "data_wait_fraction", "gpu_util_mean", "vram_peak_gb"]].round(3))
'''


ASSERT_CELL_TEMPLATE = '''
best_main   = report.recommendation
best_refine = refine_report.dataframe.sort_values("end_to_end_samples_per_s", ascending=False).iloc[0]

final = {
    "batch_size"         : best_main["batch_size"],
    "num_workers"        : best_main["num_workers"],
    "prefetch_factor"    : int(best_refine["prefetch_factor"]),
    "pin_memory"         : bool(best_refine["pin_memory"]),
    "persistent_workers" : True,
}

print("=" * 64)
print("RECOMMENDED DATALOADER CONFIGURATION")
print("=" * 64)
for key, value in final.items():
    print(f"  {key:<20}: {value}")
print(f"  {'cpu_bound':<20}: {best_main['cpu_bound']}")
print(f"  {'mean GPU util':<20}: {best_main['gpu_util_mean']:.1f}%")
print(f"  {'data wait':<20}: {100.0 * best_main['data_wait_fraction']:.1f}%")
print(f"  {'feed ratio':<20}: {best_main['feed_ratio']:.2f}")
print("=" * 64)

checks = [
    ("A successful configuration was found",        best_main.get("found", False)),
    ("Recommended batch size is within the sweep",  final["batch_size"] in BATCH_SIZES),
    ("Recommended worker count is within the sweep", final["num_workers"] in WORKER_COUNTS),
    ("GPU sampling produced utilization data",      not np.isnan(best_main["gpu_util_mean"])),
    ("GPU is not data-starved at the recommendation", best_main["data_wait_fraction"] <= DATA_WAIT_TARGET),
]
print()
for description, condition in checks:
    print(f"[{'PASS' if condition else 'FAIL'}] {description}")

if best_main["cpu_bound"]:
    print("\\nNOTE: every configuration left the GPU data-starved (data wait above target).")
    print("      The CPU input pipeline is the ceiling. Beyond max workers/prefetch, the")
    print("      durable fix is to lower the per-item cost in __ITEM_SOURCE__.")

print("\\nApply by editing the defaults so production runs inherit the tuned values:")
print(f"  - __CONFIG_HINT__")
print(f"    set batch_size={final['batch_size']}, num_workers={final['num_workers']}")
print(f"  - pipelines/shared/loaders.py Loader.build")
print(f"    set prefetch_factor={final['prefetch_factor']}, pin_memory={final['pin_memory']}")
'''


def build_profile_notebook():
    config_cell = f'''
DATASET_PATH    = "/ste/rnd/User/vice_vi/Dataset/base_dataset_w20_10"
PARAMETERS_PATH = "/ste/rnd/User/vice_vi/Dataset/base_dataset_w20_10/params/params_Ng3_sigonly_k5/parameters_Ng3_sigonly_k5.npy"

MODEL_NAME  = "mlp_ae"
N_GAUSSIANS = 5
SEED        = 0
USE_AMP     = False
GPU_INDEX   = None

PIXEL_SUBSAMPLE = 0.2
KEEP_EMPTY_FRAC = 0.05

BATCH_SIZES               = [256, 512, 1024, 2048, 4096]
WORKER_COUNTS             = [0, 2, 4, 6, 8]
PREFETCH_FACTORS          = [2, 4, 8, 16]
REFERENCE_PREFETCH_FACTOR = 4

WARMUP_BATCHES   = 8
TIMED_BATCHES    = 80
DATA_WAIT_TARGET = 0.05

WORK_DIR   = Path("/tmp/tune_dataloader_profile_autoencoder")
FIGURE_DIR = Path("figures/tune_dataloader_profile_autoencoder")
'''

    dataset_cell = '''
from configuration.training import ProfileAeEntryConfig
from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline
from pipelines.profile_autoencoder.dataset.pipeline   import ProfileDatasetPipeline

entry = ProfileAeEntryConfig(n_gaussians=N_GAUSSIANS, ae_model_name=MODEL_NAME, seed=SEED)
entry.paths.dataset_path    = Path(DATASET_PATH)
entry.paths.parameters_path = Path(PARAMETERS_PATH)
entry.pixel_subsample       = PIXEL_SUBSAMPLE
entry.keep_empty_frac       = KEEP_EMPTY_FRAC

training_pipeline      = TrainingPipeline(entry)
profile_dataset_config = training_pipeline._profile_dataset_config()

dataset_pipeline = ProfileDatasetPipeline(profile_dataset_config, WORK_DIR, seed=SEED)
(_loaders, datasets, profile_axis, profile_length, _normalizer) = dataset_pipeline.run()

train_dataset = datasets["train"]

print("Train profiles :", len(train_dataset))
print("Profile length :", profile_length)
print("Sample shape   :", np.asarray(train_dataset[0]).shape, np.asarray(train_dataset[0]).dtype)
'''

    model_cell = '''
from models.profile_autoencoder import get_profile_autoencoder

model, model_config = get_profile_autoencoder(MODEL_NAME, training_pipeline.autoencoder_cfg, profile_length=profile_length)

def to_model_input(batch, target_device):
    return batch.to(target_device, non_blocking=True).unsqueeze(-1).unsqueeze(-1)

parameter_count = sum(parameter.numel() for parameter in model.parameters())

print("Model            :", MODEL_NAME)
print("Parameters       :", f"{parameter_count:,}")
print("Embedding dim    :", model_config.embedding_dim)
print("Model input shape:", tuple(to_model_input(torch.from_numpy(np.stack([train_dataset[i] for i in range(4)])), device).shape))
'''

    assert_cell = (ASSERT_CELL_TEMPLATE
        .replace("__ITEM_SOURCE__", "ProfileDataset.__getitem__ (the per-profile GaussianMixture.evaluate_batch synthesis)")
        .replace("__CONFIG_HINT__", "configuration/dataset/profile_autoencoder.py ProfileDatasetConfig"))

    cells = [
        md('''
# DataLoader Tuning: Profile Autoencoder

This notebook finds the DataLoader settings (**batch size, worker count, prefetch factor, pin-memory**) that keep the GPU fed while training the **profile autoencoder**, eliminating the GPU-starvation observed during training.

The profile dataset synthesises a Gaussian-mixture curve **on the CPU per item** (`ProfileDataset.__getitem__` → `GaussianMixture.evaluate_batch`). When the worker pool cannot synthesise curves as fast as the GPU consumes them, the GPU idles waiting for data. The sweep quantifies that idle time and recommends the configuration that removes it.

**Method.** For each candidate configuration the harness measures three throughputs:

1. **Loader-only** — iterate the loader, touching no GPU. The maximum rate the CPU pipeline can deliver.
2. **Compute ceiling** — repeatedly run the full train step on one GPU-resident batch, with no loader. The maximum rate the GPU can consume.
3. **End-to-end** — real loader feeding the real train step, decomposed into *data-wait* and *compute* time.

A configuration **saturates** the GPU when `feed_ratio = loader_only / compute_ceiling >= 1` and the *data-wait fraction* is below the target. The recommendation is the saturating configuration with the highest end-to-end throughput and the fewest workers.
'''),
        code(STYLE_CELL),
        md("## Configuration\n\nEvery tunable lives here. On the server, point `DATASET_PATH` / `PARAMETERS_PATH` at the preprocessing run and set `GPU_INDEX` to the physical device id (matching `CUDA_VISIBLE_DEVICES`) so GPU-utilization sampling reads the right card."),
        code(config_cell),
        md("## Hardware probe\n\n> **What you should see:** a CUDA device with several GB of memory and a multi-processor count in the tens. If the device is `cpu`, the compute ceiling is meaningless and the sweep only characterises the loader."),
        code(HARDWARE_CELL),
        md('''
## Stage 1 — Build the real training dataset and model

Reuses the project pipeline (`TrainingPipeline` → `ProfileDatasetPipeline`) so the per-item cost measured here is exactly the cost paid in production. `PIXEL_SUBSAMPLE` only changes how many profiles are loaded, not the per-profile synthesis cost, so a fraction below 1.0 keeps the build fast without biasing the throughput measurement.

> **What you should see:** a train split with thousands-to-millions of profiles, a profile length equal to the elevation-axis resolution, and a `float32` sample of shape `(profile_length,)`.
'''),
        code(dataset_cell),
        code(model_cell),
        md('''
## Stage 2a — Sweep batch size and worker count

The two dominant knobs for starvation. Prefetch is held at `REFERENCE_PREFETCH_FACTOR` and refined separately in Stage 2b.

> **What you should see:** throughput rising with workers until it plateaus at the compute ceiling (GPU-bound) or at the CPU feed limit (CPU-bound). `data_wait` should fall toward 0% as workers increase if the pipeline can outpace the GPU.
'''),
        code(SWEEP_CELL),
        code(REPORT_CELL),
        md("## Stage 3 — Figures\n\nOne figure per file under `FIGURE_DIR`, publication quality. The feed-ratio plot is the clearest starvation diagnostic: points above the dashed line mean the CPU can outpace the GPU at that configuration."),
        code(FIGURES_CELL),
        md('''
## Stage 2b — Refine prefetch factor and pin-memory

At the batch size and worker count chosen by Stage 2a, sweep `prefetch_factor` and `pin_memory`.

> **What you should see:** throughput largely flat across prefetch once it is "enough" (deeper queues only help if workers are bursty); `pin_memory=True` should match or beat `False` for CUDA transfers. Pick the smallest prefetch that reaches the plateau to limit host-RAM use.
'''),
        code(REFINE_CELL),
        md("## Recommendation\n\nLayered checks (structural → starvation), the final configuration, and exactly where to write it back so every training run inherits it."),
        code(assert_cell),
        md('''
### Common mistakes — DataLoader tuning

| Symptom | Likely cause | How to diagnose |
|---|---|---|
| `gpu_util_mean` near 0 and `data_wait` high at every config | CPU `__getitem__` is the ceiling, not the loader settings | Compare `loader_only` to `compute_ceiling`; if `feed_ratio < 1` even at max workers, vectorise/cache the per-item synthesis |
| Throughput collapses at the largest batch | VRAM pressure or `status == "oom"` rows | Check `vram_peak_gb`; drop the batch or enable AMP |
| More workers makes it *slower* | Worker startup / shared-memory contention dominates a cheap item | Look for the throughput peak then decline vs `num_workers`; pick the peak, not the maximum |
| `loader_only` inflated vs real training | Tiny `PIXEL_SUBSAMPLE` re-iterated from OS cache | Raise `PIXEL_SUBSAMPLE`; confirm the dataset spans more than `TIMED_BATCHES` batches |
| GPU util reads 0 but training is fast | `GPU_INDEX` does not match the device under `CUDA_VISIBLE_DEVICES` | Set `GPU_INDEX` to the physical id; re-run the hardware probe |
| Results jitter run to run | Too few timed batches, or a shared GPU under other load | Raise `TIMED_BATCHES`; run on an idle card |

**How to apply the result:** edit the two defaults named in the recommendation cell (`ProfileDatasetConfig` for batch size / workers, `Loader.build` for prefetch / pin-memory). Re-run this notebook after the edit; the recommendation should now report the GPU as not starved.
'''),
    ]

    notebook = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return notebook


def build_image_notebook():
    config_cell = f'''
DATASET_PATH    = "/ste/rnd/User/vice_vi/Dataset/base_dataset_w20_10"
PARAMETERS_PATH = "/ste/rnd/User/vice_vi/Dataset/base_dataset_w20_10/params/params_Ng3_sigonly_k5/parameters_Ng3_sigonly_k5.npy"

MODEL_NAME  = "conv2d_ae"
N_GAUSSIANS = 5
SEED        = 0
USE_AMP     = False
GPU_INDEX   = None

BATCH_SIZES               = [4, 8, 16, 32, 64]
WORKER_COUNTS             = [0, 4, 8, 12, 16]
PREFETCH_FACTORS          = [2, 4, 8]
REFERENCE_PREFETCH_FACTOR = 4

WARMUP_BATCHES   = 5
TIMED_BATCHES    = 40
DATA_WAIT_TARGET = 0.05

WORK_DIR   = Path("/tmp/tune_dataloader_image_autoencoder")
FIGURE_DIR = Path("figures/tune_dataloader_image_autoencoder")
'''

    dataset_cell = '''
from configuration.training.image_autoencoder import ImageAeEntryConfig
from pipelines.image_autoencoder.training.pipeline import TrainingPipeline
from pipelines.backbone.dataset.pipeline           import DatasetPipeline

entry = ImageAeEntryConfig(n_gaussians=N_GAUSSIANS, ae_model_name=MODEL_NAME, seed=SEED)
entry.paths.dataset_path    = Path(DATASET_PATH)
entry.paths.parameters_path = Path(PARAMETERS_PATH)

training_pipeline = TrainingPipeline(entry)
dataset_config    = training_pipeline.dataset_config
gaussian_config   = training_pipeline.trainer_config.gaussian

dataset_config.n_gaussians = gaussian_config.n_default_gaussians

dataset_pipeline = DatasetPipeline(dataset_config, WORK_DIR, seed=SEED)
profile_length   = dataset_pipeline.profile_length
profile_axis     = np.linspace(gaussian_config.x_min, gaussian_config.x_max, profile_length, dtype=np.float32)

dataset_config.x_axis = profile_axis

_train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()

train_dataset = datasets["train"]
input_channels = train_dataset.input_channels

print("Train patches  :", len(train_dataset))
print("Input channels :", input_channels)
print("Patch shape    :", tuple(np.asarray(train_dataset[0][0]).shape), np.asarray(train_dataset[0][0]).dtype)
'''

    model_cell = '''
from models.image_autoencoder import get_image_autoencoder

model, model_config = get_image_autoencoder(MODEL_NAME, training_pipeline.autoencoder_cfg, in_channels=input_channels)

def to_model_input(batch, target_device):
    return batch[0].to(target_device, non_blocking=True)

parameter_count = sum(parameter.numel() for parameter in model.parameters())
first_patch     = torch.from_numpy(np.stack([train_dataset[i][0] for i in range(4)]))

print("Model            :", MODEL_NAME)
print("Parameters       :", f"{parameter_count:,}")
print("Embedding dim    :", model_config.embedding_dim)
print("Model input shape:", tuple(first_patch.shape))
'''

    assert_cell = (ASSERT_CELL_TEMPLATE
        .replace("__ITEM_SOURCE__", "PatchDataset.__getitem__ (patch extraction + complex->representation conversion)")
        .replace("__CONFIG_HINT__", "configuration/dataset/general/dataset.py DatasetConfig"))

    cells = [
        md('''
# DataLoader Tuning: Image Autoencoder

This notebook finds the DataLoader settings (**batch size, worker count, prefetch factor, pin-memory**) that keep the GPU fed while training the **image autoencoder**, eliminating the GPU-starvation observed during training.

Each item is a spatial patch: `PatchDataset.__getitem__` extracts the patch, converts the complex SAR stack into the configured channel representation, normalises, and augments — all on the CPU. When workers cannot prepare patches as fast as the GPU consumes them, the GPU idles. The sweep quantifies that idle time and recommends the configuration that removes it.

**Method.** For each candidate configuration the harness measures three throughputs:

1. **Loader-only** — iterate the loader, touching no GPU. Maximum CPU delivery rate.
2. **Compute ceiling** — repeatedly run the full train step on one GPU-resident batch, no loader. Maximum GPU consumption rate.
3. **End-to-end** — real loader feeding the real train step, decomposed into *data-wait* and *compute* time.

A configuration **saturates** the GPU when `feed_ratio = loader_only / compute_ceiling >= 1` and the *data-wait fraction* is below the target. The recommendation is the saturating configuration with the highest end-to-end throughput and the fewest workers.
'''),
        code(STYLE_CELL),
        md("## Configuration\n\nEvery tunable lives here. On the server, point `DATASET_PATH` / `PARAMETERS_PATH` at the preprocessing run and set `GPU_INDEX` to the physical device id (matching `CUDA_VISIBLE_DEVICES`) so GPU-utilization sampling reads the right card. Image patches are large, so batch sizes are small and worker counts go higher than for the profile autoencoder."),
        code(config_cell),
        md("## Hardware probe\n\n> **What you should see:** a CUDA device with several GB of memory and a multi-processor count in the tens. If the device is `cpu`, the compute ceiling is meaningless and the sweep only characterises the loader."),
        code(HARDWARE_CELL),
        md('''
## Stage 1 — Build the real training dataset and model

Reuses the project pipeline (`TrainingPipeline` → `DatasetPipeline`), mirroring `BackboneDatasetPreparation`, so the per-patch cost measured here is exactly the cost paid in production.

> **What you should see:** a train split with thousands of patches, an input-channel count equal to `total_channels` for the configured input representation, and a `float32` patch of shape `(input_channels, patch_h, patch_w)`.
'''),
        code(dataset_cell),
        code(model_cell),
        md('''
## Stage 2a — Sweep batch size and worker count

The two dominant knobs for starvation. Prefetch is held at `REFERENCE_PREFETCH_FACTOR` and refined separately in Stage 2b.

> **What you should see:** throughput rising with workers until it plateaus at the compute ceiling (GPU-bound) or at the CPU feed limit (CPU-bound). `data_wait` should fall toward 0% as workers increase if the pipeline can outpace the GPU.
'''),
        code(SWEEP_CELL),
        code(REPORT_CELL),
        md("## Stage 3 — Figures\n\nOne figure per file under `FIGURE_DIR`, publication quality. The feed-ratio plot is the clearest starvation diagnostic: points above the dashed line mean the CPU can outpace the GPU at that configuration."),
        code(FIGURES_CELL),
        md('''
## Stage 2b — Refine prefetch factor and pin-memory

At the batch size and worker count chosen by Stage 2a, sweep `prefetch_factor` and `pin_memory`.

> **What you should see:** throughput largely flat across prefetch once it is "enough"; `pin_memory=True` should match or beat `False` for CUDA transfers. Pick the smallest prefetch that reaches the plateau to limit host-RAM use.
'''),
        code(REFINE_CELL),
        md("## Recommendation\n\nLayered checks (structural → starvation), the final configuration, and exactly where to write it back so every training run inherits it."),
        code(assert_cell),
        md('''
### Common mistakes — DataLoader tuning

| Symptom | Likely cause | How to diagnose |
|---|---|---|
| `gpu_util_mean` near 0 and `data_wait` high at every config | CPU `__getitem__` is the ceiling, not the loader settings | Compare `loader_only` to `compute_ceiling`; if `feed_ratio < 1` even at max workers, cache decoded patches or precompute the representation |
| Throughput collapses at the largest batch | VRAM pressure or `status == "oom"` rows | Check `vram_peak_gb`; drop the batch or enable AMP |
| More workers makes it *slower* | Worker startup / shared-memory contention, or `/dev/shm` exhaustion | Watch for the throughput peak then decline vs `num_workers`; check shared memory if workers die |
| `pin_memory=True` slower than `False` | Host RAM under pressure; pinning competes for pageable memory | Compare the Stage 2b rows; check available RAM in the hardware probe |
| GPU util reads 0 but training is fast | `GPU_INDEX` does not match the device under `CUDA_VISIBLE_DEVICES` | Set `GPU_INDEX` to the physical id; re-run the hardware probe |
| Results jitter run to run | Too few timed batches, or a shared GPU under other load | Raise `TIMED_BATCHES`; run on an idle card |

**How to apply the result:** edit the two defaults named in the recommendation cell (`DatasetConfig` for batch size / workers, `Loader.build` for prefetch / pin-memory). Re-run this notebook after the edit; the recommendation should now report the GPU as not starved.
'''),
    ]

    notebook = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return notebook


def main():
    profile_path = NOTEBOOK_DIR / "tune_dataloader_profile_autoencoder.ipynb"
    image_path   = NOTEBOOK_DIR / "tune_dataloader_image_autoencoder.ipynb"

    nbformat.write(build_profile_notebook(), profile_path)
    nbformat.write(build_image_notebook(),   image_path)

    print("Wrote", profile_path)
    print("Wrote", image_path)


if __name__ == "__main__":
    main()
