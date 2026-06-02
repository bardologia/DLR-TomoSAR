# Inference Pipeline — Technical Reference

**Package:** `pipelines.inference_pipeline`  
**Location:** `pipelines/inference_pipeline/`  
**Role:** Applies a trained neural network to a SAR tomographic dataset split, reconstructs full-spatial-extent Gaussian PSD cubes via windowed overlap-add stitching, computes a comprehensive suite of evaluation metrics, generates publication-quality figures, GIF animations, and assembles a self-contained Markdown report.

---

## Table of Contents

1. [Overview](#1-overview)  
2. [Architecture](#2-architecture)  
3. [Configuration Layer](#3-configuration-layer)  
4. [Component Responsibilities](#4-component-responsibilities)  
   - 4.1 [InferenceMetadata](#41-inferencemetadata)  
   - 4.2 [RunLoader / Run](#42-runloader--run)  
   - 4.3 [ModelWrapper](#43-modelwrapper)  
   - 4.4 [Predictor / Result](#44-predictor--result)  
   - 4.5 [CubeStitcher](#45-cubestitcher)  
   - 4.6 [Metrics](#46-metrics)  
   - 4.7 [Ploter](#47-ploter)  
   - 4.8 [Animator](#48-animator)  
   - 4.9 [Report / write\_metrics\_json](#49-report--write_metrics_json)  
   - 4.10 [InferencePipeline](#410-inferencepipeline)  
5. [Pipeline Execution Stages](#5-pipeline-execution-stages)  
6. [Mathematical Formulation](#6-mathematical-formulation)  
   - 6.1 [Gaussian Curve Reconstruction](#61-gaussian-curve-reconstruction)  
   - 6.2 [Windowed Overlap-Add Stitching](#62-windowed-overlap-add-stitching)  
   - 6.3 [Per-Pixel and Global Metrics](#63-per-pixel-and-global-metrics)  
   - 6.4 [Per-Elevation Metrics](#64-per-elevation-metrics)  
   - 6.5 [Permutation Matching](#65-permutation-matching)  
   - 6.6 [Gaussian Parameter Metrics](#66-gaussian-parameter-metrics)  
   - 6.7 [Placeholder Detection](#67-placeholder-detection)  
   - 6.8 [Permutation Consensus](#68-permutation-consensus)  
   - 6.9 [$\mu$ Ordering Rate](#69-mu-ordering-rate)  
   - 6.10 [SSIM Computation](#610-ssim-computation)  
   - 6.11 [PSNR](#611-psnr)  
7. [Artifact Naming and Directory Layout](#7-artifact-naming-and-directory-layout)  
8. [Inputs and Outputs Summary](#8-inputs-and-outputs-summary)  
9. [Canonical Usage](#9-canonical-usage)  
10. [Public API Reference](#10-public-api-reference)

---

## 1. Overview

The inference pipeline consumes the artifacts produced by the **training pipeline** (checkpoint, normalisation statistics, configuration manifests) and applies the trained model to a full dataset split. It does not re-train or modify any weights. Its outputs are:

| Output class | Produced by |
|---|---|
| Stitched prediction cubes | `Predictor` + `CubeStitcher` |
| Per-pixel metric maps | `Predictor` (CPU workers) |
| Global scalar metrics | `Metrics.compute()` |
| Publication figures (PNG) | `Ploter` |
| Walk-through GIF animations | `Animator` |
| Self-contained Markdown report | `Report.assemble()` |
| JSON metrics file | `write_metrics_json` |

The pipeline is designed to run with zero GPU-side metric computation. The GPU is used **exclusively** for the forward pass (`ModelWrapper.__call__`). All metric computation is offloaded to CPU worker processes via `ProcessPoolExecutor`.

---

## 2. Architecture

```
InferenceConfig
       │
       ▼
┌────────────────────┐
│ InferencePipeline  │  (pipeline.py)
└────────┬───────────┘
         │
         ├─ _setup
         │      InferenceMetadata  ← resolves all output paths
         │      Logger
         │      Ploter
         │
         ├─ _load_run
         │      RunLoader.load
         │          ├─ read run_summary.json, dataset_creation_config.json
         │          ├─ build DatasetConfiguration
         │          ├─ models.get_model → model.to(device)
         │          ├─ torch.load checkpoint → load_state_dict
         │          ├─ EMA.apply (optional)
         │          ├─ Stats.load → normalization_stats.json
         │          ├─ Cropper.load_split → PatchDataset
         │          └─ DataLoader
         │             → Run (dataclass)
         │
         ├─ _predict
         │      Predictor.run_inference
         │          ├─ _forward_pass      (GPU, DataLoader)
         │          │      ModelWrapper.__call__
         │          │      denormalize_output + clamp_gaussian_params
         │          ├─ _compute_metrics   (CPU ProcessPoolExecutor)
         │          │      _cpu_worker per batch
         │          │          ├─ reconstruct Gaussian curves
         │          │          ├─ µ-sort GT Gaussians
         │          │          └─ MSE / MAE / R² / cosine / peak per pixel
         │          └─ _stitch_results
         │                 CubeStitcher × 4  (curves + params)
         │                 weighted accumulation of pixel metrics
         │                 → Result (dataclass)
         │
         ├─ _evaluate_metrics
         │      Metrics.compute
         │          ├─ global curve MSE / MAE / R² / PSNR
         │          ├─ per-pixel stats + percentiles
         │          ├─ per-elevation MAE / RMSE / R² / cross-entropy
         │          ├─ SSIM (ProcessPoolExecutor, multiple axes)
         │          ├─ per-Gaussian µ/σ MAE/RMSE
         │          ├─ slot µ statistics
         │          ├─ placeholder detection precision/recall/F1
         │          ├─ µ ordering rate
         │          └─ permutation consensus
         │      write_metrics_json → metrics.json
         │
         ├─ _plot_figures
         │      Ploter.plot_*  (30+ figure types)
         │
         ├─ _run_animations
         │      Animator.walk_gif  (per axis)
         │
         └─ _build_report
                Report.assemble → report.md
```

---

## 3. Configuration Layer

All inference behaviour is governed by `InferenceConfig` (`configuration/inference_config.py`).

### `InferenceConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `run_directory` | `Path` | — | Root of the training run to evaluate. |
| `output_subdir` | `Optional[str]` | `None` | Sub-directory name inside `{run_directory}/inference/`; defaults to `YYYYMMDD_HHMMSS`. |
| `device` | `str` | `"cuda"` | PyTorch device for the forward pass. |
| `use_ema` | `bool` | `True` | Apply EMA shadow weights if available in checkpoint. |
| `checkpoint_name` | `str` | `"best_model.pt"` | Filename relative to `run_directory`. |
| `split` | `str` | `"test"` | Dataset split to evaluate (`"train"`, `"val"`, `"test"`). |
| `batch_size` | `Optional[int]` | `None` | Override batch size (defaults to training-run value). |
| `num_workers` | `int` | `4` | DataLoader worker processes. |
| `gif_workers` | `int` | `40` | CPU workers for GIF frame rendering. |
| `cpu_workers` | `int` | `80` | CPU workers for metric computation. |
| `stitch_window` | `str` | `"hann"` | Overlap-add window kind: `"hann"`, `"triangular"`, `"uniform"`. |
| `cube_dtype` | `str` | `"float32"` | NumPy dtype for stitched cube arrays. |
| `save_cubes` | `bool` | `True` | Whether to persist stitched cube `.npy` files. |
| `match_strategy` | `str` | `"auto"` | Gaussian permutation matching strategy for evaluation. |
| `n_best_profiles` | `int` | `12` | Number of best-fit pixel profiles to plot. |
| `n_worst_profiles` | `int` | `12` | Number of worst-fit pixel profiles to plot. |
| `n_random_profiles` | `int` | `12` | Number of random pixel profiles to plot. |
| `n_range_slices` | `int` | `5` | Number of range-direction tomogram cross-sections to plot. |
| `n_azimuth_slices` | `int` | `5` | Number of azimuth-direction cross-sections to plot. |
| `n_elevation_slices` | `int` | `5` | Number of elevation-bin intensity-at-elevation planes to plot. |
| `gif_axes` | `List[str]` | `["elevation"]` | Axes along which to generate walk-through GIFs. |
| `gif_fps` | `int` | `12` | GIF frame rate. |
| `gif_max_frames` | `int` | `150` | Maximum GIF frames (uniformly sub-sampled if exceeded). |
| `cmap_intensity` | `str` | `"jet"` | Colormap for prediction and GT intensity maps. |
| `cmap_error` | `str` | `"magma"` | Colormap for error maps. |
| `amp_zero_thr` | `float` | `1e-3` | Amplitude below which a Gaussian component is considered inactive. |
| `save_dpi` | `int` | `300` | DPI for saved PNG figures. |

### `InferencePaths`

Governs sub-directory names and file names within the output directory.

| Field | Default |
|-------|---------|
| `figures_subdir` | `"figures"` |
| `animations_subdir` | `"animations"` |
| `logs_subdir` | `"logs"` |
| `cubes_subdir` | `"cubes"` |
| `metrics_filename` | `"metrics.json"` |
| `report_filename` | `"report.md"` |

---

## 4. Component Responsibilities

### 4.1 `InferenceMetadata`

**File:** `metadata.py`

Resolves all output paths from `InferenceConfig`. Does not perform any I/O until `create_dirs()` is called.

**Path resolution:**

```
{run_directory}/inference/{output_subdir}/
    figures/       ← all PNG figures
    animations/    ← GIF animations
    logs/          ← inference.log
    cubes/         ← stitched .npy cube files
    metrics.json
    report.md
```

**Key method:** `figure_path(name, ext="png")` returns `figures/{name}.{ext}` as an absolute `Path`.

---

### 4.2 `RunLoader` / `Run`

**File:** `loader.py`

`RunLoader` reconstructs the full inference context from the serialised artifacts of a training run. It is the inference-side counterpart of `TrainingPipeline.run()`.

**`RunLoader.load()` sequence:**

1. Parse `meta/run_summary.json` → model name, channel counts, `n_gaussians`.
2. Parse `meta/dataset_creation_config.json` → reconstruct `DatasetConfiguration` with `shuffle_train=False`.
3. Instantiate and transfer the model to `device` via `models.get_model`.
4. Load the checkpoint with `torch.load(map_location=device)`.
5. Call `model.load_state_dict(ckpt["params"])`.
6. Optionally overwrite trainable parameters with EMA shadow via `_apply_ema`.
7. Set `model.eval()`.
8. Load `meta/normalization_stats.json` → `Stats` object.
9. Reconstruct the `PatchDataset` and `DataLoader` for the requested split (no augmentation, `shuffle=False`, `drop_last=False`).
10. Wrap the model in a `ModelWrapper` (handles `no_grad`, denormalisation, clamping).
11. Return a `Run` dataclass.

**`Run` dataclass fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model` | `ModelWrapper` | Inference-ready wrapped model. |
| `model_name` | `str` | Architecture identifier. |
| `in_channels` | `int` | Number of input channels. |
| `out_channels` | `int` | Number of output channels ($K \times 3$). |
| `x_axis` | `ndarray (L,)` | Elevation axis in physical units. |
| `x_axis_length` | `int` | Number of elevation samples $L$. |
| `n_gaussians` | `int` | Number of Gaussian components $K$. |
| `dataset_config` | `DatasetConfiguration` | Reconstructed dataset configuration. |
| `split_name` | `str` | Evaluated split. |
| `split_region` | `CropRegion` | Global pixel bounding box of the split. |
| `global_crop` | `CropRegion` | Global pixel bounding box of the full pre-processing crop. |
| `grid` | `GridInfo` | Patch grid parameters. |
| `dataset` | `PatchDataset` | PyTorch Dataset for the split. |
| `loader` | `DataLoader` | DataLoader for the split. |
| `checkpoint_meta` | `dict` | `{epoch, best_val_loss, best_epoch}`. |
| `used_ema` | `bool` | Whether EMA weights were applied. |

---

### 4.3 `ModelWrapper`

**File:** `wrapper.py`

A lightweight callable that performs the full inference transform chain under `torch.no_grad()`.

**`__call__(x: ndarray) → ndarray`:**

1. Convert `ndarray` input to `float32` tensor and transfer to device.
2. Forward pass through the wrapped model.
3. `denormalize_output`: apply `Normalizer.denormalize_output` (inverse per-channel transform).
4. `clamp_gaussian_params`: enforce physical constraints — amplitude ≥ 0, $\sigma > 0$, $\mu \in [x_{\min}, x_{\max}]$ — with `leaky_slope=0.0` (hard clamping at inference time).
5. Transfer result to CPU and convert to `ndarray`.

The output is always in **physical (denormalised)** units.

---

### 4.4 `Predictor` / `Result`

**File:** `predictor.py`

`Predictor` orchestrates the three-phase inference execution:

#### Phase 1 — GPU Forward Pass (`_forward_pass`)

Iterates the `DataLoader`. For each batch:
- Calls `run.model(images)` → predicted parameters (physical, denormalised).
- Denormalises ground-truth parameters.
- Accumulates batch tensors in CPU lists.

**No metric computation occurs here.** This phase is designed to keep GPU memory pressure minimal.

#### Phase 2 — CPU Metric Workers (`_compute_metrics`)

Dispatches batches to `_cpu_worker` via `ProcessPoolExecutor`. Each worker:

1. Reconstructs Gaussian curves from predicted and GT parameters.
2. Applies $\mu$-sort to GT Gaussians (sort by $\mu$ of active components, push inactive to last).
3. Computes per-pixel metrics: MSE, MAE, $R^2$, cosine similarity, peak-location error.
4. Returns reconstructed curves, matched parameters, and metric arrays.

#### Phase 3 — Stitching (`_stitch_results`)

Creates four `CubeStitcher` objects (predicted curves, GT curves, predicted parameters, GT parameters). For each patch:
- Adds the patch to all four stitchers using weighted overlap-add.
- Accumulates per-pixel metrics using the same 2-D window weights.

After all patches are processed, calls `finalize_cube()` on each stitcher and normalises pixel metrics by accumulated weights.

**`Result` dataclass:**

| Field | Shape | Description |
|-------|-------|-------------|
| `pred_curves` | `(L, Az, Rg)` | Full stitched predicted PSD cube. |
| `gt_curves` | `(L, Az, Rg)` | Full stitched GT PSD cube. |
| `params_pred` | `(3K, Az, Rg)` | Stitched predicted Gaussian parameters. |
| `params_gt` | `(3K, Az, Rg)` | Stitched GT Gaussian parameters (NaN-masked inactive components). |
| `pixel_mse` | `(Az, Rg)` | Per-pixel MSE (weighted average over patches). |
| `pixel_mae` | `(Az, Rg)` | Per-pixel MAE. |
| `pixel_r2` | `(Az, Rg)` | Per-pixel $R^2$. |
| `pixel_cosine` | `(Az, Rg)` | Per-pixel cosine similarity. |
| `pixel_peak_err_idx` | `(Az, Rg)` | Per-pixel peak-location error in elevation index units. |
| `cube_directory` | `Path` | Output directory for cube `.npy` files. |
| `azimuth_offset` | `int` | Split azimuth start in global coordinates. |
| `range_offset` | `int` | Split range start in global coordinates. |

---

### 4.5 `CubeStitcher`

**File:** `stitching.py`

Implements **windowed overlap-add** reconstruction of a full spatial array from overlapping patches.

**Construction:** Allocates an accumulation buffer of shape `(C, H_pad, W_pad)` and a scalar weight buffer `(H_pad, W_pad)`, where `(H_pad, W_pad)` is the padded spatial size from `GridInfo`.

**`add_patch(idx, patch)`:** Locates the patch at grid position `(iv, ih)`, multiplies the patch by the 2-D window `w`, and accumulates into the buffer:

$$
\mathbf{A}[{:}, v_0:v_0+P_h, h_0:h_0+P_w] \mathrel{+}= w \cdot \text{patch}
$$
$$
\mathbf{W}[v_0:v_0+P_h, h_0:h_0+P_w] \mathrel{+}= w
$$

**`finalize_cube()`:** Normalises by the accumulated weight, removes symmetric padding, and returns the contiguous array:

$$
\hat{\mathbf{C}}[:, i, j] = \frac{\mathbf{A}[:, i + \text{pad\_top}, j + \text{pad\_left}]}{\mathbf{W}[i + \text{pad\_top}, j + \text{pad\_left}]}
$$

**Window functions** (see §6.2 for formulas):

| `window_kind` | Description |
|--------------|-------------|
| `"hann"` | Raised cosine; smooth, minimal boundary artefacts. |
| `"triangular"` | Linear ramp; moderate boundary attenuation. |
| `"uniform"` | No weighting; simple averaging. |

---

### 4.6 `Metrics`

**File:** `metrics.py`

Computes all scalar evaluation metrics from a `Result` object. The main entry point is `Metrics.compute()`, which orchestrates all sub-metrics and returns a flat `dict[str, float]`.

**Metric groups computed:**

| Group | Method | Description |
|-------|--------|-------------|
| Global curve | `compute()` direct | MSE, MAE, RMSE, $R^2$, PSNR over the full cube. |
| Per-pixel stats | `_basic_stats` + `_percentiles` | Mean, std, median, min, max, $P_1$ through $P_{99}$ for pixel MSE / MAE / $R^2$ / cosine / peak error. |
| Per-elevation | `_elev_metrics` | MAE, RMSE, $R^2$, cross-entropy at each elevation index. |
| SSIM slices | `_slice_ssim` | SSIM over sampled elevation, range, and azimuth cross-sections (parallel). |
| Per-Gaussian param | `_gaussian_param_metrics` | Per-slot and pooled MAE/RMSE for $\mu$ and $\sigma$. |
| Slot $\mu$ stats | `_slot_mu_stats` | Mean and std of $\mu$ per slot (active pixels only). |
| Placeholder detection | `_placeholder_detection` | Precision, recall, F1 for inactive-component detection per slot and globally. |
| $\mu$ ordering rate | `_mu_ordering_rate` | Fraction of multi-component pixels where predicted $\mu$s are sorted ascending. |
| Permutation consensus | `_permutation_consensus` | Dominant and identity permutation fractions across all pixels. |

**`Metrics.select_pixels`:** A utility that, given a 2-D scalar metric map, returns the indices of the `n_best`, `n_worst`, and `n_random` pixels by metric value. Used for profile plot selection.

---

### 4.7 `Ploter`

**File:** `plots.py`

Generates all static figures. All output is in Agg (non-interactive) Matplotlib backend at 300 DPI. Scientific RC parameters are applied globally.

**Figure inventory (partial):**

| Method | Output |
|--------|--------|
| `plot_profile_panel` | Grid of $N$ pixel-level PSD profiles: GT curve, predicted curve, Gaussian components. |
| `plot_pixel_metric_map` | Spatial 2-D map of a per-pixel metric (MSE, $R^2$, peak error). |
| `plot_metric_histogram` | Histogram panels for pixel MSE, $R^2$, and cosine similarity. |
| `plot_param_maps` | Spatial maps of $a$, $\mu$, $\sigma$ for each Gaussian slot (pred vs GT). |
| `plot_param_distributions` | Histogram panels for each parameter role per slot. |
| `plot_param_scatter` | Scatter plot of predicted vs GT for each parameter role. |
| `plot_param_error_maps` | Spatial maps of $|\hat{p} - p^*|$ per parameter. |
| `plot_tomogram_slice` | SAR tomogram cross-section (GT vs pred) along range or azimuth axis. |
| `plot_elevation_intensity_slice` | Intensity-at-elevation-bin plane (GT vs pred). |
| `plot_ssim_curves` | SSIM value vs. slice index for all evaluated cross-sections. |
| `plot_elev_metric_curves` | MAE, RMSE, $R^2$, cross-entropy as a function of elevation index. |
| `plot_slot_mu_distributions` | Distribution of $\mu$ per Gaussian slot. |
| `plot_placeholder_detection` | Per-slot precision/recall bar charts. |
| `plot_slot_ordering_summary` | Summary of $\mu$-ordering rate and permutation consensus. |
| `plot_active_count_map` | Spatial map of number of active Gaussian components per pixel. |

---

### 4.8 `Animator`

**File:** `animation.py`

Produces walk-through GIF animations of the tomographic cube.

**`walk_gif(pred_cube, gt_cube, axis, out_path, ...)`:**

1. Builds axis-specific frame parameters (extent, labels, title function) via `_build_axis`.
2. Sub-samples up to `max_frames` uniformly-spaced frame indices.
3. Computes shared colour limits using the $P_{0.1}$ and $P_{99.9}$ percentiles over a 16-frame sample.
4. Dispatches frame rendering to `_render_frame` workers via `ProcessPoolExecutor`.
5. Each frame contains three panels: GT, prediction, and $|\text{pred} - \text{GT}|$.
6. Assembles frames into a GIF with Pillow with `duration_ms = 1000 / fps`.

**Supported axes:**

| `axis` | Walk direction | Panel spatial axes |
|--------|---------------|-------------------|
| `"elevation"` | Sweep through elevation bins | Range (x) × Azimuth (y) |
| `"range"` | Sweep through range pixels | Azimuth (x) × Elevation (y) |
| `"azimuth"` | Sweep through azimuth pixels | Range (x) × Elevation (y) |

For `range` and `azimuth` axes, elevation is sorted by index before display.

---

### 4.9 `Report` / `write_metrics_json`

**File:** `report.py`

`Report.assemble()` produces a structured Markdown document with the following sections:

| Section | Content |
|---------|---------|
| 1. Run summary | Model name/channels, dataset split/region, patch grid, checkpoint epoch/loss, inference configuration. |
| 2. Headline metrics | Global curve MSE/MAE/RMSE/$R^2$/PSNR; per-pixel statistics; SSIM means. |
| 3. Per-Gaussian metrics | Per-slot $\mu$/σ MAE/RMSE, placeholder precision/recall/F1. |
| 4. Qualitative | Embedded figure references for all PNG outputs. |
| 5. Animations | Embedded GIF references. |
| Extra sections | Optional user-supplied Markdown blocks. |

`write_metrics_json` serialises the full `global_metrics` dict to `metrics.json` with `json.dump(default=str)`.

---

### 4.10 `InferencePipeline`

**File:** `pipeline.py`

The single top-level orchestrator. Has no state beyond the `InferenceConfig` and delegates all work to sub-components.

**`run()` return value:** `Path` to the generated Markdown report.

---

## 5. Pipeline Execution Stages

### Stage 0 — Setup

```
InferencePipeline.run()
    _setup(cfg)
        InferenceMetadata(cfg)   ← resolve all paths
        meta.create_dirs()       ← mkdir -p all output directories
        np.random.seed(cfg.seed)
        Logger(logs_dir)
        Ploter(cmaps, dpi)
```

### Stage 1 — Run Loading

```
    _load_run(cfg, logger)
        RunLoader(run_directory, logger)
        RunLoader.load(split, batch_size, num_workers, device, use_ema, checkpoint_name)
            read run_summary.json
            read dataset_creation_config.json → DatasetConfiguration
            models.get_model → model.to(device)
            torch.load checkpoint → load_state_dict
            [optional] _apply_ema(model, ckpt)
            model.eval()
            Stats.load → normalization_stats.json
            GaussianConfig.from_dataset
            ModelWrapper(model, device, normalizer, x_axis, amp_max)
            Cropper.load_split(split_region) → arrays
            Patcher.build → grid
            PatchDataset(inputs, gt_params, grid, ...)
            DataLoader(dataset, shuffle=False, drop_last=False)
            → Run (dataclass)
```

### Stage 2 — Prediction and Stitching

```
    _predict(cfg, meta, run, logger)
        Predictor(run, logger, window_kind, cube_dtype, save_cubes, meta, cpu_workers)
        Predictor.run_inference()
            _forward_pass
                for batch in DataLoader:
                    ModelWrapper(images)  ← GPU, no_grad
                    denormalize GT params
                    accumulate pred_params, gt_params, indices
            _compute_metrics (ProcessPoolExecutor)
                _cpu_worker per batch:
                    reconstruct pred_curves, gt_curves
                    µ-sort GT Gaussians
                    compute MSE / MAE / R² / cosine / peak per pixel
            _stitch_results
                CubeStitcher × 4 (curves × 2, params × 2)
                overlap-add with 2-D Hann window
                normalise pixel metrics by accumulated window weights
                [optional] save .npy cubes
                → Result (dataclass)
```

### Stage 3 — Metric Computation

```
    _evaluate_metrics(result, x_axis_np, run, meta, indices)
        Metrics(result, x_axis, n_gaussians).compute(
            elev_indices, range_indices, az_indices)
                global MSE / MAE / R² / PSNR
                per-pixel stats + percentiles
                per-elevation MAE / RMSE / R² / CE
                SSIM (ProcessPoolExecutor, parallel slices)
                per-Gaussian µ/σ errors
                slot µ statistics
                placeholder precision / recall / F1
                µ ordering rate
                permutation consensus
        write_metrics_json → metrics.json
```

### Stage 4 — Figure Generation

```
    _plot_figures(plotter, result, run, meta, ...)
        Metrics.select_pixels(pixel_mse, n_best, n_worst, n_random)
        plotter.plot_profile_panel ×3 (best, worst, random)
        plotter.plot_pixel_metric_map ×3 (MSE, R², peak)
        plotter.plot_metric_histogram
        plotter.plot_param_maps
        plotter.plot_param_distributions
        plotter.plot_param_scatter
        plotter.plot_param_error_maps
        plotter.plot_slot_mu_distributions
        plotter.plot_placeholder_detection
        plotter.plot_slot_ordering_summary
        plotter.plot_active_count_map
        plotter.plot_tomogram_slice  ×(n_range + n_azimuth) slices
        plotter.plot_elevation_intensity_slice  ×n_elevation slices
        plotter.plot_ssim_curves ×3 (range, azimuth, elev)
        plotter.plot_elev_metric_curves
```

### Stage 5 — Animation

```
    _run_animations(result, meta, x_axis_np, cfg, logger)
        for axis in cfg.gif_axes:
            Animator.walk_gif(pred_cube, gt_cube, axis, out_path, ...)
                _build_axis → frame spec
                sub-sample to max_frames
                compute shared clim
                ProcessPoolExecutor → _render_frame per frame
                Pillow: assemble and save GIF
```

### Stage 6 — Report Assembly and Teardown

```
    _build_report(meta, run, cfg, x_axis_np, global_metrics, ...)
        Report(output_dir, run_summary, inference_config, ...).assemble()
            → report.md
    logger.close()
    return report_path
```

---

## 6. Mathematical Formulation

### 6.1 Gaussian Curve Reconstruction

Given predicted parameter tensor $\hat{\mathbf{P}} \in \mathbb{R}^{3K \times Az \times Rg}$ (in physical units) and the elevation axis $\mathbf{x} \in \mathbb{R}^L$, the predicted power spectral density at position $(az, rg)$ is:

$$
\hat{S}(x;\, az, rg) = \sum_{k=1}^{K} \hat{a}_k \exp\!\left(-\frac{(x - \hat{\mu}_k)^2}{2\hat{\sigma}_k^2}\right)
$$

yielding $\hat{\mathbf{S}} \in \mathbb{R}^{L \times Az \times Rg}$.

The same formula is applied to ground-truth parameters to obtain $\mathbf{S}^* \in \mathbb{R}^{L \times Az \times Rg}$.

Physical parameter constraints enforced by `clamp_gaussian_params`:

- $\hat{a}_k \geq 0$
- $\hat{\sigma}_k > \epsilon$ (numerically safe)
- $\hat{\mu}_k \in [x_{\min}, x_{\max}]$

### 6.2 Windowed Overlap-Add Stitching

Let $(P_h, P_w)$ be the patch size and $s$ the stride. Define the 2-D window for patch $(i_v, i_h)$:

**Hann window:**

$$
w^{(v)}_i = \frac{1}{2}\left(1 - \cos\!\frac{2\pi(i + 0.5)}{P_h}\right), \quad i = 0, \ldots, P_h - 1
$$

$$
w^{(h)}_j = \frac{1}{2}\left(1 - \cos\!\frac{2\pi(j + 0.5)}{P_w}\right), \quad j = 0, \ldots, P_w - 1
$$

$$
w_{ij} = w^{(v)}_i \cdot w^{(h)}_j \geq 10^{-3}
$$

**Triangular window:**

$$
w^{(v)}_i = 1 - \left|\frac{2(i + 0.5)}{P_h} - 1\right|
$$

The accumulated cube at position $(c, p, q)$ in the padded buffer is:

$$
\mathbf{A}[c, p, q] = \sum_{\text{patches covering } (p, q)} w_{p - v_0, q - h_0} \cdot \text{patch}[c, p - v_0, q - h_0]
$$

$$
\mathbf{W}[p, q] = \sum_{\text{patches covering } (p, q)} w_{p - v_0, q - h_0}
$$

The final normalised cube (after removing padding) is:

$$
\hat{\mathbf{C}}[c, i, j] = \frac{\mathbf{A}[c,\; i + \text{pad\_top},\; j + \text{pad\_left}]}{\mathbf{W}[i + \text{pad\_top},\; j + \text{pad\_left}]}
$$

### 6.3 Per-Pixel and Global Metrics

Let $\hat{s}_{l,h,w} = \hat{\mathbf{S}}[l, h, w]$ and $s^*_{l,h,w} = \mathbf{S}^*[l, h, w]$ with $l \in \{1, \ldots, L\}$, $h \in \{1, \ldots, Az\}$, $w \in \{1, \ldots, Rg\}$.

**Global curve MSE:**

$$
\text{MSE}_{\text{global}} = \frac{1}{L \cdot Az \cdot Rg} \sum_{l,h,w} (\hat{s}_{l,h,w} - s^*_{l,h,w})^2
$$

**Global $R^2$ (coefficient of determination):**

$$
R^2 = 1 - \frac{\sum_{l,h,w} (\hat{s}_{l,h,w} - s^*_{l,h,w})^2}{\sum_{l,h,w} (s^*_{l,h,w} - \bar{s}^*)^2 + \epsilon}
$$

**Per-pixel MSE** (for pixel $(h, w)$):

$$
\text{MSE}_{h,w} = \frac{1}{L} \sum_l (\hat{s}_{l,h,w} - s^*_{l,h,w})^2
$$

**Per-pixel $R^2$:**

$$
R^2_{h,w} = 1 - \frac{\sum_l (\hat{s}_{l,h,w} - s^*_{l,h,w})^2}{\sum_l (s^*_{l,h,w} - \bar{s}^*_{h,w})^2 + \epsilon}
$$

**Per-pixel cosine similarity:**

$$
\cos_{h,w} = \frac{\hat{\mathbf{s}}_{h,w} \cdot \mathbf{s}^*_{h,w}}{\|\hat{\mathbf{s}}_{h,w}\|_2 \cdot \|\mathbf{s}^*_{h,w}\|_2 + \epsilon}
$$

**Per-pixel peak-location error (in elevation index units):**

$$
\text{peak}_{h,w} = \left|\operatorname{argmax}_l \hat{s}_{l,h,w} - \operatorname{argmax}_l s^*_{l,h,w}\right|
$$

Converted to physical units by multiplying by $\Delta x = x_2 - x_1$.

The **weighted pixel metric accumulation** during stitching is:

$$
\widetilde{M}_{h,w} = \frac{\sum_{\text{patches}} w_{h,w}^{\text{patch}} \cdot M_{h,w}^{\text{patch}}}{\sum_{\text{patches}} w_{h,w}^{\text{patch}}}
$$

where $w_{h,w}^{\text{patch}}$ is the window weight at position $(h, w)$ within the patch.

### 6.4 Per-Elevation Metrics

For each elevation index $l \in \{1, \ldots, L\}$, treating the prediction and GT as vectors over all $(az, rg)$ pixels:

**MAE at elevation $l$:**

$$
\text{MAE}_l = \frac{1}{Az \cdot Rg} \sum_{h,w} |\hat{s}_{l,h,w} - s^*_{l,h,w}|
$$

**RMSE at elevation $l$:**

$$
\text{RMSE}_l = \sqrt{\frac{1}{Az \cdot Rg} \sum_{h,w} (\hat{s}_{l,h,w} - s^*_{l,h,w})^2}
$$

**$R^2$ at elevation $l$:**

$$
R^2_l = 1 - \frac{\sum_{h,w}(\hat{s}_{l,h,w} - s^*_{l,h,w})^2}{\sum_{h,w}(s^*_{l,h,w} - \bar{s}^*_l)^2 + \epsilon}
$$

**Column-normalised cross-entropy at elevation $l$:**

Let $p^*_{l,n} = s^*_{l,n} / \sum_{l'} s^*_{l',n}$ and $\hat{p}_{l,n} = \hat{s}_{l,n} / \sum_{l'} \hat{s}_{l',n}$ be column-normalised (per-pixel) probability distributions, where $n$ indexes the $(h, w)$ pixel.

$$
\text{CE}_l = -\frac{1}{Az \cdot Rg} \sum_n p^*_{l,n} \log\!\max(\hat{p}_{l,n}, 10^{-12})
$$

### 6.5 Permutation Matching

Before computing parameter-space metrics in the CPU worker, GT Gaussian components are sorted by their $\mu$ value to align with the model's implied slot ordering:

$$
\pi^* = \operatorname{argsort}\!\left[\mu^*_1, \ldots, \mu^*_K\right]_{\text{stable}}, \quad \text{with inactive } (a^* < 10^{-3}) \text{ components sent to last}
$$

The predicted parameters are left in their original slot order. This is equivalent to the sorted-$\mu$ matching strategy, which is the default for evaluation.

### 6.6 Gaussian Parameter Metrics

For slot $k$, restrict to pixels where the GT amplitude $a^*_k \geq 10^{-3}$ (active set $\mathcal{A}_k$):

$$
\text{MAE}_{\mu,k} = \frac{1}{|\mathcal{A}_k|} \sum_{(h,w) \in \mathcal{A}_k} |\hat{\mu}_k(h,w) - \mu^*_k(h,w)|
$$

$$
\text{RMSE}_{\mu,k} = \sqrt{\frac{1}{|\mathcal{A}_k|} \sum_{(h,w) \in \mathcal{A}_k} (\hat{\mu}_k(h,w) - \mu^*_k(h,w))^2}
$$

Analogously for $\sigma_k$. Pooled versions aggregate over all $K$ slots.

### 6.7 Placeholder Detection

A Gaussian component is declared *inactive* (placeholder) at pixel $(h,w)$ if its amplitude $a_k(h,w) < 10^{-3}$. Let $\hat{b}_{k,n}$ and $b^*_{k,n}$ be binary inactive indicators for prediction and GT respectively, over all valid pixels $n$.

$$
\text{Precision}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k + \epsilon}, \quad \text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k + \epsilon}
$$

$$
F1_k = \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k + \epsilon}
$$

where $\text{TP}_k = \sum_n \hat{b}_{k,n} b^*_{k,n}$, $\text{FP}_k = \sum_n \hat{b}_{k,n}(1 - b^*_{k,n})$, $\text{FN}_k = \sum_n (1 - \hat{b}_{k,n}) b^*_{k,n}$.

Global values aggregate over all slots.

### 6.8 Permutation Consensus

For each pixel, the optimal predicted-to-GT Gaussian permutation is identified by minimising the total $\mu$-distance cost:

$$
\pi^*_n = \operatorname{argmin}_{\pi \in \mathcal{S}_K} \sum_{k=1}^{K} |\hat{\mu}_k(n) - \mu^*_{\pi(k)}(n)|
$$

where $\mathcal{S}_K$ is the set of all $K!$ permutations.

**Dominant fraction:** The fraction of pixels for which the most-common permutation is optimal.

$$
f_{\text{dominant}} = \frac{\max_\pi \#\{n : \pi^*_n = \pi\}}{N_{\text{pixels}}}
$$

**Identity fraction:** The fraction of pixels for which the identity permutation (slot $k$ matched to GT slot $k$) is optimal.

$$
f_{\text{identity}} = \frac{\#\{n : \pi^*_n = \text{id}\}}{N_{\text{pixels}}}
$$

A high $f_{\text{identity}}$ indicates that the model has learnt globally consistent slot roles, typically via $\mu$-sorted Gaussian labelling.

### 6.9 $\mu$ Ordering Rate

For pixels with at least two active Gaussian components, the $\mu$ ordering rate measures the fraction where the predicted $\mu$ values respect ascending order:

$$
\text{ordering\_rate} = \frac{\#\left\{(h,w) : \hat{\mu}_k(h,w) < \hat{\mu}_{k+1}(h,w)\; \forall k \in \text{active pairs}\right\}}{|\{(h,w) : n_{\text{active}} \geq 2\}|}
$$

Values close to 1 indicate that the model consistently uses $\mu$-ordered slot assignments.

### 6.10 SSIM Computation

SSIM is computed on 2-D cross-sectional slices using `skimage.metrics.structural_similarity` with adaptive window size:

$$
\text{win\_size} = \min\!\left(7, \text{odd}(\min(H_{\text{slice}}, W_{\text{slice}}))\right)
$$

$$
\text{SSIM} = \frac{(2\mu_{\hat{s}}\mu_{s^*} + C_1)(2\sigma_{\hat{s}s^*} + C_2)}{(\mu_{\hat{s}}^2 + \mu_{s^*}^2 + C_1)(\sigma_{\hat{s}}^2 + \sigma_{s^*}^2 + C_2)}
$$

with data range $= \max(s^*) - \min(s^*)$. SSIM values are computed in parallel via `ProcessPoolExecutor` and averaged over sampled slices for each axis.

### 6.11 PSNR

$$
\text{PSNR} = 10 \log_{10}\!\frac{(\max(s^*) - \min(s^*))^2}{\text{MSE}_{\text{global}}}
$$

Returns `inf` when $\text{MSE} = 0$ and `nan` when the data range is zero.

---

## 7. Artifact Naming and Directory Layout

### Training Run (input)

```
{run_directory}/
    best_model.pt                  ← checkpoint (weights, EMA shadow, state)
    meta/
        run_summary.json           ← model name, channel counts
        dataset_creation_config.json
        normalization_stats.json
        crop.json
        patch.json
```

### Inference Run (output)

```
{run_directory}/inference/{output_subdir}/
    metrics.json                   ← all scalar metrics (flat dict)
    report.md                      ← self-contained Markdown report
    logs/
        inference.log
    cubes/
        pred_curves.npy            ← (L, Az, Rg) float32
        gt_curves.npy              ← (L, Az, Rg) float32
        params_pred.npy            ← (3K, Az, Rg) float32
        params_gt.npy              ← (3K, Az, Rg) float32
        pixel_mse.npy              ← (Az, Rg) float32
        pixel_mae.npy              ← (Az, Rg) float32
        pixel_r2.npy               ← (Az, Rg) float32
        pixel_cos.npy              ← (Az, Rg) float32
        pixel_peak.npy             ← (Az, Rg) int32
    figures/
        profiles_best.png
        profiles_worst.png
        profiles_random.png
        pixel_mse_map.png
        pixel_r2_map.png
        pixel_peak_map.png
        metric_histograms.png
        param_maps.png
        param_distributions.png
        param_scatter.png
        param_error_maps.png
        slot_mu_distributions.png
        placeholder_detection.png
        slot_ordering_summary.png
        active_count_map.png
        slice_range_{i}.png        ×n_range_slices
        slice_azimuth_{i}.png      ×n_azimuth_slices
        slice_elev_idx_{i}.png     ×n_elevation_slices
        ssim_range.png
        ssim_azimuth.png
        ssim_elev.png
        elev_metric_curves.png
    animations/
        walk_{axis}.gif            ×len(gif_axes)
```

---

## 8. Inputs and Outputs Summary

### Inputs

| Artifact | Source | Description |
|----------|--------|-------------|
| `best_model.pt` | Training run root | PyTorch checkpoint with `params`, `ema_shadow`, `x_axis`, `config`. |
| `meta/run_summary.json` | Training run | Model name, channel counts, x-axis length. |
| `meta/dataset_creation_config.json` | Training run | Full `DatasetConfiguration` for split/patch reconstruction. |
| `meta/normalization_stats.json` | Training run | Per-channel `(loc, scale)` for denormalisation. |
| Pre-processed `.npy` arrays | Processing run | `primary_reduced`, `secondaries_reduced`, `interferograms_reduced`, `parameters`. |
| `InferenceConfig` | Caller | All inference hyperparameters. |

### Outputs

| Artifact | Shape / Type | Description |
|----------|-------------|-------------|
| `pred_curves.npy` | `(L, Az, Rg)` | Stitched predicted PSD cube in physical units. |
| `gt_curves.npy` | `(L, Az, Rg)` | Stitched GT PSD cube in physical units. |
| `params_pred.npy` | `(3K, Az, Rg)` | Stitched predicted Gaussian parameters. |
| `params_gt.npy` | `(3K, Az, Rg)` | Stitched GT parameters (NaN for inactive components). |
| `pixel_*.npy` | `(Az, Rg)` | Per-pixel metric maps (MSE, MAE, $R^2$, cosine, peak error). |
| `metrics.json` | flat dict | All scalar metrics (≈100+ keys). |
| `report.md` | Markdown | Self-contained report with embedded figure links. |
| `figures/*.png` | PNG at 300 DPI | Full figure set (≈25+ files). |
| `animations/walk_*.gif` | GIF | Walk-through animations per configured axis. |
| Return value | `Path` | Path to `report.md`. |

---

## 9. Canonical Usage

```python
from pathlib import Path
from configuration.inference_config import InferenceConfig
from pipelines.inference_pipeline.pipeline import InferencePipeline

cfg = InferenceConfig(
    run_directory      = Path("/runs/train/unet_K2_run01"),
    split              = "test",
    device             = "cuda",
    use_ema            = True,
    checkpoint_name    = "best_model.pt",
    stitch_window      = "hann",
    save_cubes         = True,
    n_best_profiles    = 12,
    n_worst_profiles   = 12,
    n_random_profiles  = 12,
    n_range_slices     = 5,
    n_azimuth_slices   = 5,
    n_elevation_slices = 5,
    gif_axes           = ["elevation", "range"],
    gif_fps            = 12,
    save_dpi           = 300,
)

report_path = InferencePipeline(cfg).run()
print(f"Report written to: {report_path}")
```

**Batch inference over all runs in a directory:**

```python
from pathlib import Path

base_logdir = Path("/runs/train")

for run_dir in sorted(base_logdir.iterdir()):
    if not (run_dir / "best_model.pt").exists():
        continue
    cfg = InferenceConfig(run_directory=run_dir, split="test")
    InferencePipeline(cfg).run()
```

**Loading stitched cubes for downstream analysis:**

```python
import numpy as np
from pathlib import Path

output_dir = Path("/runs/train/unet_K2_run01/inference/20260601_120000/cubes")

pred_curves = np.load(output_dir / "pred_curves.npy")  # (L, Az, Rg)
gt_curves   = np.load(output_dir / "gt_curves.npy")
params_pred = np.load(output_dir / "params_pred.npy")  # (3K, Az, Rg)
pixel_r2    = np.load(output_dir / "pixel_r2.npy")     # (Az, Rg)
```

---

## 10. Public API Reference

### `InferencePipeline` (`pipeline.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(config: InferenceConfig)` | Stores config; no I/O. |
| `run` | `() → Path` | Full pipeline execution; returns path to report. |

### `RunLoader` (`loader.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(run_directory, logger)` | — |
| `load` | `(split, batch_size, num_workers, device, use_ema, checkpoint_name) → Run` | Reconstruct full inference context from training artifacts. |

### `ModelWrapper` (`wrapper.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(x: ndarray) → ndarray` | GPU forward + denormalise + clamp; returns physical-unit parameters. |
| `denormalize_output` | `(out: Tensor) → Tensor` | Inverse normalisation + clamping (PyTorch). |

### `Predictor` (`predictor.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `run_inference` | `() → Result` | Full three-phase execution: forward → metrics → stitch. |

### `CubeStitcher` (`stitching.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_patch` | `(idx, patch: ndarray)` | Accumulate one patch with window weighting. |
| `add_patch_batch` | `(indices, batch_patches)` | Vectorised multi-patch accumulation. |
| `finalize_cube` | `() → ndarray` | Normalise, de-pad, and return contiguous cube. |
| `make_patch_window` | `(patch_size, kind) → ndarray` | Static; construct 2-D window function. |

### `Metrics` (`metrics.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `compute` | `(elev_indices, range_indices, az_indices) → dict` | All metrics; returns flat `dict[str, float]`. |
| `select_pixels` | `(metric_map, n_best, n_worst, n_random, seed) → dict` | Static; select pixel indices by metric value. |

### `Animator` (`animation.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `walk_gif` | `(pred_cube, gt_cube, axis, out_path, *, x_axis, az_offset, rg_offset) → Path` | Render and save walk-through GIF. |

### `Report` (`report.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `assemble` | `() → Path` | Build and write Markdown report. |

### `write_metrics_json` (`report.py`)

| Function | Signature | Description |
|----------|-----------|-------------|
| `write_metrics_json` | `(metrics: dict, path: Path) → Path` | Serialise metrics dict to JSON. |

### `InferenceMetadata` (`metadata.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `create_dirs` | `()` | Create all output sub-directories. |
| `figure_path` | `(name, ext="png") → Path` | Resolve figure output path. |
