# Parameter Extraction Pipeline — Technical Reference

**Package:** `pipelines.param_pipeline`  
**Entry points:** `main/extract_params.py`, `scripts/extract_params_sweep.py`  
**Last updated:** June 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Configuration Layer](#3-configuration-layer)
   - 3.1 [FitMode.SigmaOnly](#31-fitmodesigmaonly)
   - 3.2 [FitSettings](#32-fitsettings)
   - 3.3 [ExtractionConfig](#33-extractionconfig)
4. [Component Responsibilities](#4-component-responsibilities)
   - 4.1 [ParameterExtractor](#41-parameterextractor)
   - 4.2 [SigmaFittingExtractor](#42-sigmafittingextractor)
   - 4.3 [FittingMetricsCalculator](#43-fittingmetricscalculator)
   - 4.4 [FittingResultPlotter](#44-fittingresultplotter)
   - 4.5 [ExtractionMetadataManager](#45-extractionmetadatamanager)
   - 4.6 [ParamExtractionPipeline](#46-paramextractionpipeline)
5. [Pipeline Execution Stages](#5-pipeline-execution-stages)
   - 5.1 [Stage 1 — Parameter Extraction](#51-stage-1--parameter-extraction)
   - 5.2 [Stage 2 — Parameter Array Persistence](#52-stage-2--parameter-array-persistence)
   - 5.3 [Stage 3 — Fitting Quality Metrics](#53-stage-3--fitting-quality-metrics)
   - 5.4 [Stage 4 — Visualisation](#54-stage-4--visualisation)
6. [Mathematical Formulation](#6-mathematical-formulation)
   - 6.1 [Tomographic Profile Model](#61-tomographic-profile-model)
   - 6.2 [Sigma-Only Fitting Problem](#62-sigma-only-fitting-problem)
   - 6.3 [Adam Optimiser](#63-adam-optimiser)
   - 6.4 [Complexity-Penalised Model Selection](#64-complexity-penalised-model-selection)
   - 6.5 [Coefficient of Determination](#65-coefficient-of-determination)
7. [GPU Execution and Parallelism Strategy](#7-gpu-execution-and-parallelism-strategy)
   - 7.1 [CPU Initialisation Phase](#71-cpu-initialisation-phase)
   - 7.2 [GPU Sigma Fitting Phase](#72-gpu-sigma-fitting-phase)
   - 7.3 [Best-K Selection Phase](#73-best-k-selection-phase)
   - 7.4 [Multi-GPU with pmap](#74-multi-gpu-with-pmap)
8. [Artifact Naming and Directory Layout](#8-artifact-naming-and-directory-layout)
   - 8.1 [Output Files](#81-output-files)
   - 8.2 [Directory Structure](#82-directory-structure)
9. [Inputs and Outputs Summary](#9-inputs-and-outputs-summary)
10. [Canonical Usage — Entry Points](#10-canonical-usage--entry-points)
11. [Public API Reference](#11-public-api-reference)

---

## 1. Overview

The `param_pipeline` package implements the **parameter extraction stage** of the DLR-TomoSAR system. Its purpose is to decompose the per-pixel tomographic power spectral density (PSD) profiles produced by the `processing_pipeline` into a compact, parametric representation consisting of a sum of Gaussian components.

Each pixel's elevation profile $p(h)$ is approximated by $K^* \in \{1, \ldots, K_{\max}\}$ Gaussian functions, where $K^*$ is selected automatically per pixel via a complexity-penalised scoring criterion. The three parameters of each Gaussian — **amplitude** $A_k$, **centroid height** $\mu_k$, and **elevation spread** $\sigma_k$ — constitute the physical observables of interest: $\mu_k$ encodes ground or canopy layer elevation; $\sigma_k$ encodes the vertical extent of a scattering layer; $A_k$ encodes relative backscatter intensity.

The fitting engine is accelerated by **JAX** and exploits GPU hardware through either `jit` (single-device) or `pmap` (multi-device) compilation. Amplitude and centroid parameters are initialised on CPU via a peak-prominence detector; only the spread parameters $\{\sigma_k\}$ are optimised on GPU via the Adam algorithm, which constitutes the core computational kernel (`SigmaAdamKernel`).

Following extraction, the pipeline computes fitting quality metrics ($R^2$ maps, active-Gaussian count distributions, per-Gaussian parameter statistics) and generates a comprehensive set of scientific figures.

---

## 2. Architecture

```
main/extract_params.py  |  scripts/extract_params_sweep.py
│
│   ExtractionConfig
│   (processed_data_path, fit_settings, adam params,
│    gpu config, output paths)
│
└── ParamExtractionPipeline
      │
      ├── ParameterExtractor
      │     └── SigmaFittingExtractor
      │           ├── _prepare_data()         → load tomogram mmap, build height axis
      │           ├── _warmup_kernel()        → JIT/pmap compilation (dummy run)
      │           ├── _run_fitting()
      │           │     ├── _load_batch()     → range-bin slice, normalisation
      │           │     └── _fit_batch()
      │           │           ├── Phase 1: _prominence_batch()  [CPU, multiprocess]
      │           │           │     └── _prominence_worker()    [per chunk]
      │           │           ├── Phase 2: SigmaAdamKernel / PmapSigmaAdamKernel [GPU]
      │           │           │     └── _per_pixel_loss()  → Adam scan (JAX lax.scan)
      │           │           └── Phase 3: penalty scoring → best-K selection
      │           └── _estimate_r2()          → quick R² diagnostic (sampled)
      │
      ├── ExtractionMetadataManager
      │     └── save_run_metadata()           → param_extraction_meta.json
      │
      ├── FittingMetricsCalculator
      │     ├── _reconstruct_profiles()       → full Gaussian sum
      │     ├── _compute_r2_map()             → per-pixel R²
      │     ├── _compute_activity_map()       → count of active Gaussians
      │     ├── _compute_per_gaussian_maps()  → A_k, μ_k, σ_k spatial maps
      │     └── _compute_mu_separation_maps() → |μ_{k+1} − μ_k| maps
      │
      └── FittingResultPlotter
            ├── colormaps/                    → n_gaussians_map, r2_map,
            │                                    amplitude_maps, mu_maps,
            │                                    sigma_maps, mu_separation_maps
            ├── distributions/               → r2_distribution, amp/mu/sigma violin plots
            ├── metrics/                     → global_summary bar chart
            └── example_fits/                → tier_low, tier_mid, tier_high

configuration/param_extraction_config.py
   ├── FitMode.SigmaOnly  (fit hyperparameters)
   ├── FitSettings        (number_of_gaussians, max_fit_iterations, fit_config)
   └── ExtractionConfig   (root config: paths, GPU settings, Adam params)
```

The `ParamExtractionPipeline` class is the sole orchestrator. It instantiates `ParameterExtractor`, `ExtractionMetadataManager`, `FittingMetricsCalculator`, and `FittingResultPlotter` once at construction time, then sequences the four processing stages inside `run()`. Each stage writes its result to disk and the subsequent stage reloads it, keeping peak memory usage bounded regardless of scene size.

---

## 3. Configuration Layer

**Module:** `configuration/param_extraction_config.py`

All pipeline behaviour is governed by a hierarchy of dataclasses. No mutable global state exists outside these objects.

### 3.1 FitMode.SigmaOnly

Dataclass encoding all hyperparameters specific to the sigma-only Adam fitting strategy:

| Field | Type | Default | Description |
|---|---|---|---|
| `threshold_factor` | `float` | `0.25` | Minimum fractional amplitude threshold relative to the per-pixel profile maximum; bins below this are zeroed before fitting |
| `truncation_index` | `int` | `170` | Height-axis index beyond which the profile is zeroed (eliminates far-range artefacts in the tomogram) |
| `k_max` | `int` | `5` | Maximum number of Gaussian components considered per pixel |
| `lambda_k` | `float` | `3e-3` | Complexity penalty coefficient in the model-selection criterion |
| `prominence_frac` | `float` | `0.05` | Minimum peak prominence as a fraction of the profile maximum, used in the CPU peak-detection initialisation |

---

### 3.2 FitSettings

Wraps a `FitMode` instance with additional bookkeeping fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `number_of_gaussians` | `int` | `3` | Nominal number of Gaussians (used in output naming and metric reporting) |
| `max_fit_iterations` | `int` | `5000` | Passed to the optimiser as an upper bound on Adam steps |
| `fit_config` | `FitConfig` | `FitMode.SigmaOnly()` | Nested fit hyperparameter dataclass |

**Derived properties:**

- `parameters_per_profile → int` — equals `3 × number_of_gaussians`
- `fitting_method → str` — returns `"sigma_only_adam"`

---

### 3.3 ExtractionConfig

Root configuration object for a single pipeline run.

| Field | Type | Default | Description |
|---|---|---|---|
| `processed_data_path` | `Path` | *(required)* | Root of the dataset produced by `processing_pipeline` (must contain `data/` and `meta/` subdirectories) |
| `pyrat_directory` | `Path` | `/ste/rnd/User/vice_vi/pyrat` | Location of the PyRat source tree |
| `output_prefix` | `str` | `"params"` | Prefix component of the output subdirectory name |
| `output_suffix` | `str \| None` | `None` | Optional override for the auto-generated output suffix |
| `tomogram_filename` | `str \| None` | `None` | Filename of the tomogram to process; discovered automatically if `None` |
| `height_range` | `Tuple[float, float] \| None` | `None` | Elevation range in metres; discovered from `config_state_*.json` if `None` |
| `fit_settings` | `FitSettings` | default instance | Fitting configuration |
| `parameter_workers` | `int` | `20` | Number of CPU workers for the prominence-based initialisation |
| `use_gpu` | `bool` | `True` | Enable JAX GPU backend |
| `gpu_batch_size` | `int` | `256` | Number of range bins loaded per batch |
| `gpu_pixel_batch_size` | `int` | `24576` | Number of pixels per Adam call on GPU |
| `adam_steps` | `int` | `3000` | Number of Adam optimisation steps |
| `adam_lr` | `float` | `2e-1` | Adam learning rate |
| `adam_b1` | `float` | `0.95` | Adam first-moment decay coefficient |
| `adam_b2` | `float` | `0.999` | Adam second-moment decay coefficient |
| `gpu_device_ids` | `List[int] \| None` | `[0, 1, 3]` | GPU device indices to activate; `None` uses all available devices |

**Derived properties:**

| Property | Value |
|---|---|
| `data_directory` | `processed_data_path / "data"` |
| `metadata_directory` | `processed_data_path / "meta"` |
| `output_suffix_value` | Auto-generated from fitting method and `k_max`, e.g. `"Ng5_sigonly_k5"` |
| `output_subdir_name` | `f"{output_prefix}_{output_suffix_value}"` |
| `output_directory` | `processed_data_path / "params" / output_subdir_name` |
| `parameters_npy_path` | `output_directory / f"parameters_{output_suffix_value}.npy"` |

**Autodiscovery methods:**

- `discover_tomogram_path()` — Resolves the full-stack tomogram path by checking, in order: (1) the explicitly supplied `tomogram_filename` in `data_directory`; (2) the `artifacts.tomogram_full` key in `dataset.json`; (3) glob pattern `tomogram_full_*params*.npy`; (4) glob pattern `tomogram_full_*.npy`. This ensures compatibility with datasets produced by different `processing_pipeline` runs without requiring hard-coded paths.

- `discover_height_range()` — Returns `height_range` if explicitly set. Otherwise, parses `config_state_*.json` files in `metadata_directory` and extracts the `height_range` field from either `output_configs` or `input_configs`, ensuring consistency with the beamforming parameters used during tomogram generation.

---

## 4. Component Responsibilities

### 4.1 ParameterExtractor

**Module:** `pipelines/param_pipeline/fitting.py`  
**Class:** `ParameterExtractor`

The `ParameterExtractor` is the public-facing fitting interface. It owns a `SigmaFittingExtractor` instance and provides two additional responsibilities:

1. **Post-processing — by-centroid sorting:** After the raw parameter array is returned by `SigmaFittingExtractor`, components are re-ordered along each pixel's parameter vector in ascending centroid height $\mu_k$ order. Inactive Gaussians (amplitude $\leq 10^{-7}$) are assigned a sentinel centroid of $+\infty$ and therefore sorted to the end of the parameter vector. This canonical ordering guarantees that Gaussian slot $k=0$ consistently carries the lowest-elevation scatterer when multiple components are active, enabling unambiguous downstream interpretation.

2. **Logging and configuration delegation:** Logs the chosen backend and fitting method on initialisation.

**Constructor:**

```python
ParameterExtractor(
    parameter_extraction : FitSettings,
    parameter_workers    : int,
    logger               : Logger,
    use_gpu              : bool  = True,
    gpu_batch_size       : int   = 256,
    adam_steps           : int   = 800,
    adam_lr              : float = 1e-2,
    adam_b1              : float = 0.9,
    adam_b2              : float = 0.999,
    gpu_device_ids       : list | None = None,
    gpu_pixel_batch_size : int   = 8192,
    init_workers         : int | None = None,
)
```

**`run()` method:**

```python
run(tomogram_path: Path, height_range: Tuple[float, float]) -> np.ndarray
```

Returns a parameter array of shape `(3 × K_max, azimuth, range)`, dtype `float32`. Channels are interleaved as $[A_0, \mu_0, \sigma_0,\; A_1, \mu_1, \sigma_1,\; \ldots,\; A_{K-1}, \mu_{K-1}, \sigma_{K-1}]$.

---

### 4.2 SigmaFittingExtractor

**Module:** `pipelines/param_pipeline/sigma_fitting.py`  
**Class:** `SigmaFittingExtractor`

The core computational engine. Implements the full three-phase fitting algorithm (CPU initialisation → GPU sigma optimisation → best-K selection) described in detail in [Section 7](#7-gpu-execution-and-parallelism-strategy).

**Key design decisions:**

- The tomogram is accessed via a **memory-mapped** NumPy array (`mmap_mode="r"`), enabling processing of arbitrarily large tomograms without loading the full array into RAM.
- Data is loaded in **range-bin batches** (`range_batch_size`) with double-buffered prefetching via a `ThreadPoolExecutor(max_workers=2)` to overlap I/O with computation.
- Profile amplitudes are **normalised per-pixel** before GPU fitting (`profile / max(profile)`) and **denormalised** after fitting, decoupling the optimisation problem from absolute backscatter scale and improving gradient conditioning.
- The JAX kernel is **compiled once** via a warm-up call before the main loop, amortising JIT compilation cost across all batches.

**Module-level helpers:**

| Function | Description |
|---|---|
| `_evaluate_gaussian(height_axis, amps, mus, sigs)` | Evaluates the multi-Gaussian sum for a batch of pixels (NumPy); used for scoring in Phase 3 |
| `_prominence_worker(smoothed_chunk, ...)` | Per-chunk CPU worker that identifies initial peak positions via `scipy.signal.find_peaks` |
| `_prominence_batch(prof_raw, ...)` | Dispatches `_prominence_worker` across CPU processes; returns initial `(amps, mus, sigs)` for $K = K_{\max}$ |

---

### 4.3 FittingMetricsCalculator

**Module:** `pipelines/param_pipeline/metrics.py`  
**Class:** `FittingMetricsCalculator`

Computes the full suite of spatial quality metrics from the parameter array. All operations are performed in **float64** for numerical stability in $R^2$ computation. Large intermediate arrays (tomogram, reconstructed profiles) are explicitly deleted and `gc.collect()` is called after use to keep peak memory usage bounded.

**Constructor:**

```python
FittingMetricsCalculator(n_gaussians: int, logger: Logger, amp_threshold: float = 1e-3)
```

The `amp_threshold` parameter defines the minimum amplitude below which a Gaussian slot is considered **inactive** and its parameters are masked to `NaN` in spatial maps.

**`run()` method:**

```python
run(
    parameters_array : np.ndarray,   # shape (3K, Az, R)
    metadata         : dict,         # deserialized param_extraction_meta.json
    tomogram_path    : Path,
) -> dict
```

Returns a dictionary with the following keys:

| Key | Shape | Description |
|---|---|---|
| `r2_map` | `(Az, R)` | Per-pixel $R^2$ of the Gaussian fit against the tomogram |
| `activity_map` | `(Az, R)` | Integer count of active Gaussians per pixel |
| `height_axis` | `(H,)` | Elevation axis in metres |
| `global_summary` | `dict` | Scalar statistics: $R^2$ percentiles (p10–p90), mean, median, std, negative fraction, active-$K$ fractions |
| `amp_{k}` | `(Az, R)` | Amplitude map for Gaussian slot $k$ (inactive pixels = `NaN`) |
| `mu_{k}` | `(Az, R)` | Centroid height map for slot $k$ in metres (inactive pixels = `NaN`) |
| `sigma_{k}` | `(Az, R)` | Spread map for slot $k$ in metres (inactive pixels = `NaN`) |
| `mu_sep_{k}_{k+1}` | `(Az, R)` | $\vert\mu_{k+1} - \mu_k\vert$ in metres (masked unless both slots active) |

---

### 4.4 FittingResultPlotter

**Module:** `pipelines/param_pipeline/plots.py`  
**Class:** `FittingResultPlotter`

Generates a comprehensive set of publication-quality figures using Matplotlib. The backend is forced to `"Agg"` (headless rendering). All figures use the `_SCIENTIFIC_RC` stylesheet, which configures serif fonts (Times New Roman / DejaVu Serif), inward ticks, minor tick marks, and 300 DPI rasterisation — consistent with IEEE/IGARSS publication standards.

**Constructor:**

```python
FittingResultPlotter(
    output_directory : Path,
    n_gaussians      : int,
    logger           : Logger,
    fig_dpi          : int   = 150,   # screen rendering DPI
    save_dpi         : int   = 300,   # output file DPI
    n_fits_per_tier  : int   = 5,     # example fits per R² tier
    amp_threshold    : float = 1e-3,
)
```

**Figure catalogue** (produced by `run()`):

| Key | File | Description |
|---|---|---|
| `n_gaussians_map` | `colormaps/n_gaussians_map.png` | Discrete colour map of active-Gaussian count per pixel with per-value percentage annotations |
| `r2_spatial_map` | `colormaps/r2_map.png` | Per-pixel $R^2$ spatial map (RdYlGn colormap, clipped at 1st percentile) |
| `amplitude_maps` | `colormaps/amplitude_maps.png` | Spatial amplitude maps for each Gaussian slot (inactive pixels masked) |
| `mu_maps` | `colormaps/mu_maps.png` | Spatial centroid height maps in metres (RdYlGn) |
| `sigma_maps` | `colormaps/sigma_maps.png` | Spatial spread maps in metres (viridis) |
| `mu_separation_maps` | `colormaps/mu_separation_maps.png` | Adjacent centroid separation maps in metres (magma); only generated if $K \geq 2$ |
| `r2_distribution` | `distributions/r2_distribution.png` | $R^2$ histogram + Gaussian KDE + percentile markers; empirical CDF panel |
| `amp_dist` | `distributions/amp_distribution.png` | Violin plot of amplitude distributions per Gaussian slot (active pixels only) |
| `mu_dist` | `distributions/mu_distribution.png` | Violin plot of centroid height distributions |
| `sigma_dist` | `distributions/sigma_distribution.png` | Violin plot of spread distributions |
| `global_summary` | `metrics/global_summary.png` | Bar charts of $R^2$ percentiles and active-Gaussian count fractions |
| `example_fits_low` | `example_fits/tier_low.png` | Per-pixel fit plots for $R^2 \leq p_{25}$ pixels |
| `example_fits_mid` | `example_fits/tier_mid.png` | Per-pixel fit plots for $R^2 \in [p_{40}, p_{60}]$ pixels |
| `example_fits_high` | `example_fits/tier_high.png` | Per-pixel fit plots for $R^2 \geq p_{75}$ pixels |

Example fit figures use a two-panel layout per pixel: the upper panel overlays the measured profile (black), the total Gaussian fit (coloured dashed), and individual Gaussian components (filled areas); the lower panel shows the signed residual $\varepsilon = \text{data} - \text{fit}$ with positive/negative colour fill.

---

### 4.5 ExtractionMetadataManager

**Module:** `pipelines/param_pipeline/metadata.py`  
**Class:** `ExtractionMetadataManager`

Writes a single JSON provenance record for the pipeline run.

**`save_run_metadata()` output** — `{output_directory}/param_extraction_meta.json`:

```json
{
    "timestamp"           : "2026-06-01T14:32:05",
    "processed_data_path" : "/ste/rnd/User/vice_vi/Dataset/clean_dataset",
    "source_tomogram"     : "/ste/rnd/.../tomogram_full_*.npy",
    "height_range"        : [-20.0, 80.0],
    "output_directory"    : "/ste/rnd/.../params/params_Ng5_sigonly_k5",
    "output_prefix"       : "params",
    "output_suffix"       : "Ng5_sigonly_k5",
    "parameters_npy"      : "parameters_Ng5_sigonly_k5.npy",
    "number_of_gaussians" : 5
}
```

---

### 4.6 ParamExtractionPipeline

**Module:** `pipelines/param_pipeline/pipeline.py`  
**Class:** `ParamExtractionPipeline`

The top-level orchestrator. Instantiates all four components, resolves the tomogram path and height range via `ExtractionConfig`'s autodiscovery methods, and sequences the four processing stages. A default `Logger` writing to `{output_directory}/logs/param_extraction.log` is created if none is provided.

**Constructor:**

```python
ParamExtractionPipeline(config: ExtractionConfig, logger: Logger | None = None)
```

**`run()` method:**

```python
run() -> dict[str, Path]
```

Returns:

```python
{
    "parameters_npy"   : Path,   # fitted parameter array
    "metadata"         : Path,   # param_extraction_meta.json
    "output_directory" : Path,   # root output directory
    "source_tomogram"  : Path,   # resolved tomogram path
    "plots"            : dict,   # key → Path for each figure
}
```

**Execution order:**

1. Extract parameters → `_stage_extract()`
2. Persist parameter array → `_stage_save()`
3. Save run metadata → `ExtractionMetadataManager.save_run_metadata()`
4. Compute fitting quality metrics → `_stage_metrics()`
5. Generate visualisations → `_stage_plots()`

---

## 5. Pipeline Execution Stages

### 5.1 Stage 1 — Parameter Extraction

**Method:** `ParamExtractionPipeline._stage_extract()`  
**Delegated to:** `ParameterExtractor.run()` → `SigmaFittingExtractor.run()`

The tomogram identified by `ExtractionConfig.discover_tomogram_path()` is loaded as a memory-mapped array. For each pixel $(a, r)$, the elevation profile $p_{a,r}(h)$ is extracted and fitted by a sum of up to $K_{\max}$ Gaussians using the three-phase GPU algorithm described in [Section 7](#7-gpu-execution-and-parallelism-strategy).

The output is a parameter array of shape $(3 K_{\max},\; A,\; R)$ where channels are interleaved as $[A_0, \mu_0, \sigma_0,\; A_1, \mu_1, \sigma_1,\; \ldots]$. After the fitting engine returns, `ParameterExtractor` applies the by-centroid sorting step.

---

### 5.2 Stage 2 — Parameter Array Persistence

**Method:** `ParamExtractionPipeline._stage_save()`

The parameter array is saved as an uncompressed NumPy binary file (`allow_pickle=False`) to `parameters_npy_path`. The array is converted to C-contiguous layout before saving (`np.ascontiguousarray`). Immediately after, the in-memory array is deleted and `gc.collect()` is called, releasing GPU-side memory before the metrics stage loads the data from disk. This explicit deallocation is essential for large-scene processing where the parameter array can occupy several gigabytes.

---

### 5.3 Stage 3 — Fitting Quality Metrics

**Method:** `ParamExtractionPipeline._stage_metrics()`  
**Delegated to:** `FittingMetricsCalculator.run()`

The parameter array is reloaded from disk (cast to `float32`). The run metadata JSON is also reloaded to obtain the `height_range`. The tomogram is loaded (cast to `float32`). For every pixel, the fitted Gaussian sum is reconstructed and compared against the measured profile to compute the per-pixel $R^2$ map. Additional spatial maps and a global scalar summary are computed. All intermediate large arrays are released before proceeding.

---

### 5.4 Stage 4 — Visualisation

**Method:** `ParamExtractionPipeline._stage_plots()`  
**Delegated to:** `FittingResultPlotter.run()`

The parameter array is reloaded once more. The tomogram is accessed via a fresh memory map for the example-fit pixel profiles — specifically, individual elevation profiles are extracted for a stratified sample of pixels across three $R^2$ tiers (low, mid, high). All figures in the catalogue described in [Section 4.4](#44-fittingresultplotter) are generated and saved to `{output_directory}/images/`.

---

## 6. Mathematical Formulation

### 6.1 Tomographic Profile Model

Let $p(h) \in \mathbb{R}_{\geq 0}$ denote the tomographic power spectral density at height $h$ for a given pixel $(a, r)$, evaluated at $H$ discrete heights $h_1, \ldots, h_H$ uniformly spanning $[h_{\min}, h_{\max}]$.

The model approximates this profile as a mixture of $K$ Gaussian components:

$$
\hat{p}(h;\, \boldsymbol{\theta}) = \sum_{k=1}^{K} A_k \exp\!\left(-\frac{(h - \mu_k)^2}{2\sigma_k^2}\right)
$$

where $\boldsymbol{\theta} = \{A_k, \mu_k, \sigma_k\}_{k=1}^{K}$ are the amplitude, centroid, and standard deviation of the $k$-th component, respectively. The constraint $\sigma_k \in [\Delta h,\; (h_{\max} - h_{\min})/2]$ is enforced by clipping throughout, where $\Delta h = (h_{\max} - h_{\min}) / (H - 1)$ is the height resolution.

---

### 6.2 Sigma-Only Fitting Problem

The fitting strategy **fixes** the amplitudes $\{A_k\}$ and centroids $\{\mu_k\}$ from a data-driven initialisation (see [Section 7.1](#71-cpu-initialisation-phase)) and optimises only the spreads $\{\sigma_k\}$. The loss is defined on the **normalised** profile $\tilde{p}(h) = p(h) / \max_h p(h)$:

$$
\mathcal{L}(\boldsymbol{\sigma}) = \frac{1}{H} \sum_{i=1}^{H} \left(\hat{\tilde{p}}(h_i;\, \tilde{A}_k, \mu_k, \sigma_k) - \tilde{p}(h_i)\right)^2
$$

where $\tilde{A}_k = A_k / \max_h p(h)$ are the normalised amplitudes. Normalisation decouples the loss surface from the absolute backscatter intensity, improving gradient conditioning across pixels with widely varying backscatter levels.

The per-pixel loss is implemented as a JAX-differentiable scalar function (`SigmaAdamKernel._per_pixel_loss`). The gradient $\nabla_{\boldsymbol{\sigma}} \mathcal{L}$ is obtained via automatic differentiation (`jax.value_and_grad`), and the entire pixel batch is vectorised with `jax.vmap`.

---

### 6.3 Adam Optimiser

The spread parameters are optimised with the Adam algorithm. For step $t = 1, 2, \ldots, T$:

$$
\mathbf{g}_t = \nabla_{\boldsymbol{\sigma}} \mathcal{L}(\boldsymbol{\sigma}_{t-1})
$$

$$
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\, \mathbf{g}_t
$$

$$
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t \odot \mathbf{g}_t
$$

$$
\boldsymbol{\sigma}_t = \boldsymbol{\sigma}_{t-1} - \alpha \cdot \frac{\mathbf{m}_t \,/\, (1 - \beta_1^t)}{\sqrt{\mathbf{v}_t \,/\, (1 - \beta_2^t)} + \varepsilon}
$$

$$
\boldsymbol{\sigma}_t = \mathrm{clip}(\boldsymbol{\sigma}_t,\; \sigma_{\min},\; \sigma_{\max})
$$

with default hyperparameters $\alpha = 2 \times 10^{-1}$, $\beta_1 = 0.95$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$. The optimisation loop is implemented as a `jax.lax.scan` over $T$ steps, enabling full JIT compilation without Python-level iteration overhead.

---

### 6.4 Complexity-Penalised Model Selection

Because the loss $\mathcal{L}$ is monotonically non-increasing in $K$ (adding more Gaussians can only improve the fit quality on training data), a complexity-penalised score is used to determine the appropriate number of components $K^*$ for each pixel independently:

$$
\mathcal{S}(K) = \mathrm{MSE}(K) + \lambda_K \cdot K \cdot \bar{A}(K)
$$

where:

- $\mathrm{MSE}(K) = \mathcal{L}(\boldsymbol{\sigma}^*_K)$ is the optimised loss for $K$ components
- $\bar{A}(K) = \frac{1}{K}\sum_{k=1}^{K} \tilde{A}_k$ is the mean normalised amplitude across all $K$ slots
- $\lambda_K \geq 0$ is the user-controlled penalty coefficient (default `3e-3`)

The optimal component count is:

$$
K^* = \arg\min_{K \in \{1, \ldots, K_{\max}\}} \mathcal{S}(K)
$$

Parameters from non-selected slots are set to zero in the output array. The product $K \cdot \bar{A}(K)$ penalises models that use many components with non-negligible amplitudes, discouraging overfitting while allowing the model to use the full $K_{\max}$ budget when the profile is genuinely multi-layered.

---

### 6.5 Coefficient of Determination

The per-pixel goodness-of-fit is assessed by the coefficient of determination $R^2$, computed by `FittingMetricsCalculator`:

$$
R^2(a,r) = 1 - \frac{\displaystyle\sum_{i=1}^{H} \bigl(\hat{p}(h_i) - p(h_i)\bigr)^2}{\displaystyle\sum_{i=1}^{H} \bigl(p(h_i) - \bar{p}\bigr)^2 + \epsilon}
$$

where $\bar{p} = H^{-1}\sum_i p(h_i)$ and $\epsilon = 10^{-12}$ prevents division by zero for flat profiles. Note that $R^2$ can be negative for poor fits, as the denominator is measured relative to the variance of the profile rather than a fixed baseline model. The fraction of pixels with $R^2 < 0$ is reported as a quality diagnostic in the global summary.

---

## 7. GPU Execution and Parallelism Strategy

### 7.1 CPU Initialisation Phase

Before GPU optimisation begins, amplitudes and centroids are initialised for all **active pixels** (those satisfying $\max_h p(h) > 10^{-7}$) using a prominence-based peak detector:

1. Each profile is smoothed with a uniform filter of width 5 (`scipy.ndimage.uniform_filter1d`, `mode="nearest"`).
2. `scipy.signal.find_peaks` is called with minimum prominence `prominence_frac × max(profile)` and minimum inter-peak distance $\max(1,\; \sigma_{\text{guess}} / \Delta h)$ samples.
3. If more than $K_{\max}$ peaks are found, the $K_{\max}$ most prominent are retained.
4. If fewer peaks are found, remaining slots are filled by iteratively selecting the maximum of the residual signal with a suppression window of radius `min_dist` around each already-selected peak.
5. Initial spread is set to $\sigma_{\text{guess}} = \max(2\Delta h,\; (h_{\max} - h_{\min}) / (8 K_{\max}))$.

This initialisation is parallelised across CPU cores using a `ProcessPoolExecutor`, with the pixel array divided into chunks of size $\approx N_{\text{active}} / (8 \times n_{\text{workers}})$.

**Shared initialisation across all $K$:** The $K_{\max}$-component initialisation (amps, mus, sigs) is computed **once** and sub-arrays of size $K$ are reused for all $K \in \{1, \ldots, K_{\max}\}$, eliminating redundant peak detection and reducing CPU time by a factor of $K_{\max}$.

---

### 7.2 GPU Sigma Fitting Phase

For each $K \in \{1, \ldots, K_{\max}\}$:

1. The normalised amplitude and centroid arrays are constructed from the shared CPU initialisation (slicing to the first $K$ components).
2. The JAX Adam kernel is called in sub-batches of `gpu_pixel_batch_size` pixels to manage GPU memory.
3. The kernel executes a vectorised Adam loop over all pixels in the sub-batch simultaneously via `jax.vmap`.
4. The optimised $\boldsymbol{\sigma}^*_K$ values are assembled into `gpu_results[K]`.

JAX device memory is released after all $K$ values are processed via `jax.clear_caches()` and `gc.collect()`.

---

### 7.3 Best-K Selection Phase

After GPU fitting is complete, all $K$ solutions are scored in parallel using a `ThreadPoolExecutor` (GIL-releasing NumPy operations, one thread per $K$ value). The penalised score $\mathcal{S}(K)$ is computed for every pixel and every $K$. `argmin` along the $K$ axis selects $K^*$ per pixel. Parameters from non-selected slots are zeroed and the selected parameters are written to the output array.

The empirical distribution of selected $K^*$ values across all active pixels is logged as a diagnostic at the end of this phase.

---

### 7.4 Multi-GPU with pmap

When multiple GPU devices are available (or `gpu_device_ids` selects more than one), the pipeline uses `PmapSigmaAdamKernel` instead of `SigmaAdamKernel`. The pixel batch of size $N$ is padded to the nearest multiple of the device count $D$ (zero-padding), then reshaped to $(D, N/D, K)$ and dispatched to all devices via `jax.pmap`.

Each device executes the complete Adam scan independently on its pixel shard. The `in_axes` mapping correctly shards per-pixel arrays (sigmas, profiles, amps, mus) along axis 0, while the shared `height_axis` and bound scalars ($\sigma_{\min}$, $\sigma_{\max}$) are broadcast to all devices via `static_broadcasted_argnums`. Results are reassembled from shape $(D, N/D, K)$ to $(N, K)$ and de-padded before returning.

---

## 8. Artifact Naming and Directory Layout

### 8.1 Output Files

The `output_suffix_value` property of `ExtractionConfig` auto-generates a descriptive string embedded in all output filenames and directory names:

```
output_suffix_value = f"Ng{number_of_gaussians}_sigonly_k{k_max}"
```

Example for $K_{\max} = 5$, `number_of_gaussians = 5`: `"Ng5_sigonly_k5"`

| File | Description |
|---|---|
| `parameters_Ng5_sigonly_k5.npy` | Parameter array, shape `(3K_max, Az, R)`, `float32` |
| `param_extraction_meta.json` | Run provenance record |
| `images/colormaps/n_gaussians_map.png` | Active-Gaussian count spatial map |
| `images/colormaps/r2_map.png` | Per-pixel $R^2$ spatial map |
| `images/colormaps/amplitude_maps.png` | $A_k$ spatial maps for each slot |
| `images/colormaps/mu_maps.png` | $\mu_k$ spatial maps in metres |
| `images/colormaps/sigma_maps.png` | $\sigma_k$ spatial maps in metres |
| `images/colormaps/mu_separation_maps.png` | $\|\mu_{k+1} - \mu_k\|$ separation maps |
| `images/distributions/r2_distribution.png` | $R^2$ PDF histogram + KDE + empirical CDF |
| `images/distributions/amp_distribution.png` | Amplitude violin plots per slot |
| `images/distributions/mu_distribution.png` | Centroid height violin plots |
| `images/distributions/sigma_distribution.png` | Spread violin plots |
| `images/metrics/global_summary.png` | $R^2$ percentile bars + active-$K$ count fractions |
| `images/example_fits/tier_low.png` | Example fits — low-$R^2$ pixels ($R^2 \leq p_{25}$) |
| `images/example_fits/tier_mid.png` | Example fits — mid-$R^2$ pixels ($R^2 \in [p_{40}, p_{60}]$) |
| `images/example_fits/tier_high.png` | Example fits — high-$R^2$ pixels ($R^2 \geq p_{75}$) |
| `logs/param_extraction.log` | Full execution log |

### 8.2 Directory Structure

```
{processed_data_path}/
├── data/
│   ├── tomogram_full_*.npy            ← INPUT: consumed by this pipeline
│   └── dataset.json                   ← INPUT: tomogram autodiscovery
├── meta/
│   └── config_state_*.json            ← INPUT: height_range autodiscovery
└── params/
    └── params_Ng5_sigonly_k5/         ← output_directory
        ├── parameters_Ng5_sigonly_k5.npy
        ├── param_extraction_meta.json
        ├── logs/
        │   └── param_extraction.log
        └── images/
            ├── colormaps/
            │   ├── n_gaussians_map.png
            │   ├── r2_map.png
            │   ├── amplitude_maps.png
            │   ├── mu_maps.png
            │   ├── sigma_maps.png
            │   └── mu_separation_maps.png
            ├── distributions/
            │   ├── r2_distribution.png
            │   ├── amp_distribution.png
            │   ├── mu_distribution.png
            │   └── sigma_distribution.png
            ├── metrics/
            │   └── global_summary.png
            └── example_fits/
                ├── tier_low.png
                ├── tier_mid.png
                └── tier_high.png
```

---

## 9. Inputs and Outputs Summary

| Stage | Primary Input | External Dependency | Outputs |
|---|---|---|---|
| Initialisation | `ExtractionConfig` | Filesystem (autodiscovery) | Log directory, `output_directory` |
| Stage 1 — Extraction | `tomogram_full_*.npy`, `height_range` | JAX GPU, `scipy.signal`, `ProcessPoolExecutor` | Parameter array `(3K, Az, R)` in memory |
| Stage 2 — Persistence | Parameter array | NumPy | `parameters_*.npy`, `param_extraction_meta.json` |
| Stage 3 — Metrics | `parameters_*.npy`, `tomogram_full_*.npy` | NumPy (float64) | `r2_map`, `activity_map`, per-Gaussian maps, `global_summary` dict |
| Stage 4 — Plots | `parameters_*.npy`, metrics dict, `tomogram_full_*.npy` | Matplotlib, SciPy KDE | 14+ PNG figures in `images/` subdirectories |

**Return value of `ParamExtractionPipeline.run()`:**

```python
{
    "parameters_npy"   : Path,   # fitted parameter array (.npy)
    "metadata"         : Path,   # param_extraction_meta.json
    "output_directory" : Path,   # root of this run's output tree
    "source_tomogram"  : Path,   # resolved input tomogram path
    "plots"            : {       # key → Path for every generated figure
        "n_gaussians_map"      : Path,
        "r2_spatial_map"       : Path,
        "amplitude_maps"       : Path,
        "mu_maps"              : Path,
        "sigma_maps"           : Path,
        "mu_separation_maps"   : Path,   # only if K >= 2
        "r2_distribution"      : Path,
        "amp_dist"             : Path,
        "mu_dist"              : Path,
        "sigma_dist"           : Path,
        "global_summary"       : Path,
        "example_fits_low"     : Path,
        "example_fits_mid"     : Path,
        "example_fits_high"    : Path,
    }
}
```

---

## 10. Canonical Usage — Entry Points

### 10.1 `main/extract_params.py`

Fixed single-run configuration over the clean dataset with $K_{\max} = 5$ and the `SigmaOnly` fitting mode:

| Parameter | Value |
|---|---|
| Dataset path | `/ste/rnd/User/vice_vi/Dataset/clean_dataset` |
| Tomogram filename | `tomogram_full_1000a16000a500a4000_1_Xtomo_id2X.npy` |
| $K_{\max}$ | `5` |
| $\lambda_K$ | `3e-3` |
| `parameter_workers` | `50` |
| Height range | Autodiscovered from `config_state_*.json` |

**Minimal invocation:**

```python
from configuration.param_extraction_config import FitMode, ExtractionConfig, FitSettings
from pipelines.param_pipeline.pipeline import ParamExtractionPipeline
from pathlib import Path

config = ExtractionConfig(
    processed_data_path = Path("/ste/rnd/User/vice_vi/Dataset/clean_dataset"),
    tomogram_filename   = "tomogram_full_1000a16000a500a4000_1_Xtomo_id2X.npy",
    fit_settings        = FitSettings(
        fit_config = FitMode.SigmaOnly(k_max=5, lambda_k=3e-3),
    ),
    parameter_workers   = 50,
)

outputs = ParamExtractionPipeline(config).run()
```

### 10.2 `scripts/extract_params_sweep.py`

Iterates over $K_{\max} \in \{2, 3, 4, 5\}$, producing a separate output directory per value (encoded via `output_prefix = f"params_g{n_gaussians}"`). Used for comparative analysis of model complexity versus fitting quality. Each iteration is a fully independent `ParamExtractionPipeline` run with an automatically distinct `output_directory`.

---

## 11. Public API Reference

### `ParamExtractionPipeline`

```python
class ParamExtractionPipeline:
    def __init__(
        self,
        config : ExtractionConfig,
        logger : Logger | None = None,
    ) -> None: ...

    def run(self) -> dict[str, Path]: ...
```

### `ParameterExtractor`

```python
class ParameterExtractor:
    def __init__(
        self,
        parameter_extraction : FitSettings,
        parameter_workers    : int,
        logger               : Logger,
        use_gpu              : bool  = True,
        gpu_batch_size       : int   = 256,
        adam_steps           : int   = 800,
        adam_lr              : float = 1e-2,
        adam_b1              : float = 0.9,
        adam_b2              : float = 0.999,
        gpu_device_ids       : list | None = None,
        gpu_pixel_batch_size : int   = 8192,
        init_workers         : int | None = None,
    ) -> None: ...

    def run(
        self,
        tomogram_path : Path,
        height_range  : Tuple[float, float],
    ) -> np.ndarray: ...   # shape (3 * K_max, Az, R), float32
```

### `SigmaFittingExtractor`

```python
class SigmaFittingExtractor:
    def __init__(
        self,
        fit_settings         : FitSettings,
        logger               : Logger,
        range_batch_size     : int   = 256,
        adam_steps           : int   = 2000,
        adam_lr              : float = 1e-2,
        adam_b1              : float = 0.9,
        adam_b2              : float = 0.999,
        k_max                : int   = 5,
        lambda_k             : float = 3e-3,
        prominence_frac      : float = 0.05,
        gpu_pixel_batch_size : int   = 8192,
        gpu_device_ids       : list | None = None,
        init_workers         : int | None = None,
    ) -> None: ...

    def run(
        self,
        tomogram_path : Path,
        height_range  : Tuple[float, float],
    ) -> np.ndarray: ...   # shape (3 * K_max, Az, R), float32
```

### `FittingMetricsCalculator`

```python
class FittingMetricsCalculator:
    def __init__(
        self,
        n_gaussians   : int,
        logger        : Logger,
        amp_threshold : float = 1e-3,
    ) -> None: ...

    def run(
        self,
        parameters_array : np.ndarray,   # (3K, Az, R)
        metadata         : dict,
        tomogram_path    : Path,
    ) -> dict: ...
```

### `FittingResultPlotter`

```python
class FittingResultPlotter:
    def __init__(
        self,
        output_directory : Path,
        n_gaussians      : int,
        logger           : Logger,
        fig_dpi          : int   = 150,
        save_dpi         : int   = 300,
        n_fits_per_tier  : int   = 5,
        amp_threshold    : float = 1e-3,
    ) -> None: ...

    def run(
        self,
        parameters_array : np.ndarray,   # (3K, Az, R)
        metrics_dict     : dict,
        metadata         : dict,
        tomogram_path    : Path,
    ) -> Dict[str, Path]: ...
```

### `ExtractionMetadataManager`

```python
class ExtractionMetadataManager:
    def __init__(self, config: ExtractionConfig, logger: Logger) -> None: ...

    def save_run_metadata(
        self,
        npy_path      : Path,
        tomogram_path : Path,
        height_range  : tuple,
    ) -> Path: ...
```
