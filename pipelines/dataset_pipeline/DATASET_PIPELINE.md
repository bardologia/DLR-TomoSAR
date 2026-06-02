# Dataset Pipeline — Technical Reference

**Package:** `pipelines.dataset_pipeline`  
**Location:** `pipelines/dataset_pipeline/`  
**Role:** Transforms pre-processed SAR tomographic artifacts into training-ready PyTorch `DataLoader` objects with statistically fitted per-channel normalisation, configurable patch tiling, and on-the-fly spatial augmentation.

---

## Table of Contents

1. [Overview](#1-overview)  
2. [Architecture](#2-architecture)  
3. [Configuration Layer](#3-configuration-layer)  
4. [Component Responsibilities](#4-component-responsibilities)  
   - 4.1 [Layout](#41-layout)  
   - 4.2 [Cropper](#42-cropper)  
   - 4.3 [Patcher / GridInfo](#43-patcher--gridinfo)  
   - 4.4 [PatchDataset](#44-patchdataset)  
   - 4.5 [SpatialAugmenter](#45-spatialaugmenter)  
   - 4.6 [StatsComputer / Stats / Normalizer](#46-statscomputer--stats--normalizer)  
   - 4.7 [MetadataWriter](#47-metadatawriter)  
   - 4.8 [Loader](#48-loader)  
   - 4.9 [DatasetPipeline](#49-datasetpipeline)  
5. [Pipeline Execution Stages](#5-pipeline-execution-stages)  
6. [Mathematical Formulation](#6-mathematical-formulation)  
   - 6.1 [Input Tensor Construction](#61-input-tensor-construction)  
   - 6.2 [Output Tensor Construction](#62-output-tensor-construction)  
   - 6.3 [Patch Grid Geometry](#63-patch-grid-geometry)  
   - 6.4 [Normalisation Strategies](#64-normalisation-strategies)  
   - 6.5 [Spatial Augmentations](#65-spatial-augmentations)  
7. [Normalisation Strategy Table](#7-normalisation-strategy-table)  
8. [Artifact Naming and Directory Layout](#8-artifact-naming-and-directory-layout)  
9. [Inputs and Outputs Summary](#9-inputs-and-outputs-summary)  
10. [Canonical Usage](#10-canonical-usage)  
11. [Public API Reference](#11-public-api-reference)

---

## 1. Overview

The dataset pipeline bridges the **pre-processing pipeline** (SAR focusing, tomogram formation, interferogram reduction) and the **neural-network training pipeline**. It performs six logically distinct operations:

| Step | Operation |
|------|-----------|
| **Layout** | Parses the `dataset.json` manifest written by the pre-processing pipeline to locate `.npy` artifact files. |
| **Cropping** | Reads memory-mapped `.npy` arrays and extracts the spatial sub-region assigned to each dataset split (train / val / test) in global pixel coordinates. |
| **Patching** | Tiles each split region into overlapping fixed-size patches with symmetric reflective padding to ensure full coverage. |
| **Normalisation** | Computes per-slot-kind statistics from the training split and applies them consistently across all splits. |
| **Augmentation** | Applies randomised spatial transforms (flips, rotations, additive noise) to training samples at retrieval time. |
| **Loading** | Wraps each split's `PatchDataset` in a PyTorch `DataLoader` configured for multi-process prefetching and pinned-memory transfer. |

The single public entry point is `DatasetPipeline.run()`, which returns three `DataLoader` objects and a dictionary of the underlying `PatchDataset` instances.

---

## 2. Architecture

```
DatasetConfiguration
        │
        ▼
┌───────────────────┐
│  DatasetPipeline  │  (pipeline.py)
└──────┬────────────┘
       │ reads manifest
       ▼
  ┌──────────┐
  │  Layout  │  (metadata.py)  ← dataset.json
  └────┬─────┘
       │ resolves artifact paths
       ▼
  ┌──────────┐
  │  Cropper │  (crop.py)  ← primary_reduced.npy
  └────┬─────┘               secondaries_reduced.npy
       │                     interferograms_reduced.npy
       │                     parameters.npy
       │ per-split arrays
       ▼
  ┌──────────┐
  │  Patcher │  (patch.py)  ← spatial_size, patch_size, stride
  └────┬─────┘
       │ patch coordinates
       ▼
  ┌───────────────────────────────────────────────────────┐
  │                    PatchDataset                       │
  │  (load.py)                                            │
  │  ┌──────────────────┐  ┌────────────────────────────┐ │
  │  │ _build_input_    │  │  _build_output_tensor      │ │
  │  │   tensor         │  │  (select Gaussian channels)│ │
  │  └────────┬─────────┘  └──────────┬─────────────────┘ │
  │           │  SpatialAugmenter      │                   │
  │           │  (augmentation.py)     │                   │
  │           │  Normalizer            │                   │
  │           │  (normalize.py)        │                   │
  └───────────┼────────────────────────┼───────────────────┘
              │                        │
              ▼                        ▼
        input_tensor              gt_params
         (float32)                (float32)
              │                        │
              └──────────┬─────────────┘
                         ▼
                  PyTorch DataLoader
                  (Loader, load.py)

  StatsComputer (normalize.py) ─── fit on train split ──► normalization_stats.json
  MetadataWriter (metadata.py) ─── write JSON records ──► meta/
```

---

## 3. Configuration Layer

All behaviour is governed by a hierarchy of frozen dataclasses defined in `configuration/dataset_config.py` and `configuration/norm_config.py`. No global mutable state exists.

### `DatasetConfiguration`

The root configuration object passed to `DatasetPipeline`.

| Field | Type | Description |
|-------|------|-------------|
| `preprocessing_run_directory` | `Path` | Root of the pre-processing run whose artifacts are consumed. |
| `split_regions` | `SplitRegions` | Ordered dict mapping split names (`"train"`, `"val"`, `"test"`) to `CropRegion` objects in **global** pixel coordinates. |
| `parameters_path` | `Optional[Path]` | Override path to the Gaussian parameter `.npy` artifact; defaults to the path declared in `dataset.json`. |
| `patch` | `PatchConfiguration` | Patch size, stride, and padding mode. |
| `input_config` | `InputConfig` | Which SAR data modalities to include and their complex-to-real representation. |
| `output_config` | `OutputConfig` | Which Gaussian parameter roles (amplitude, mean, sigma) to regress. |
| `augmentation` | `AugmentationConfig` | Probabilities and magnitudes for spatial augmentations. |
| `batch_size` | `int` | DataLoader batch size. |
| `num_workers` | `int` | DataLoader worker processes. |
| `shuffle_train` | `bool` | Shuffle training DataLoader. Default: `True`. |
| `pin_memory` | `bool` | Pin host memory for GPU transfers. Default: `True`. |
| `n_gaussians` | `int` | Number of Gaussian components per pixel in the output. |

### `InputConfig`

Controls which modalities enter the network and how complex SLC data is converted to real-valued channels.

| Field | Default | Description |
|-------|---------|-------------|
| `use_primary` | `True` | Include the primary (master) SLC magnitude or complex channels. |
| `primary_representation` | `MAG_ONLY` | `Representation` enum: `MAG_ONLY`, `ANGLE_ONLY`, `REAL_IMAG`, `MAG_PHASE`. |
| `use_secondaries` | `False` | Include secondary (slave) SLC channels. |
| `secondaries_representation` | `MAG_ONLY` | Same `Representation` enum as above. |
| `use_interferograms` | `True` | Include co-registered interferograms. |
| `interferograms_representation` | `ANGLE_ONLY` | Typically interferometric phase only. |

Total input channels:

$$
C_{\text{in}} = C_{\text{prim}} + N_s \cdot (C_{\text{sec}} + C_{\text{ifg}})
$$

where $N_s$ is the number of secondary passes and $C_{\cdot}$ denotes channels-per-pass for the respective representation.

### `OutputConfig`

Selects which Gaussian parameter roles form the regression target.

| Field | Default | Description |
|-------|---------|-------------|
| `use_amplitude` | `True` | Include peak amplitude $a$ per Gaussian. |
| `use_mu` | `True` | Include elevation mean $\mu$ per Gaussian. |
| `use_sigma` | `True` | Include elevation width $\sigma$ per Gaussian. |

Total output channels:

$$
C_{\text{out}} = K \cdot P
$$

where $K$ = `n_gaussians` and $P \in \{1, 2, 3\}$ is the number of selected roles per component.

### `PatchConfiguration`

| Field | Default | Description |
|-------|---------|-------------|
| `size` | `(64, 64)` | Patch height and width in pixels. |
| `stride` | `32` | Step between consecutive patch origins; controls overlap. |
| `use_reflective_padding` | `True` | Pad image boundaries with symmetric (mirror) reflection rather than zeros. |

### `AugmentationConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `p_flip_h` | `0.5` | Probability of horizontal (range-axis) flip. |
| `p_flip_v` | `0.5` | Probability of vertical (azimuth-axis) flip. |
| `p_rot90` | `0.0` | Probability of a random 90°/180°/270° rotation. |
| `noise_std` | `0.01` | Standard deviation of additive Gaussian noise. |
| `p_noise` | `0.25` | Probability of applying additive noise. |

### `NormMethod` and `ChannelStrategy` (`configuration/norm_config.py`)

Each slot-kind (e.g. `"pass/mag"`, `"ifg/phase"`, `"out/sigma"`) is associated with a `ChannelStrategy` that specifies:

- **`NormMethod`**: one of `MIN_MAX_P999`, `ROBUST_IQR`, `FIXED_DIV_PI`, `ZSCORE`.
- **`apply_log1p`**: whether to apply $\log(1 + |x|)$ before fitting or transforming.

---

## 4. Component Responsibilities

### 4.1 `Layout`

**File:** `metadata.py`

`Layout` is the bridge between a pre-processing run and the dataset pipeline. On construction it opens and parses the `dataset.json` manifest located at `{preprocessing_run_directory}/data/dataset.json`.

**Parsed fields:**

| JSON key | Python attribute | Description |
|----------|-----------------|-------------|
| `global_crop` | `global_crop: CropRegion` | Bounding box (azimuth_start, azimuth_end, range_start, range_end) of the entire cropped region in global pixel coordinates. |
| `dataset_type` | `dataset_type: str` | Dataset provenance tag (e.g. `"simulated"`). |
| `tomogram_tag` | `tomogram_tag: str` | Tag identifying the tomographic processing variant. |
| `parameter_tag` | `parameter_tag: str` | Tag identifying the Gaussian fitting variant. |
| `artifacts` | `artifacts: dict` | Maps logical artifact keys (`"primary_reduced"`, `"secondaries_reduced"`, `"interferograms_reduced"`) to file names relative to `data/`. |

**Key method:**

```python
Layout.artifact_path(artifact_key: str) -> Path
```

Returns the absolute path to the requested artifact. For `artifact_key == "parameters"` it redirects to `DatasetConfiguration.parameters_path`.

---

### 4.2 `Cropper`

**File:** `crop.py`

`Cropper` extracts the spatial region corresponding to a dataset split from the full memory-mapped `.npy` arrays. It performs coordinate translation from the global pixel reference frame to the local reference frame of the pre-processed crop.

**Coordinate translation:**

For a split region $R = (az_0, az_1, rg_0, rg_1)$ and global crop $G = (G_{az_0}, G_{az_1}, G_{rg_0}, G_{rg_1})$:

$$
\text{az\_slice} = \left[az_0 - G_{az_0},\; az_1 - G_{az_0}\right)
$$
$$
\text{rg\_slice} = \left[rg_0 - G_{rg_0},\; rg_1 - G_{rg_0}\right)
$$

**Loaded arrays per split:**

| Key | Source artifact | Shape |
|-----|----------------|-------|
| `primary` | `primary_reduced.npy` | `(1, Az, Rg)` |
| `secondaries` | `secondaries_reduced.npy` | `(N_s, Az, Rg)` |
| `interferograms` | `interferograms_reduced.npy` | `(N_s, Az, Rg)` |
| `parameters` | path from `Layout` | `(3K, Az, Rg)` |
| `inputs` | concatenation | `(1 + 2N_s, Az, Rg)` |

The `inputs` tensor is assembled as:

$$
\mathbf{X}_{\text{raw}} = \left[\mathbf{x}_{\text{prim}},\; \mathbf{x}^{(1)}_{\text{sec}}, \ldots, \mathbf{x}^{(N_s)}_{\text{sec}},\; \mathbf{x}^{(1)}_{\text{ifg}}, \ldots, \mathbf{x}^{(N_s)}_{\text{ifg}} \right]
$$

along the channel axis (axis 0). All arrays are loaded with `mmap_mode="r"` and extracted as contiguous copies.

---

### 4.3 `Patcher` / `GridInfo`

**File:** `patch.py`

`Patcher` computes a regular grid of overlapping patches that tiles a spatial region of size $(H, W)$ with patch size $(P_h, P_w)$ and stride $s$.

**Grid dimensions:**

$$
n_v = \begin{cases} 1 & H \leq P_h \\ \left\lceil \dfrac{H - P_h}{s} \right\rceil + 1 & \text{otherwise} \end{cases}
\qquad
n_h = \begin{cases} 1 & W \leq P_w \\ \left\lceil \dfrac{W - P_w}{s} \right\rceil + 1 & \text{otherwise} \end{cases}
$$

**Required padding:**

$$
\text{pad}_v = P_h + (n_v - 1) \cdot s - H
\qquad
\text{pad}_h = P_w + (n_h - 1) \cdot s - W
$$

Distributed symmetrically:

$$
\text{pad\_top} = \lfloor \text{pad}_v / 2 \rfloor, \quad \text{pad\_bot} = \text{pad}_v - \lfloor \text{pad}_v / 2 \rfloor
$$

$$
\text{pad\_left} = \lfloor \text{pad}_h / 2 \rfloor, \quad \text{pad\_right} = \text{pad}_h - \lfloor \text{pad}_h / 2 \rfloor
$$

Total number of patches:

$$
N_{\text{patches}} = n_v \cdot n_h
$$

Each patch's clipping coordinates and any required edge padding are pre-computed at construction time and stored as a list of 5-tuples `(v0c, v1c, h0c, h1c, pw_spec)`. At extraction time (`Patcher.extract`) the clipped sub-array is loaded and, if needed, padded with `numpy.pad` using mode `"symmetric"` (reflective) or `"constant"` (zero).

**`GridInfo`** is a serialisable dataclass that records all grid parameters and is written to `meta/patch.json` for downstream reproducibility.

---

### 4.4 `PatchDataset`

**File:** `load.py`

`PatchDataset` is a PyTorch `Dataset` that maps an integer patch index to a `(input_tensor, gt_params)` pair.

**Construction parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `inputs` | `ndarray` `(C, Az, Rg)` | Full complex split array from `Cropper`. |
| `gt_parameters` | `ndarray` `(3K, Az, Rg)` | Full Gaussian parameter map. |
| `grid` | `Patcher` | Pre-computed patch grid. |
| `input_config` | `InputConfig` | Controls channel selection and representation conversion. |
| `output_config` | `OutputConfig` | Controls which parameter roles are included. |
| `norm_stats` | `Optional[Stats]` | If provided, wraps a `Normalizer` for in-place normalisation. |
| `augmenter` | `Optional[SpatialAugmenter]` | Applied only when `split_name == "train"`. |
| `n_gaussians` | `int` | Number of Gaussian components. |

**`__getitem__` pipeline:**

```
idx  →  Patcher.extract(inputs, idx)      # complex patch  (C, Ph, Pw)
     →  _build_input_tensor               # real channels  (C_in, Ph, Pw)
     →  Patcher.extract(gt_parameters, idx)
     →  _build_output_tensor              # selected roles (C_out, Ph, Pw)
     →  SpatialAugmenter (train only)
     →  Normalizer.normalize_input
     →  Normalizer.normalize_output
     →  return (float32, float32)
```

---

### 4.5 `SpatialAugmenter`

**File:** `augmentation.py`

`SpatialAugmenter` applies identical geometric transforms to both the input tensor and the ground-truth parameter map to preserve their spatial correspondence. It is stateful only in its `numpy.random.Generator` instance (seeded per-process).

**Transforms applied (in order):**

1. Horizontal flip (axis −1) with probability $p_{\text{flip\_h}}$.
2. Vertical flip (axis −2) with probability $p_{\text{flip\_v}}$.
3. Random 90° rotation ($k \in \{1, 2, 3\}$) with probability $p_{\text{rot90}}$.
4. Additive Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$ applied to input tensor only, with probability $p_{\text{noise}}$.

Note: augmentation is applied **after** patch extraction but **before** normalisation, operating on the real-valued input representation.

---

### 4.6 `StatsComputer` / `Stats` / `Normalizer`

**File:** `normalize.py`

#### `Stats`

A lightweight dataclass holding two `ChannelStats` objects (one for inputs, one for outputs). Serialisable to / from `normalization_stats.json`.

```python
@dataclass
class Stats:
    input_stats  : Optional[ChannelStats]
    output_stats : Optional[ChannelStats]
```

#### `StatsComputer`

A stateless class of `@staticmethod` methods responsible for computing normalisation parameters from data.

**Input statistics (`compute_input_stats`):**

1. Assigns each input channel to a *slot-kind* group (e.g. `"pass/mag"`, `"ifg/phase"`) based on `InputConfig` and the number of secondary passes.
2. Builds a `DataLoader` over an optional random subset of the training `PatchDataset`.
3. Collects a reservoir of scalar values per group (up to `max_vals_per_group = 1_000_000` per channel).
4. Fits a `(loc, scale)` pair for each group using the assigned `ChannelStrategy.fit`.
5. Returns a `Stats` with `input_stats` populated.

**Output statistics (`compute_output_stats`):**

1. Memory-maps the full parameter `.npy` file.
2. Pools amplitude, mean, and sigma values across all pixels and all Gaussian components.
3. Applies an amplitude threshold ($a > 10^{-2}$) to exclude inactive-component means and sigmas.
4. Fits per-role strategies (`"out/amp"`, `"out/mu"`, `"out/sigma"`).
5. Returns a `Stats` with `output_stats` populated.

#### `Normalizer`

Applies the fitted `(loc, scale)` transformations. Supports both `numpy` arrays and `torch.Tensor` objects.

**Forward (normalise):**

$$
\hat{x}_c = \frac{f(x_c) - \mu_c}{\sigma_c}
$$

where $f(x) = \log(1 + \max(x, 0))$ if `apply_log1p` is `True`, otherwise $f(x) = x$.

**Inverse (denormalise):**

$$
x_c = f^{-1}(\hat{x}_c \cdot \sigma_c + \mu_c)
$$

where $f^{-1}(y) = e^y - 1$ (clamped to $y \leq 15$ to prevent overflow) if `apply_log1p` is `True`.

---

### 4.7 `MetadataWriter`

**File:** `metadata.py`

`MetadataWriter` serialises configuration and grid metadata to the training run's `meta/` subdirectory.

| Output file | Method | Content |
|-------------|--------|---------|
| `meta/dataset_creation_config.json` | `save_dataset_configuration` | Full `DatasetConfiguration` as JSON. |
| `meta/crop.json` | `save_crop_metadata` | Global crop bounding box and per-split bounding boxes. |
| `meta/patch.json` | `save_patch_metadata` | Per-split `GridInfo` dicts including patch count, padding, stride. |

---

### 4.8 `Loader`

**File:** `load.py`

`Loader.build` is a factory static method that wraps three `PatchDataset` instances in PyTorch `DataLoader` objects.

| Parameter | Train | Val | Test |
|-----------|-------|-----|------|
| `shuffle` | `True` (configurable) | `False` | `False` |
| `drop_last` | `True` | `False` | `False` |
| `prefetch_factor` | 8 | 8 | 8 |
| `persistent_workers` | `True` if `num_workers > 0` | same | same |

---

### 4.9 `DatasetPipeline`

**File:** `pipeline.py`

Orchestrates all components. Constructed with a `DatasetConfiguration` and a training run directory, then driven by a single call to `run()`.

**Constructor** initialises `Layout`, `Cropper`, `MetadataWriter`, and `SpatialAugmenter`. No data is loaded at construction time.

**`run()` return type:** `Tuple[DataLoader, DataLoader, DataLoader, dict[str, PatchDataset]]`

---

## 5. Pipeline Execution Stages

### Stage 0 — Initialisation

```
DatasetPipeline.__init__
    Layout(preprocessing_run_directory)   ← parses dataset.json
    Cropper(layout, split_regions)
    MetadataWriter(training_run_directory)
    SpatialAugmenter(augmentation_config)
```

No I/O beyond JSON parsing.

---

### Stage 1 — Training Split Construction and Statistics Fitting

```
DatasetPipeline.run()
    ├─ _build_dataset("train")
    │      Cropper.load_split(train_region)        ← mmap + copy 4 × .npy
    │      Patcher.build(spatial_size, patch_size, stride)
    │      PatchDataset(inputs, gt_params, patcher, ...)  [no norm_stats yet]
    │
    ├─ StatsComputer.compute_input_stats(train_dataset)
    │      builds DataLoader over (subset of) train_dataset
    │      collects per-slot-kind value reservoirs
    │      fits (loc, scale) per channel
    │      → Stats(input_stats=...)
    │
    ├─ StatsComputer.compute_output_stats(parameters_path)
    │      mmaps parameters.npy
    │      pools per-role values
    │      fits (loc, scale) per role
    │      → Stats(output_stats=...)
    │
    ├─ Stats.save(training_run_directory / "meta")
    │      → meta/normalization_stats.json
    │
    └─ train_dataset.norm_stats = Normalizer(norm_stats)
```

---

### Stage 2 — Validation and Test Split Construction

```
    ├─ _build_dataset("val",  norm_stats=norm_stats)
    │      Cropper.load_split(val_region)
    │      Patcher.build(...)
    │      PatchDataset(..., norm_stats=norm_stats)
    │
    └─ _build_dataset("test", norm_stats=norm_stats)
           Cropper.load_split(test_region)
           Patcher.build(...)
           PatchDataset(..., norm_stats=norm_stats)
```

Statistics are fitted **only** on the training split and applied identically to val and test to prevent data leakage.

---

### Stage 3 — DataLoader Assembly and Metadata Persistence

```
    ├─ Loader.build(train_ds, val_ds, test_ds, ...)
    │      → (train_loader, val_loader, test_loader)
    │
    ├─ MetadataWriter.save_dataset_configuration(config)
    │      → meta/dataset_creation_config.json
    ├─ MetadataWriter.save_crop_metadata(global_crop, splits)
    │      → meta/crop.json
    └─ MetadataWriter.save_patch_metadata({train, val, test})
           → meta/patch.json
```

---

## 6. Mathematical Formulation

### 6.1 Input Tensor Construction

Let $\mathbf{X}_{\text{raw}} \in \mathbb{C}^{(1 + 2N_s) \times H \times W}$ be the raw complex split array. Define the three modality groups:

$$
\mathbf{x}_{\text{prim}} = \mathbf{X}_{\text{raw}}[0] \in \mathbb{C}^{H \times W}
$$
$$
\mathbf{x}_{\text{sec}}^{(i)} = \mathbf{X}_{\text{raw}}[i] \in \mathbb{C}^{H \times W}, \quad i = 1, \ldots, N_s
$$
$$
\mathbf{x}_{\text{ifg}}^{(i)} = \mathbf{X}_{\text{raw}}[N_s + i] \in \mathbb{C}^{H \times W}, \quad i = 1, \ldots, N_s
$$

A `Representation` $\rho$ maps a complex array to $C_\rho$ real channels:

| `Representation` | $C_\rho$ | Conversion $\rho(\mathbf{z})$ |
|-----------------|---------|-------------------------------|
| `MAG_ONLY` | 1 | $|\mathbf{z}|$ |
| `ANGLE_ONLY` | 1 | $\angle \mathbf{z}$ |
| `REAL_IMAG` | 2 | $[\operatorname{Re}(\mathbf{z}),\; \operatorname{Im}(\mathbf{z})]$ |
| `MAG_PHASE` | 2 | $[|\mathbf{z}|,\; \angle \mathbf{z}]$ |

The assembled input tensor for a patch is:

$$
\mathbf{T}_{\text{in}} = \left[
  \rho_{\text{prim}}(\mathbf{x}_{\text{prim}}),\;
  \rho_{\text{sec}}(\mathbf{x}^{(1)}_{\text{sec}}), \ldots, \rho_{\text{sec}}(\mathbf{x}^{(N_s)}_{\text{sec}}),\;
  \rho_{\text{ifg}}(\mathbf{x}^{(1)}_{\text{ifg}}), \ldots, \rho_{\text{ifg}}(\mathbf{x}^{(N_s)}_{\text{ifg}})
\right] \in \mathbb{R}^{C_{\text{in}} \times P_h \times P_w}
$$

where $C_{\text{in}} = C_{\rho_{\text{prim}}} + N_s (C_{\rho_{\text{sec}}} + C_{\rho_{\text{ifg}}})$.

### 6.2 Output Tensor Construction

The full parameter map has shape $(3K, H, W)$ with layout:

$$
\mathbf{P}[3k + 0] = a^{(k)}, \quad \mathbf{P}[3k + 1] = \mu^{(k)}, \quad \mathbf{P}[3k + 2] = \sigma^{(k)}, \quad k = 0, \ldots, K-1
$$

`OutputConfig.selected_indices` constructs a flat index list selecting the active roles. For example, with `use_amplitude=True, use_mu=True, use_sigma=False` and $K = 2$:

$$
\text{selected} = [0, 1, 3, 4]
$$

The output tensor is:

$$
\mathbf{T}_{\text{out}} = \mathbf{P}[\text{selected}, :, :] \in \mathbb{R}^{C_{\text{out}} \times P_h \times P_w}
$$

### 6.3 Patch Grid Geometry

For a spatial region of size $(H, W)$, patch size $(P_h, P_w)$, and stride $s$:

The origin of patch $(i_v, i_h)$ before padding correction is:

$$
v_0 = i_v \cdot s - \text{pad\_top}, \quad h_0 = i_h \cdot s - \text{pad\_left}
$$

The clipped extraction region is:

$$
v_0^c = \max(0, v_0), \quad v_1^c = \min(H, v_0 + P_h)
$$
$$
h_0^c = \max(0, h_0), \quad h_1^c = \min(W, h_0 + P_w)
$$

Any shortfall (patch exceeds image boundary) is filled by reflective padding:

$$
\text{pad\_top}^{\text{patch}} = \max(0, -v_0), \quad \text{pad\_bot}^{\text{patch}} = \max(0, v_0 + P_h - H)
$$

with symmetric mode: $\mathbf{A}_{\text{padded}}[i] = \mathbf{A}[\text{reflect}(i)]$.

### 6.4 Normalisation Strategies

Let $x_c$ denote the raw value of channel $c$ and $\mathcal{D}_c$ the set of observed values for that channel in the training split. The four strategies are:

**`ZSCORE`:**

$$
\mu_c = \mathbb{E}[\mathcal{D}_c], \quad \sigma_c = \operatorname{std}(\mathcal{D}_c)
$$

**`MIN_MAX_P999`:**

$$
\mu_c = P_{0.1}(\mathcal{D}_c), \quad \sigma_c = P_{99.9}(\mathcal{D}_c) - P_{0.1}(\mathcal{D}_c)
$$

**`ROBUST_IQR`:**

$$
\mu_c = P_{50}(\mathcal{D}_c), \quad \sigma_c = P_{75}(\mathcal{D}_c) - P_{25}(\mathcal{D}_c)
$$

**`FIXED_DIV_PI`:**

$$
\mu_c = 0, \quad \sigma_c = \pi
$$

(used for phase channels, where the range $(-\pi, \pi]$ is known analytically)

Applied transform:

$$
\hat{x}_c = \frac{g(x_c) - \mu_c}{\sigma_c}, \quad g(x) = \begin{cases} \log(1 + \max(x, 0)) & \text{if apply\_log1p} \\ x & \text{otherwise} \end{cases}
$$

### 6.5 Spatial Augmentations

Let $\mathbf{T} \in \mathbb{R}^{C \times H \times W}$ be the input or output patch. Augmentations are applied jointly to both:

**Horizontal flip** (Bernoulli trial with $p = p_{\text{flip\_h}}$):

$$
\mathbf{T}'[c, i, j] = \mathbf{T}[c, i, W - 1 - j]
$$

**Vertical flip** (Bernoulli trial with $p = p_{\text{flip\_v}}$):

$$
\mathbf{T}'[c, i, j] = \mathbf{T}[c, H - 1 - i, j]
$$

**Rotation by $k \times 90°$** ($k \sim \mathcal{U}\{1, 2, 3\}$, Bernoulli trial with $p = p_{\text{rot90}}$):

$$
\mathbf{T}' = \operatorname{rot90}(\mathbf{T},\; k)
$$

**Additive noise** (Bernoulli trial with $p = p_{\text{noise}}$, applied to input only):

$$
\mathbf{T}'_{\text{in}} = \mathbf{T}_{\text{in}} + \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},\, \sigma_n^2 \mathbf{I})
$$

---

## 7. Normalisation Strategy Table

The slot-kind to strategy mapping is globally fixed in `configuration/norm_config.py`:

| Slot kind | `NormMethod` | `apply_log1p` | Rationale |
|-----------|-------------|--------------|-----------|
| `pass/mag` | `MIN_MAX_P999` | `True` | SAR magnitude is heavy-tailed; log-compression and robust range scaling. |
| `pass/raw_re_im` | `MIN_MAX_P999` | `False` | Complex amplitudes; percentile-based range without log. |
| `pass/norm_re_im` | `ROBUST_IQR` | `False` | Normalised complex channels; IQR robust to outliers. |
| `pass/phase` | `FIXED_DIV_PI` | `False` | Phase is uniformly distributed on $(-\pi, \pi]$; divide by $\pi$. |
| `ifg/mag` | `MIN_MAX_P999` | `True` | Same as `pass/mag`. |
| `ifg/raw_re_im` | `MIN_MAX_P999` | `False` | — |
| `ifg/norm_re_im` | `ROBUST_IQR` | `False` | — |
| `ifg/phase` | `FIXED_DIV_PI` | `False` | Interferometric phase; same analytical range. |
| `out/amp` | `MIN_MAX_P999` | `True` | Amplitude is non-negative and heavy-tailed. |
| `out/mu` | `MIN_MAX_P999` | `False` | Elevation mean; bounded range, no log. |
| `out/sigma` | `MIN_MAX_P999` | `True` | Elevation width is non-negative and skewed. |

---

## 8. Artifact Naming and Directory Layout

### Pre-processing Run (input)

```
{preprocessing_run_directory}/
    data/
        dataset.json                  ← layout manifest
        primary_reduced.npy           ← (1, Az, Rg) complex64
        secondaries_reduced.npy       ← (N_s, Az, Rg) complex64
        interferograms_reduced.npy    ← (N_s, Az, Rg) complex64
```

### Parameter Artifact (input)

```
{parameters_path}                     ← (3K, Az, Rg) float32
```

Typically located at `{param_pipeline_run}/params/parameters.npy`.

### Training Run (output)

```
{training_run_directory}/
    meta/
        dataset_creation_config.json  ← full DatasetConfiguration
        crop.json                     ← global_crop + per-split CropRegion
        patch.json                    ← per-split GridInfo
        normalization_stats.json      ← fitted (loc, scale) per channel
    logs/
        dataset_pipeline.log          ← structured run log
```

---

## 9. Inputs and Outputs Summary

### Inputs

| Artifact | Shape | dtype | Description |
|----------|-------|-------|-------------|
| `primary_reduced.npy` | `(1, Az, Rg)` | `complex64` | Pre-processed primary SLC. |
| `secondaries_reduced.npy` | `(N_s, Az, Rg)` | `complex64` | Co-registered secondary SLCs. |
| `interferograms_reduced.npy` | `(N_s, Az, Rg)` | `complex64` | Flattened interferograms. |
| `parameters.npy` | `(3K, Az, Rg)` | `float32` | Gaussian fit parameters $(a, \mu, \sigma) \times K$. |
| `dataset.json` | — | JSON | Layout manifest from pre-processing run. |
| `DatasetConfiguration` | — | Python | All hyperparameters. |

### Outputs

| Artifact | Type | Description |
|----------|------|-------------|
| `train_loader` | `DataLoader` | Shuffled, drop-last, normalised, augmented. |
| `val_loader` | `DataLoader` | Sequential, no drop, normalised, no augmentation. |
| `test_loader` | `DataLoader` | Sequential, no drop, normalised, no augmentation. |
| `datasets["train"]` | `PatchDataset` | Direct access to training `Dataset`. |
| `datasets["val"]` | `PatchDataset` | Direct access to validation `Dataset`. |
| `datasets["test"]` | `PatchDataset` | Direct access to test `Dataset`. |
| `meta/normalization_stats.json` | JSON | Per-channel $(loc, scale)$ for input and output. |
| `meta/dataset_creation_config.json` | JSON | Serialised `DatasetConfiguration`. |
| `meta/crop.json` | JSON | Global and per-split crop coordinates. |
| `meta/patch.json` | JSON | Per-split patch grid parameters. |

---

## 10. Canonical Usage

`DatasetPipeline` is consumed exclusively through `TrainingPipeline`, which is called from training entry points:

```python
# main/single_train.py  (simplified illustration)
from configuration.dataset_config import DatasetConfiguration, InputConfig, OutputConfig, PatchConfiguration
from pipelines.dataset_pipeline   import DatasetPipeline
from tools.split_regions          import SplitRegions

config = DatasetConfiguration(
    preprocessing_run_directory = Path("/runs/preproc/run_001"),
    split_regions               = SplitRegions({
        "train" : CropRegion(0,   800, 0, 1000),
        "val"   : CropRegion(800, 950, 0, 1000),
        "test"  : CropRegion(950, 1024, 0, 1000),
    }),
    parameters_path             = Path("/runs/params/run_001/params/parameters.npy"),
    patch                       = PatchConfiguration(size=(64, 64), stride=32),
    input_config                = InputConfig(use_interferograms=True),
    output_config               = OutputConfig(use_amplitude=True, use_mu=True, use_sigma=True),
    batch_size                  = 16,
    num_workers                 = 8,
    n_gaussians                 = 2,
)

pipeline = DatasetPipeline(config, training_run_directory=Path("/runs/train/run_001"))
train_loader, val_loader, test_loader, datasets = pipeline.run()

# Ready for training loop
for inputs, gt in train_loader:
    inputs = inputs.to("cuda")    # (B, C_in, 64, 64)  float32
    gt     = gt.to("cuda")        # (B, C_out, 64, 64) float32
    ...
```

**To reload statistics from a previous run** (e.g. for inference or fine-tuning):

```python
from pipelines.dataset_pipeline.normalize import Stats
from tools.logger import Logger

stats = Stats.load(Path("/runs/train/run_001/meta"), logger=logger)
normalizer = Normalizer(stats)
```

---

## 11. Public API Reference

### `DatasetPipeline` (`pipeline.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(config, training_run_directory, logger=None)` | Initialises all sub-components. No data I/O. |
| `run` | `() → (DataLoader, DataLoader, DataLoader, dict)` | Full pipeline execution; returns loaders and datasets. |

### `Layout` (`metadata.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(run_directory, logger, parameters_path)` | Parses `dataset.json`. |
| `artifact_path` | `(artifact_key: str) → Path` | Resolves logical key to absolute path. |

### `Cropper` (`crop.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(layout, split_regions, logger)` | Logs split geometry table. |
| `to_local_slices` | `(region: CropRegion) → (slice, slice)` | Translates global to local coordinates. |
| `load_split` | `(region: CropRegion) → dict[str, ndarray]` | Memory-maps and crops all artifact arrays. |

### `Patcher` (`patch.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `build` | `(spatial_size, patch_size, stride, ...) → Patcher` | Classmethod; computes full patch grid. |
| `extract` | `(array, idx) → ndarray` | Extracts patch at index with padding if needed. |

### `PatchDataset` (`load.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__len__` | `() → int` | Total number of patches. |
| `__getitem__` | `(idx) → (ndarray, ndarray)` | Full per-item pipeline: extract → represent → augment → normalise. |

### `StatsComputer` (`normalize.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `compute_input_stats` | `(dataset, logger, input_config, n_slaves, ...) → Stats` | Fits input normalisation from training data. |
| `compute_output_stats` | `(params_path, n_gaussians, output_config, ...) → Stats` | Fits output normalisation from parameter file. |

### `Normalizer` (`normalize.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `normalize_input` | `(tensor) → tensor` | Forward normalisation for inputs. |
| `normalize_output` | `(tensor) → tensor` | Forward normalisation for outputs. |
| `denormalize_input` | `(tensor) → tensor` | Inverse transform for inputs. |
| `denormalize_output` | `(tensor) → tensor` | Inverse transform for outputs (used at inference). |

### `Stats` (`normalize.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `save` | `(directory: Path) → Path` | Writes `normalization_stats.json`. |
| `load` | `(directory: Path, logger) → Stats` | Class method; reloads from JSON. |

### `Loader` (`load.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `build` | `(train_ds, val_ds, test_ds, batch_size, num_workers, ...) → (DL, DL, DL)` | Wraps datasets in configured DataLoaders. |

### `MetadataWriter` (`metadata.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `save_dataset_configuration` | `(config) → Path` | Serialises `DatasetConfiguration` to JSON. |
| `save_crop_metadata` | `(global_crop, splits) → Path` | Writes crop bounding boxes. |
| `save_patch_metadata` | `(grids: dict) → Path` | Writes per-split `GridInfo` dicts. |

### `SpatialAugmenter` (`augmentation.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(input_tensor, gt_params) → (ndarray, ndarray)` | Applies stochastic spatial transforms in-place. |
