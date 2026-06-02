# Processing Pipeline — Technical Reference

**Package:** `pipelines.processing_pipeline`  
**Entry point:** `main/pre_process.py`  
**Last updated:** June 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Configuration Layer](#3-configuration-layer)
   - 3.1 [CropRegion](#31-cropregion)
   - 3.2 [TomogramConfiguration](#32-tomogramconfiguration)
   - 3.3 [ParallelConfiguration](#33-parallelconfiguration)
   - 3.4 [PathConfiguration](#34-pathconfiguration)
   - 3.5 [ProcessingConfiguration](#35-processingconfiguration)
4. [Component Responsibilities](#4-component-responsibilities)
   - 4.1 [ArtifactRegistry](#41-artifactregistry)
   - 4.2 [MetadataManager](#42-metadatamanager)
   - 4.3 [TomogramProcessor](#43-tomogramprocessor)
   - 4.4 [InterferogramBuilder](#44-interferogrambuilder)
   - 4.5 [ProcessingPipeline](#45-processingpipeline)
5. [Pipeline Execution Stages](#5-pipeline-execution-stages)
   - 5.1 [Stage 1 — Full Tomogram Generation](#51-stage-1--full-tomogram-generation)
   - 5.2 [Stage 2 — Reduced Tomogram Generation](#52-stage-2--reduced-tomogram-generation)
   - 5.3 [Stage 3 — Interferometric Stack Construction](#53-stage-3--interferometric-stack-construction)
6. [Mathematical Formulation](#6-mathematical-formulation)
   - 6.1 [Interferogram Formation](#61-interferogram-formation)
   - 6.2 [DEM Phase Removal](#62-dem-phase-removal)
   - 6.3 [Amplitude-Weighted Phasor](#63-amplitude-weighted-phasor)
7. [Parallelism and Crop Subdivision](#7-parallelism-and-crop-subdivision)
8. [Artifact Naming and Directory Layout](#8-artifact-naming-and-directory-layout)
   - 8.1 [Output Artifacts](#81-output-artifacts)
   - 8.2 [Metadata Outputs](#82-metadata-outputs)
   - 8.3 [Directory Structure](#83-directory-structure)
9. [Inputs and Outputs Summary](#9-inputs-and-outputs-summary)
10. [Canonical Usage — Entry Point](#10-canonical-usage--entry-point)
11. [Public API Reference](#11-public-api-reference)

---

## 1. Overview

The `processing_pipeline` package implements the **pre-processing stage** of the DLR-TomoSAR system. Its purpose is to transform raw FSAR (F-SAR airborne SAR instrument, DLR) data products into the structured numerical arrays required by downstream parameter estimation and neural-network inference pipelines.

Concretely, the pipeline produces seven output artifacts from a single configured run:

| Artifact | Description |
|---|---|
| `tomogram_full` | Full-stack 3-D SAR tomogram (Capon/other beamformer output) |
| `dem_full` | Digital Elevation Model derived from the full stack |
| `tomogram_reduced` | Tomogram generated from the reduced (training/inference) stack |
| `dem_reduced` | DEM derived from the reduced stack |
| `primary_reduced` | Complex SLC array of the primary (master) track |
| `secondaries_reduced` | Complex SLC array of all secondary (slave) tracks |
| `interferograms_reduced` | DEM-phase-corrected, amplitude-weighted complex interferograms |

The pipeline is designed to operate on **FSAR** data via the **PyRat** toolbox (DLR internal). All numerical outputs are saved as uncompressed NumPy `.npy` binary files to facilitate fast loading by subsequent processing steps.

---

## 2. Architecture

```
main/pre_process.py
│
│   ProcessingConfiguration
│   (crop, input_configs, output_configs,
│    parallel, paths, dataset_type,
│    stack identifiers, output tags)
│
└── ProcessingPipeline
      │
      ├── MetadataManager
      │     │
      │     └── ArtifactRegistry
      │           ├── ensure_directory_structure()
      │           ├── artifact_filenames()  →  dict[str, str]
      │           └── artifact_path()       →  Path
      │
      ├── TomogramProcessor
      │     ├── _divide_crop()         → list of azimuth sub-crops
      │     ├── _dispatch_workers()    → ProcessPoolExecutor (spawn)
      │     │     └── _run_pyrat()     [subprocess: tomo.fusartomo]
      │     │           └── writes HDF5 to tmp/TOMO/TOMO-SR/
      │     ├── _concatenate()         → (DEM, tomogram) numpy arrays
      │     └── _save() / _cleanup_temp()
      │
      └── InterferogramBuilder
            ├── _build_from_fsar()     → tomo.FuSARtomo object
            └── _compute_interferograms()
                  ├── pyrat.load.fsar         (primary RGI-SLC)
                  ├── pyrat.load.fsar         (secondary INF-SLC)
                  ├── pyrat.load.fsar_phadem  (DEM phase per secondary)
                  └── amplitude-weighted phasor formation
```

The `ProcessingPipeline` class is the sole orchestrator. It instantiates `MetadataManager`, `TomogramProcessor`, and `InterferogramBuilder` once at construction time, then executes the three processing stages sequentially inside `run()`. Each stage writes its result to disk and records provenance metadata before proceeding.

---

## 3. Configuration Layer

**Module:** `configuration/processing_config.py`

All pipeline behaviour is governed by a hierarchy of frozen-like dataclasses. No mutable global state exists outside these objects.

### 3.1 CropRegion

Imported from `tools.crop_region`. Encodes the 2-D spatial crop applied uniformly to all stages.

| Field | Type | Description |
|---|---|---|
| `azimuth_start` | `int` | First azimuth line (inclusive) |
| `azimuth_end` | `int` | Last azimuth line (exclusive) |
| `range_start` | `int` | First range sample (inclusive) |
| `range_end` | `int` | Last range sample (exclusive) |

**Key methods:**

- `as_tuple() → Tuple[int, int, int, int]` — Returns `(azimuth_start, azimuth_end, range_start, range_end)` in the format expected by PyRat's `crop=` argument.
- `as_identifier_string() → str` — Returns a filesystem-safe string encoding all four coordinates, used as a prefix in artifact filenames to ensure spatial traceability.

---

### 3.2 TomogramConfiguration

Parameterises a single call to the PyRat `fusartomo` beamformer. Two instances of this class may coexist within a `ProcessingConfiguration` (`input_configs` for the reduced stack, `output_configs` for the full stack).

| Field | Type | Default | Description |
|---|---|---|---|
| `fusar_project_path` | `str` | `""` | Absolute path to the F-SAR CSV project file listing all tracks |
| `base_directory` | `str` | `"/ste/rnd/"` | Root directory under which PyRat locates raw data products |
| `polarisation` | `str` | `"hv"` | SAR polarisation channel (e.g., `"hv"`, `"vv"`) |
| `track_selection` | `str` | `"*"` | Glob pattern selecting which tracks to include |
| `height_range` | `Tuple[float, float]` | `(-20.0, 80.0)` | Elevation search interval in metres for the beamformer |
| `filter_method` | `str` | `"Boxcar"` | Pre-processing spatial filter applied by PyRat (e.g., `"Boxcar"`) |
| `filter_arguments` | `dict` | `{"win": [20, 10]}` | Arguments forwarded to the filter (window sizes: `[azimuth, range]`) |
| `beamforming_method` | `str` | `"Capon"` | Spectral estimation method (e.g., `"Capon"`, `"BF"`) |
| `beamforming_arguments` | `list` | `[]` | Additional arguments forwarded to the beamformer |
| `max_crop_azimuth_width` | `int` | `1000` | Maximum azimuth dimension per PyRat subprocess call (governs parallelism granularity) |
| `apply_resampling` | `bool` | `False` | Enable PyRat resampling step |
| `apply_presumming` | `bool` | `False` | Enable PyRat pre-summing step |
| `max_amplitude_clip` | `float` | `1.25` | Upper clipping threshold applied to secondary SLC amplitudes before interferogram formation |

---

### 3.3 ParallelConfiguration

Controls the degree of parallelism at two distinct levels:

| Field | Type | Default | Description |
|---|---|---|---|
| `tomogram_workers` | `int` | `10` | Number of concurrent `ProcessPoolExecutor` worker processes; each handles one azimuth sub-crop |
| `pyrat_threads` | `int` | `15` | Number of threads passed to `pyrat_init()` within each worker process |

The total thread count at peak utilisation is bounded by `tomogram_workers × pyrat_threads`.

---

### 3.4 PathConfiguration

Manages the filesystem layout of a pipeline run. All paths are derived from a single `main_directory` root.

| Field | Type | Default | Description |
|---|---|---|---|
| `main_directory` | `Path` | `/ste/rnd/User/vice_vi/Dataset` | Root dataset directory |
| `pyrat_directory` | `Path` | `/ste/rnd/User/vice_vi/pyrat` | Location of the PyRat source tree (prepended to `sys.path`) |
| `data_subdirectory` | `str` | `"data"` | Subdirectory name for numerical artifact storage |
| `metadata_subdirectory` | `str` | `"meta"` | Subdirectory name for provenance records |
| `temporary_subdirectory` | `str` | `"tmp"` | Subdirectory for transient PyRat HDF5 outputs |
| `run_subdirectory` | `str \| None` | `None` | Isolating subdirectory for this run; auto-generated if `None` |

**Derived properties:**

- `run_directory → Path` — `main_directory / run_subdirectory`
- `data_directory → Path` — `run_directory / data_subdirectory`
- `metadata_directory → Path` — `run_directory / metadata_subdirectory`
- `temporary_directory → Path` — `run_directory / temporary_subdirectory`

---

### 3.5 ProcessingConfiguration

Root configuration object. Aggregates all sub-configurations and exposes derived tags used throughout the pipeline for artifact naming.

| Field | Type | Default | Description |
|---|---|---|---|
| `crop` | `CropRegion` | *(required)* | Global spatial crop applied to all stages |
| `input_configs` | `TomogramConfiguration` | default instance | Configuration for reduced-stack operations (tomogram + interferograms) |
| `output_configs` | `TomogramConfiguration \| None` | `None` | Configuration for full-stack tomogram; falls back to `input_configs` if `None` |
| `parallel` | `ParallelConfiguration` | default instance | Parallelism settings |
| `paths` | `PathConfiguration` | default instance | Filesystem layout |
| `dataset_type` | `str` | `"FSAR"` | Data source type; only `"FSAR"` is currently supported |
| `full_stack_identifier` | `str` | `"flaca"` | PyRat stack ID for the full baseline set |
| `reduced_stack_identifier` | `str` | `"flaca_2"` | PyRat stack ID for the reduced baseline set |
| `tomogram_output_tag` | `str` | `"Xtomo_id2X"` | User-defined label embedded in reduced-stack artifact names |
| `parameter_output_tag` | `str` | `"Xparams_id2X"` | User-defined label embedded in full-stack artifact names |

**Derived properties:**

```
tomogram_tag  = f"{crop.as_identifier_string()}_{reduced_stack_identifier}_{tomogram_output_tag}"
parameter_tag = f"{crop.as_identifier_string()}_{full_stack_identifier}_{parameter_output_tag}"
```

These tags serve as the primary traceability tokens embedded in every artifact filename and metadata record.

**`__post_init__` behaviour:** If `paths.run_subdirectory` is `None`, it is set to `f"run_{tomogram_tag}_{timestamp}"` where `timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")`, creating a unique, self-describing run directory for each execution.

---

## 4. Component Responsibilities

### 4.1 ArtifactRegistry

**Module:** `pipelines/processing_pipeline/artifacts.py`  
**Class:** `ArtifactRegistry`

The `ArtifactRegistry` is the single source of truth for output file naming and directory scaffolding. It encapsulates all conventions that relate a logical artifact type to a concrete filesystem path, ensuring that every other component obtains paths through a consistent, traceable interface.

**Constructor:**

```python
ArtifactRegistry(config: ProcessingConfiguration, logger: Logger)
```

**Public interface:**

| Method | Returns | Description |
|---|---|---|
| `ensure_directory_structure()` | `None` | Creates `data_directory`, `metadata_directory`, and `temporary_directory` (with `mkdir -p` semantics) |
| `artifact_filenames()` | `dict[str, str]` | Returns a mapping from logical artifact key to filename string |
| `artifact_path(artifact_type: ArtifactType)` | `Path` | Returns the fully-qualified `Path` for the given artifact type |

**Artifact type literal:**

```python
ArtifactType = Literal[
    "tomogram_full", "tomogram_reduced",
    "dem_full",      "dem_reduced",
    "primary_reduced",
    "secondaries_reduced",
    "interferograms_reduced",
]
```

**Naming conventions:**

| Artifact key | Filename template | Tag used |
|---|---|---|
| `tomogram_full` | `tomogram_full_{parameter_tag}.npy` | `parameter_tag` |
| `dem_full` | `dem_full_{parameter_tag}.npy` | `parameter_tag` |
| `tomogram_reduced` | `tomogram_reduced_{tomogram_tag}.npy` | `tomogram_tag` |
| `dem_reduced` | `dem_reduced_{tomogram_tag}.npy` | `tomogram_tag` |
| `primary_reduced` | `primary_reduced_{tomogram_tag}.npy` | `tomogram_tag` |
| `secondaries_reduced` | `secondaries_reduced_{tomogram_tag}.npy` | `tomogram_tag` |
| `interferograms_reduced` | `interferograms_reduced_{tomogram_tag}.npy` | `tomogram_tag` |

Full-stack artifacts (tomogram, DEM) embed the `parameter_tag` because they are produced from the full baseline configuration. Reduced-stack artifacts embed the `tomogram_tag` because they depend on the reduced stack identifier and crop.

---

### 4.2 MetadataManager

**Module:** `pipelines/processing_pipeline/metadata.py`  
**Class:** `MetadataManager`

The `MetadataManager` is the provenance interface of the pipeline. It wraps an `ArtifactRegistry` instance and provides methods to persist run configuration and per-stage processing metadata to disk in human-readable and machine-parseable formats.

**Constructor:**

```python
MetadataManager(config: ProcessingConfiguration, logger: Logger)
```

Internally constructs an `ArtifactRegistry` and delegates path resolution to it.

**Public interface:**

| Method | Returns | Description |
|---|---|---|
| `ensure_directory_structure()` | `None` | Delegates to `ArtifactRegistry.ensure_directory_structure()` |
| `artifact_path(artifact_type)` | `Path` | Delegates to `ArtifactRegistry.artifact_path()` |
| `save_stage_metadata(stage_name, identifier_tag, metadata_entries)` | `Path` | Writes a `meta_{stage_name}_{identifier_tag}.txt` file with key-value pairs |
| `save_pipeline_configuration()` | `Path` | Serialises the full `ProcessingConfiguration` to `config_state_{tomogram_tag}.json` |
| `save_dataset_layout()` | `Path` | Writes `dataset.json` mapping artifact filenames to the crop and tag context |

**Metadata file locations:** All metadata files are written to `paths.metadata_directory`.

---

### 4.3 TomogramProcessor

**Module:** `pipelines/processing_pipeline/tomogram.py`  
**Class:** `TomogramProcessor`

The `TomogramProcessor` generates a SAR tomogram and an associated DEM by invoking PyRat's `tomo.fusartomo` function. Because PyRat is not thread-safe at the process level and because the azimuth extent of a scene may significantly exceed practical per-call limits, the processor partitions the crop along the azimuth dimension and dispatches each partition to an independent worker process. Partial results are stored as HDF5 files by PyRat and subsequently concatenated into a single array pair.

**Constructor:**

```python
TomogramProcessor(config: ProcessingConfiguration, logger: Logger)
```

**Internal workflow:**

1. **`_create_temp()`** — Creates a uniquely-named temporary directory under `paths.temporary_directory` using `tempfile.mkdtemp`. This directory serves as the `dir=` output target for PyRat.

2. **`_divide_crop(tomogram_config)`** — Partitions the global azimuth range `[azimuth_start, azimuth_end)` into contiguous sub-crops of width at most `max_crop_azimuth_width`. Each sub-crop spans the full range extent of the global crop. Returns a list of `(az_start, az_end, range_start, range_end)` tuples.

3. **`_dispatch_workers(subsections, stack_identifier, tomogram_config, temporary_directory)`** — Submits one task per sub-crop to a `ProcessPoolExecutor` using a `spawn` multiprocessing context. Each task calls the module-level function `_run_pyrat()` in a fresh subprocess. The spawn context is mandatory because PyRat modifies global C-library state incompatible with forked processes.

4. **`_run_pyrat()` (module-level function)** — Executed inside each worker process. Initialises the PyRat runtime (`pyrat_init`), then calls `tomo.fusartomo` with the sub-crop parameters. PyRat writes the tomogram and DEM for that sub-crop into an HDF5 file under `{temporary_directory}/TOMO/TOMO-SR/`. A numeric suffix index (zero-padded to four digits) is embedded in the output filename to preserve ordering.

5. **`_concatenate(temporary_directory)`** — Reads all HDF5 files from `TOMO/TOMO-SR/` in sorted filename order. Extracts the `"DEM"` and `"tomogram"` datasets from each file. Concatenates DEM chunks along axis 0 (azimuth) and tomogram chunks along axis 1 (azimuth, given the tomogram's 3-D layout `[height, azimuth, range]`).

6. **`_save()`** — Writes the combined arrays to the artifact paths as `.npy` files.

7. **`_cleanup_temp()`** — Recursively deletes the temporary directory in a `finally` block, ensuring no residual HDF5 data persists regardless of success or failure.

**Public method:**

```python
run(
    tomogram_path    : Path,
    dem_path         : Path,
    stack_identifier : str,
    tomogram_config  : TomogramConfiguration,
) -> Tuple[Path, Path]
```

---

### 4.4 InterferogramBuilder

**Module:** `pipelines/processing_pipeline/interferogram.py`  
**Class:** `InterferogramBuilder`

The `InterferogramBuilder` constructs the complex interferometric measurement stack from which physical scattering parameters are subsequently estimated. It loads raw SLC (Single-Look Complex) imagery via PyRat, removes topographic phase contributions using pre-computed DEM phase screens, and forms normalised, amplitude-weighted complex interferograms for each secondary track.

**Constructor:**

```python
InterferogramBuilder(config: ProcessingConfiguration, logger: Logger)
```

On construction, the PyRat source tree (`paths.pyrat_directory`) is prepended to `sys.path` if not already present. PyRat itself is not initialised until `build()` is called, deferring the Qt/OpenGL runtime to the execution phase.

**Public interface:**

| Method | Description |
|---|---|
| `build(crop_tuple)` | Loads and computes all arrays; returns `(primary, secondaries, interferograms)` |
| `run(crop_tuple, primary_path, secondaries_path, interferograms_path)` | Calls `build()` then persists all three arrays; returns shape tuples |

**Internal data loading sequence** (within `_compute_interferograms`):

For the **primary** track, PyRat loads the co-registered SLC from the `RGI-SLC` product directory.

For each **secondary** track $k \in \{1, \ldots, N\}$:

1. The SLC is loaded from the `INF-SLC` product directory (already co-registered to the primary geometry).
2. The DEM phase screen $\phi_{\text{DEM},k}$ is loaded via `pyrat.load.fsar_phadem`.
3. The interferogram is formed as described in [Section 6](#6-mathematical-formulation).
4. Intermediate arrays are explicitly deleted and `gc.collect()` is invoked to manage memory pressure during large-scene processing.

All secondary SLCs and interferograms are collected in Python lists and stacked with `np.stack(..., axis=0)`, yielding arrays of shape `(N, azimuth, range)` where $N$ is the number of secondary tracks.

---

### 4.5 ProcessingPipeline

**Module:** `pipelines/processing_pipeline/pipeline.py`  
**Class:** `ProcessingPipeline`

The `ProcessingPipeline` is the top-level orchestrator. It does not perform any SAR processing itself; instead, it sequences the three processing stages, manages the logger lifecycle, and aggregates output paths into a single return structure.

**Constructor:**

```python
ProcessingPipeline(config: ProcessingConfiguration, logger: Logger | None = None)
```

If no `logger` is provided, a default `Logger` instance writing to `{run_directory}/logs/preprocessing.log` is created automatically.

**`run()` method:**

```python
run() -> dict[str, Path]
```

Executes the full pipeline and returns a dictionary mapping each logical artifact name to its resolved output `Path`. The keys are identical to those defined by `ArtifactType` plus `"run_directory"`.

**Execution order:**

1. Serialize pipeline configuration to `config_state_{tomogram_tag}.json`
2. Generate full tomogram + DEM → `_stage_tomogram("full")`
3. Generate reduced tomogram + DEM → `_stage_tomogram("reduced")`
4. Build interferometric stack → `_stage_inputs()`
5. Write `dataset.json` layout manifest

---

## 5. Pipeline Execution Stages

### 5.1 Stage 1 — Full Tomogram Generation

**Orchestrated by:** `ProcessingPipeline._stage_tomogram("full")`  
**Processor:** `TomogramProcessor`  
**Configuration:** `config.output_config` (which resolves to `output_configs` if set, else `input_configs`)  
**Stack identifier:** `config.full_stack_identifier`  
**Artifact tag:** `config.parameter_tag`

This stage generates a tomographic reconstruction over the **full** flight-pass baseline set. The full stack typically spans all available acquisitions, providing the maximum interferometric diversity and thus the highest resolution in the elevation (height) dimension. The resulting tomogram is the reference product from which physical scattering parameters (e.g., ground topography, vegetation height) are extracted by downstream parameter estimation pipelines.

**Outputs:**
- `tomogram_full_{parameter_tag}.npy` — 3-D power spectral estimate, shape `[n_heights, azimuth, range]`
- `dem_full_{parameter_tag}.npy` — 2-D DEM, shape `[azimuth, range]`

**Stage metadata** (`meta_tomogram_full_{parameter_tag}.txt`) records:

```
tomo_full      : <absolute path>
crop           : [az_start, az_end, r_start, r_end]
FuSARproject   : <project CSV path>
id             : <full_stack_identifier>
basedir        : <base_directory>
polarisation   : <polarisation>
select         : <track_selection>
range          : [height_min, height_max]
filter         : <filter_method>
method         : <beamforming_method>
win            : [filter_win_az, filter_win_range]
```

---

### 5.2 Stage 2 — Reduced Tomogram Generation

**Orchestrated by:** `ProcessingPipeline._stage_tomogram("reduced")`  
**Processor:** `TomogramProcessor`  
**Configuration:** `config.input_configs`  
**Stack identifier:** `config.reduced_stack_identifier`  
**Artifact tag:** `config.tomogram_tag`

This stage generates a tomographic reconstruction over the **reduced** baseline set — a deliberately smaller subset of acquisitions that matches the input geometry of the neural network or parameter estimator. The reduced stack is the operational configuration: it defines which tracks and which spatial crop are used during both dataset generation and inference.

The beamformer settings (method, filter, height range) for this stage are drawn from `input_configs`, which may differ from `output_configs`. In the canonical usage example, both configurations share the same FSAR parameters; the distinction matters when, for example, the full stack uses a wider height search range than the reduced stack.

**Outputs:**
- `tomogram_reduced_{tomogram_tag}.npy` — 3-D power spectral estimate, shape `[n_heights, azimuth, range]`
- `dem_reduced_{tomogram_tag}.npy` — 2-D DEM, shape `[azimuth, range]`

---

### 5.3 Stage 3 — Interferometric Stack Construction

**Orchestrated by:** `ProcessingPipeline._stage_inputs()`  
**Processor:** `InterferogramBuilder`  
**Configuration:** `config.input_configs`, `config.crop`  
**Stack identifier:** `config.reduced_stack_identifier`  
**Artifact tag:** `config.tomogram_tag`

This stage constructs the three fundamental measurement arrays consumed by the neural network:

1. **Primary SLC** — the complex backscatter signal of the master acquisition.
2. **Secondary SLCs** — the complex backscatter signals of all slave acquisitions, co-registered to the primary geometry.
3. **Interferograms** — DEM-phase-corrected, amplitude-weighted complex cross-products between the primary and each secondary.

The interferometric stack encodes the coherence and phase structure of the multi-baseline SAR signal, from which vertical scatterer distribution profiles can be estimated. See [Section 6](#6-mathematical-formulation) for the exact mathematical construction.

**Outputs:**
- `primary_reduced_{tomogram_tag}.npy` — shape `[azimuth, range]`, dtype `complex64`
- `secondaries_reduced_{tomogram_tag}.npy` — shape `[N, azimuth, range]`, dtype `complex64`
- `interferograms_reduced_{tomogram_tag}.npy` — shape `[N, azimuth, range]`, dtype `complex64`

where $N$ is the number of secondary tracks.

**Stage metadata** (`meta_inputs_{tomogram_tag}.txt`) additionally records the shapes of all three arrays, the crop parameters, the FuSAR project path, the stack identifier, the base directory, the polarisation, the track selection, and the dataset type.

---

## 6. Mathematical Formulation

### 6.1 Interferogram Formation

Let $s_1 \in \mathbb{C}^{A \times R}$ denote the primary SLC image (master acquisition), and let $s_k \in \mathbb{C}^{A \times R}$ for $k = 1, \ldots, N$ denote the $N$ secondary SLC images (slave acquisitions), where $A$ and $R$ denote the azimuth and range dimensions respectively.

### 6.2 DEM Phase Removal

Each secondary image $s_k$ is affected by a topographic phase component $\phi_{\text{DEM},k}(a,r)$, which arises from the non-zero normal baseline and the scene topography. This phase is computed externally by PyRat from a reference DEM and subtracted in the complex domain:

$$
\tilde{s}_k(a,r) = s_k(a,r) \cdot e^{\,j\,\phi_{\text{DEM},k}(a,r)}
$$

After this correction, the residual phase of $\tilde{s}_k$ relative to $s_1$ is dominated by the volumetric scattering response, making it suitable as input to tomographic inversion.

### 6.3 Amplitude-Weighted Phasor

The interferometric measurement $d_k \in \mathbb{C}^{A \times R}$ for the $k$-th baseline is defined as an amplitude-weighted unit-phasor:

$$
\gamma_k(a,r) = \frac{s_1(a,r) \cdot \overline{\tilde{s}_k(a,r)}}{\left|s_1(a,r) \cdot \overline{\tilde{s}_k(a,r)}\right| + \epsilon}
$$

$$
d_k(a,r) = \left|\tilde{s}_k(a,r)\right|^{\dagger} \cdot \gamma_k(a,r)
$$

where:

- $\overline{(\cdot)}$ denotes complex conjugation
- $\epsilon = 10^{-30}$ is a small regularisation constant preventing division by zero
- $\left|\tilde{s}_k\right|^{\dagger}$ denotes the amplitude of the DEM-corrected secondary, clipped at `max_amplitude_clip` (default 1.25) prior to this step

The resulting complex array $d_k$ encodes the **phase** of the cross-product phasor (i.e., the interferometric phase after DEM removal) in its argument, and the **amplitude** of the DEM-corrected secondary in its modulus. This representation preserves both the coherence-bearing phase information and a proxy for backscatter intensity simultaneously, without the additional noise floor introduced by the primary amplitude.

The full interferometric stack is assembled as:

$$
\mathbf{D} \in \mathbb{C}^{N \times A \times R}, \quad \mathbf{D}[k, :, :] = d_k
$$

---

## 7. Parallelism and Crop Subdivision

Large scene extents can exceed the memory capacity manageable by a single PyRat process, and serialised processing of many azimuth lines is prohibitively slow. The `TomogramProcessor` addresses this through azimuth-axis partitioning combined with process-level parallelism.

**Subdivision strategy:**

Given a total azimuth extent $A_{\text{total}} = \texttt{azimuth\_end} - \texttt{azimuth\_start}$ and a maximum per-call width $W_{\max} = \texttt{max\_crop\_azimuth\_width}$, the azimuth range is divided into $\lceil A_{\text{total}} / W_{\max} \rceil$ contiguous, non-overlapping sub-crops:

$$
\text{sub-crops} = \left\{\, (a_i,\; a_{i+1},\; r_0,\; r_1) \;\middle|\; a_i = \texttt{azimuth\_start} + i \cdot W_{\max},\;\; a_{i+1} = \min(a_i + W_{\max},\; \texttt{azimuth\_end}) \,\right\}
$$

Each sub-crop spans the full range extent `[range_start, range_end)`.

**Dispatch mechanism:**

Sub-crops are submitted to a `concurrent.futures.ProcessPoolExecutor` with `max_workers = tomogram_workers` using Python's `spawn` start method. The spawn context is required because:

1. PyRat calls into C/Qt shared libraries that maintain global process-level state incompatible with `fork`.
2. `LD_LIBRARY_PATH` must be explicitly set to include the conda environment's `lib/` directory before the shared libraries are loaded.

Each worker receives the full `sys.path` of the parent process to ensure that both the PyRat package and the DLR-TomoSAR project are importable without reinstallation.

**Concatenation:**

PyRat names its HDF5 output files with the user-supplied `suffix` argument (set to the zero-padded sub-crop index). After all workers complete, files are read in **sorted** filename order to guarantee correct spatial assembly. DEM chunks are concatenated along **axis 0** (azimuth), and tomogram chunks are concatenated along **axis 1** (the azimuth axis of a `[height, azimuth, range]` array).

In the canonical usage example (`main/pre_process.py`):

$$
W_{\max} = \left\lfloor \frac{16000 - 1000}{16} \right\rfloor = 937 \;\text{azimuth lines per worker}
$$

This divides the 15 000-line scene into 16 sub-crops processed across up to 10 concurrent workers.

---

## 8. Artifact Naming and Directory Layout

### 8.1 Output Artifacts

All numerical artifacts are written to `paths.data_directory` as uncompressed NumPy binary files (`.npy`, `allow_pickle=False`). Tags embedded in filenames ensure that artifacts from different runs, crops, or stack configurations do not collide.

**Tag composition:**

```
tomogram_tag  = {crop_identifier}_{reduced_stack_id}_{tomogram_output_tag}
parameter_tag = {crop_identifier}_{full_stack_id}_{parameter_output_tag}
```

Example (canonical Traun dataset run):

```
tomogram_tag  = "az1000_16000_r500_4000_dtmf_Xtomo_id2X"
parameter_tag = "az1000_16000_r500_4000_1_Xparams_id2X"
```

### 8.2 Metadata Outputs

All provenance records are written to `paths.metadata_directory`.

| File | Contents |
|---|---|
| `config_state_{tomogram_tag}.json` | Full serialisation of `ProcessingConfiguration` (via `dataclasses.asdict`); all `Path` objects converted to strings |
| `meta_tomogram_full_{parameter_tag}.txt` | Key-value provenance for Stage 1 |
| `meta_tomogram_reduced_{tomogram_tag}.txt` | Key-value provenance for Stage 2 |
| `meta_inputs_{tomogram_tag}.txt` | Key-value provenance for Stage 3 (includes array shapes) |
| `dataset.json` | JSON manifest mapping artifact keys to filenames plus crop and tag context |

**`dataset.json` schema:**

```json
{
  "global_crop"   : [az_start, az_end, r_start, r_end],
  "dataset_type"  : "FSAR",
  "tomogram_tag"  : "...",
  "parameter_tag" : "...",
  "artifacts"     : {
    "tomogram_full"          : "tomogram_full_<parameter_tag>.npy",
    "tomogram_reduced"       : "tomogram_reduced_<tomogram_tag>.npy",
    "dem_full"               : "dem_full_<parameter_tag>.npy",
    "dem_reduced"            : "dem_reduced_<tomogram_tag>.npy",
    "primary_reduced"        : "primary_reduced_<tomogram_tag>.npy",
    "secondaries_reduced"    : "secondaries_reduced_<tomogram_tag>.npy",
    "interferograms_reduced" : "interferograms_reduced_<tomogram_tag>.npy"
  }
}
```

### 8.3 Directory Structure

```
{main_directory}/
└── run_{tomogram_tag}_{timestamp}/       ← run_directory (auto-created)
    ├── data/                             ← data_directory
    │   ├── tomogram_full_{param_tag}.npy
    │   ├── dem_full_{param_tag}.npy
    │   ├── tomogram_reduced_{tomo_tag}.npy
    │   ├── dem_reduced_{tomo_tag}.npy
    │   ├── primary_reduced_{tomo_tag}.npy
    │   ├── secondaries_reduced_{tomo_tag}.npy
    │   ├── interferograms_reduced_{tomo_tag}.npy
    │   └── dataset.json
    ├── meta/                             ← metadata_directory
    │   ├── config_state_{tomo_tag}.json
    │   ├── meta_tomogram_full_{param_tag}.txt
    │   ├── meta_tomogram_reduced_{tomo_tag}.txt
    │   └── meta_inputs_{tomo_tag}.txt
    ├── logs/
    │   └── preprocessing.log
    └── tmp/                              ← temporary_directory (deleted on completion)
        └── tomo_{random}/
            └── TOMO/
                └── TOMO-SR/
                    ├── 0000.hd5
                    ├── 0001.hd5
                    └── ...
```

---

## 9. Inputs and Outputs Summary

| Stage | Primary Input | External Dependency | Outputs |
|---|---|---|---|
| Initialisation | `ProcessingConfiguration` | Filesystem | Run directory tree, `config_state_*.json` |
| Stage 1 — Full tomogram | FSAR CSV project, full stack ID | PyRat `tomo.fusartomo` | `tomogram_full_*.npy`, `dem_full_*.npy`, `meta_tomogram_full_*.txt` |
| Stage 2 — Reduced tomogram | FSAR CSV project, reduced stack ID | PyRat `tomo.fusartomo` | `tomogram_reduced_*.npy`, `dem_reduced_*.npy`, `meta_tomogram_reduced_*.txt` |
| Stage 3 — Interferometric stack | FSAR CSV project, reduced stack ID, crop | PyRat `load.fsar`, `load.fsar_phadem` | `primary_reduced_*.npy`, `secondaries_reduced_*.npy`, `interferograms_reduced_*.npy`, `meta_inputs_*.txt` |
| Finalisation | All stage outputs | — | `dataset.json` |

**Return value of `ProcessingPipeline.run()`:**

```python
{
    "tomogram_full"          : Path,  # full tomogram array
    "tomogram_reduced"       : Path,  # reduced tomogram array
    "dem_full"               : Path,  # full DEM array
    "dem_reduced"            : Path,  # reduced DEM array
    "primary_reduced"        : Path,  # primary SLC array
    "secondaries_reduced"    : Path,  # secondary SLC stack
    "interferograms_reduced" : Path,  # interferogram stack
    "run_directory"          : Path,  # root of this run's output tree
}
```

---

## 10. Canonical Usage — Entry Point

**File:** `main/pre_process.py`

The canonical invocation configures a run over the **Traun** F-SAR dataset:

| Parameter | Value |
|---|---|
| Azimuth crop | `[1000, 16000)` (15 000 lines) |
| Range crop | `[500, 4000)` (3 500 samples) |
| FuSAR project | `/ste/rnd/User/sera_se/17sartom-traun_L.csv` |
| Polarisation | `hv` |
| Beamforming | Capon |
| Spatial filter | Boxcar, window `[30, 10]` (azimuth × range) |
| Height search range | `[-20.0, 80.0]` m |
| Full stack ID | `"1"` |
| Reduced stack ID | `"dtmf"` |
| Azimuth width per worker | `(16000 - 1000) // 16 = 937` lines |

Both `input_configs` and `output_configs` share all FSAR parameters; `output_configs` additionally sets `height_range = (-20.0, 80.0)` explicitly (matching the default), so the full-stack tomogram uses the same elevation search interval as the reduced-stack tomogram.

**Minimal invocation:**

```python
from configuration.processing_config import CropRegion, ProcessingConfiguration, TomogramConfiguration
from pipelines.processing_pipeline.pipeline import ProcessingPipeline

config = ProcessingConfiguration(
    crop                     = CropRegion(1000, 16000, 500, 4000),
    input_configs            = TomogramConfiguration(
        fusar_project_path     = "/ste/rnd/User/sera_se/17sartom-traun_L.csv",
        polarisation           = "hv",
        beamforming_method     = "Capon",
        filter_method          = "Boxcar",
        filter_arguments       = {"win": [30, 10]},
        max_crop_azimuth_width = 937,
    ),
    dataset_type             = "FSAR",
    full_stack_identifier    = "1",
    reduced_stack_identifier = "dtmf",
)

outputs = ProcessingPipeline(config).run()
```

---

## 11. Public API Reference

### `ProcessingPipeline`

```python
class ProcessingPipeline:
    def __init__(
        self,
        config : ProcessingConfiguration,
        logger : Logger | None = None,
    ) -> None: ...

    def run(self) -> dict[str, Path]: ...
```

### `TomogramProcessor`

```python
class TomogramProcessor:
    def __init__(self, config: ProcessingConfiguration, logger: Logger) -> None: ...

    def run(
        self,
        tomogram_path    : Path,
        dem_path         : Path,
        stack_identifier : str,
        tomogram_config  : TomogramConfiguration,
    ) -> Tuple[Path, Path]: ...
```

### `InterferogramBuilder`

```python
class InterferogramBuilder:
    def __init__(self, config: ProcessingConfiguration, logger: Logger) -> None: ...

    def build(
        self,
        crop_tuple : Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def run(
        self,
        crop_tuple           : Tuple[int, int, int, int],
        primary_path         : Path,
        secondaries_path     : Path,
        interferograms_path  : Path,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]: ...
```

### `MetadataManager`

```python
class MetadataManager:
    def __init__(self, config: ProcessingConfiguration, logger: Logger) -> None: ...

    def artifact_path(self, artifact_type: ArtifactType) -> Path: ...

    def save_stage_metadata(
        self,
        stage_name       : str,
        identifier_tag   : str,
        metadata_entries : dict[str, str],
    ) -> Path: ...

    def save_pipeline_configuration(self) -> Path: ...
    def save_dataset_layout(self) -> Path: ...
```

### `ArtifactRegistry`

```python
class ArtifactRegistry:
    def __init__(self, config: ProcessingConfiguration, logger: Logger) -> None: ...

    def ensure_directory_structure(self) -> None: ...
    def artifact_filenames(self) -> dict[str, str]: ...
    def artifact_path(self, artifact_type: ArtifactType) -> Path: ...
```
