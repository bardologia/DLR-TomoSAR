# Pre-Processing Pipeline — Detailed Report

> Source: [pipelines/pre_processing_pipeline/](../pipelines/pre_processing_pipeline)

## 1. Purpose

The pre-processing pipeline turns raw F-SAR / UAVSAR project data on disk into the three NumPy artifacts that every downstream stage of the project consumes:

| Artifact          | Filename pattern                                  | Content                                                                                |
|-------------------|---------------------------------------------------|----------------------------------------------------------------------------------------|
| `full_tomogram`   | `tomofull_<parameter_tag>.npy`                    | High-quality SAR tomogram cube generated from the **full pass stack** (used as GT).    |
| `input_tomogram`  | `tomofull_<tomogram_tag>.npy`                     | Lower-quality tomogram from the **reduced pass stack** (used for sanity / inference).  |
| `inputs`          | `inputs_<tomogram_tag>.npy`                       | Complex SAR pass stack (master + N×slaves + N×interferograms) for the reduced stack.   |

The pipeline is implemented in [pipelines/pre_processing_pipeline/pipeline.py](../pipelines/pre_processing_pipeline/pipeline.py) and is orchestrated by the dataclass [PreProcessingConfiguration](../configuration/preprocessing_config.py).

## 2. Top-level flow

[PreProcessingPipeline.run()](../pipelines/pre_processing_pipeline/pipeline.py#L42) executes three stages in order, each writing one artifact and a sidecar metadata file in `meta/`:

```
save_pipeline_configuration()           ── meta/config_state_<tomogram_tag>.json
_stage_full_tomogram()                  ── data/tomofull_<parameter_tag>.npy
_stage_input_tomogram()                 ── data/tomofull_<tomogram_tag>.npy
_stage_inputs()                         ── data/inputs_<tomogram_tag>.npy
save_dataset_layout()                   ── data/dataset.json
```

After every stage [_clear_memory()](../pipelines/pre_processing_pipeline/pipeline.py#L39) calls `gc.collect()` to release the GBs of complex data held by PyRat.

## 3. Configuration surface

The pipeline is parameterised by [PreProcessingConfiguration](../configuration/preprocessing_config.py#L62) which aggregates:

- **Crop geometry** — [CropRegion(azimuth_start, azimuth_end, range_start, range_end)](../configuration/preprocessing_config.py).
- **Two `TomogramConfiguration` blocks** ([input_configs](../configuration/preprocessing_config.py#L17) for the reduced stack and an optional `output_configs` for the full stack) carrying:
  - `fusar_project_path`, `base_directory`, `polarisation` (e.g. `"hv"`), `track_selection` (PyRat selector string).
  - `height_range = (h_min, h_max)` — elevation extent of the beamformed cube.
  - `filter_method` + `filter_arguments` (e.g. `"Boxcar"` with `win=[20, 10]`).
  - `beamforming_method` + `beamforming_arguments` — `"Capon"`, `"Fourier"`, `"MUSIC"`, etc.
  - `max_crop_azimuth_width` — azimuth split limit for the parallel runner.
  - `apply_resampling`, `apply_presumming` — PyRat geometric pre-treatments.
- **Parallelism** — [PreProcessingParallelConfiguration](../configuration/preprocessing_config.py#L31): `tomogram_workers` (subprocesses) and `pyrat_threads` (threads inside each subprocess).
- **Path layout** — [PathConfiguration](../configuration/preprocessing_config.py#L37) deriving `run_directory`, `data_directory`, `metadata_directory`, `temporary_directory` from a single `main_directory`. If `run_subdirectory` is `None` it is auto-set in `__post_init__` to `run_<tomogram_tag>_<timestamp>`.
- **Identifiers** — `dataset_type` (`"FSAR"` or `"UAVSAR"`), `full_stack_identifier`, `reduced_stack_identifier`, plus tag templates that are concatenated in `tomogram_tag` and `parameter_tag` properties.

## 4. Stage 1 — Full tomogram (GT)

Implemented by [TomogramProcessor.run()](../pipelines/pre_processing_pipeline/tomogram.py#L150) called from [_stage_full_tomogram](../pipelines/pre_processing_pipeline/pipeline.py#L62) with `stack_identifier = full_stack_identifier` and `tomogram_config = output_config`.

### 4.1 Crop subdivision

`max_crop_azimuth_width` bounds the azimuth extent that PyRat is asked to handle in one shot. The crop is split contiguously by [_divide_crop()](../pipelines/pre_processing_pipeline/tomogram.py#L80):

$$
\Delta_a = a_\text{end} - a_\text{start},\quad
n_{\text{sec}} = \left\lceil \frac{\Delta_a}{W_\text{max}} \right\rceil
$$

producing `n_sec` `(az_start_i, az_end_i, rg_start, rg_end)` quadruples that share the range extent.

### 4.2 Parallel PyRat dispatch

[_dispatch_workers()](../pipelines/pre_processing_pipeline/tomogram.py#L96) submits one task per subsection to a `ProcessPoolExecutor` using a **fork** multiprocessing context (so that the lazy PyRat import does not have to be re-pickled). Each task runs the top-level function [_run_pyrat()](../pipelines/pre_processing_pipeline/tomogram.py#L18), which:

1. Inserts the configured PyRat root into `sys.path`.
2. Calls `pyrat_init(debug=True, nthreads=pyrat_threads, silent=True)`.
3. Invokes `pyrat.tomo.fusartomo(...)` with the full configuration plus a per-subsection `suffix=f"{subsection_index:04d}"` and `dir=temporary_directory`.
4. Validates that `<tmp>/TOMO/TOMO-SR/` is not empty — raising a `RuntimeError` if PyRat produced nothing.

The temp directory is created with `tempfile.mkdtemp(prefix="tomo_", dir=paths.temporary_directory)` and removed by [_cleanup_temp_dir()](../pipelines/pre_processing_pipeline/tomogram.py#L146) inside a `finally` block.

### 4.3 HDF5 concatenation

[_concatenate_tomos()](../pipelines/pre_processing_pipeline/tomogram.py#L114) walks the sorted `TOMO/TOMO-SR/*.hd5` files. For each it reads the shapes of `DEM` and `tomogram`. It then allocates two combined NumPy arrays whose shapes are:

| Array      | Shape                                                                    |
|------------|---------------------------------------------------------------------------|
| `DEM`      | $(\sum_i \text{az}_i,\ \dots)$                                           |
| `tomogram` | $(N_\text{height},\ \sum_i \text{az}_i,\ N_\text{range})$                |

Each partial file is `read_direct(...)` into the slice `[:, az_offset : az_offset + az_i, :]` of the combined cube. This avoids loading the whole cube of every subprocess in memory.

### 4.4 Save

`np.save(output_path, tomogram_array)` (the DEM is intentionally discarded — only the tomogram is persisted), followed by stage metadata via [MetadataManager.save_stage_metadata()](../pipelines/pre_processing_pipeline/metadata.py#L34).

## 5. Stage 2 — Input tomogram

Identical to Stage 1, but with `stack_identifier = reduced_stack_identifier` and `tomogram_config = input_configs`. The “reduced stack” typically uses fewer passes (e.g. 4–6 instead of 16) so that the resulting cube mimics what the model would see at inference time.

## 6. Stage 3 — Complex pass stack (`inputs`)

This stage is performed by [InterferogramBuilder](../pipelines/pre_processing_pipeline/interferogram.py#L13) and dispatches on `dataset_type`.

### 6.1 UAVSAR branch

`_build_from_uavsar()` simply calls `pyrat.load.uavsar(...)` on the reduced track selection and stacks the SLCs:

```text
shape = (N_passes, H, W),  dtype = complex
```

### 6.2 F-SAR branch

`_build_from_fsar()` instantiates a `pyrat.tomo.FuSARtomo` object (which discovers the master and slaves automatically) and calls [_compute_interferograms()](../pipelines/pre_processing_pipeline/interferogram.py#L70).

For each secondary pass $k$:

1. Load the secondary `INF-SLC` and the corresponding DEM phase $\phi^{\text{DEM}}_k$.
2. Compute the **deramped secondary**:

   $$ \tilde{s}_k = s_k\, e^{j\,\phi^{\text{DEM}}_k} $$

3. Compute the unit-modulus interferometric phasor between master $s_m$ and the deramped secondary, scaled by the secondary amplitude (clipped to $[0, 1.25]$):

   $$ A_k = \min(|s_k|,\ 1.25), \qquad
      \phi_k = \frac{s_m \cdot \overline{\tilde{s}_k}}{|s_m \cdot \overline{\tilde{s}_k}| + 10^{-30}} $$

4. The complex interferogram is

   $$ I_k = A_k \cdot \phi_k \in \mathbb{C}^{H\times W}. $$

The final stack is `[s_m, I_1, …, I_{N-1}]` cast to `complex64`. Stage 3 always uses the **reduced** track selection.

### 6.3 Save

`np.save(output_path, np.ascontiguousarray(stack))`. The on-disk shape is `(N_passes, H, W)` and the metadata records crop, saved shape, project path, base directory, polarisation, selector, and dataset type.

## 7. Artifact registry

[ArtifactRegistry](../pipelines/pre_processing_pipeline/artifacts.py#L13) centralises the *naming convention* so that every stage and every downstream pipeline resolves the same files:

```python
{
  "full_tomogram"  : f"tomofull_{parameter_tag}.npy",
  "input_tomogram" : f"tomofull_{tomogram_tag}.npy",
  "inputs"         : f"inputs_{tomogram_tag}.npy",
}
```

`existence_map()` is used at startup to log a human-readable “FOUND / MISSING” table per artifact.

## 8. Metadata files

| File                                        | Producer                                                                                    | Content                                                          |
|---------------------------------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| `meta/config_state_<tag>.json`              | [save_pipeline_configuration](../pipelines/pre_processing_pipeline/metadata.py#L51)         | Full `asdict(PreProcessingConfiguration)` dump.                  |
| `meta/meta_tomofull_full_<param_tag>.txt`   | [save_stage_metadata](../pipelines/pre_processing_pipeline/metadata.py#L34)                 | Plain-text key/value lines for the full-stack tomogram stage.    |
| `meta/meta_tomofull_input_<tomo_tag>.txt`   | same                                                                                        | Same for the reduced-stack tomogram.                             |
| `meta/meta_inputs_<tomo_tag>.txt`           | same                                                                                        | Crop, saved shape, project, base dir, polarisation, selector.    |
| `data/dataset.json`                         | [save_dataset_layout](../pipelines/pre_processing_pipeline/metadata.py#L67)                 | `global_crop`, `dataset_type`, both tags, and artifact filenames — read by the dataset pipeline. |

## 9. Failure modes & safeguards

- **Empty PyRat output** — detected per worker; raises `RuntimeError` with the offending suffix and crop.
- **Memory pressure** — every stage explicitly `del`s heavy arrays and calls `gc.collect()`; the secondary loop in `_compute_interferograms` deletes per-pass tensors right after concatenation.
- **Path ordering** — `sorted(partial_files_directory.iterdir())` guarantees that `0000`, `0001`, … subsections are concatenated in azimuth order.

## 10. Outputs returned

```python
{
  "full_tomogram"  : Path,
  "input_tomogram" : Path,
  "inputs"         : Path,
  "run_directory"  : Path,
}
```

These four paths are everything the parameter-extraction pipeline and the dataset-creation pipeline need.
