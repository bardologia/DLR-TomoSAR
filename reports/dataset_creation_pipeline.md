# Dataset Creation Pipeline — Detailed Report

> Source: [pipelines/dataset_creation_pipeline/](../pipelines/dataset_creation_pipeline)

## 1. Purpose

Convert the three on-disk artifacts produced by the pre-processing and parameter-extraction stages into three PyTorch `DataLoader`s (train / val / test), using a configurable patch grid, complex-to-real channel representations, and online-Welford normalization statistics.

The pipeline is implemented in [pipelines/dataset_creation_pipeline/pipeline.py](../pipelines/dataset_creation_pipeline/pipeline.py) and configured by [DatasetCreationConfiguration](../configuration/dataset_config.py).

## 2. Inputs

The pipeline points at a previously-completed pre-processing run. From `<run>/data/dataset.json` it reads:

| Key                | Use                                                                        |
|--------------------|-----------------------------------------------------------------------------|
| `global_crop`      | Reference frame for converting global split coordinates to local slices.   |
| `artifacts.full_tomogram` | Tomogram cube → **target** (`(N_h, A, R)`).                          |
| `artifacts.input_tomogram`| Optional, not currently used inside this pipeline.                  |
| `artifacts.inputs` | Complex SAR pass stack → **input** (`(N_passes, A, R)` complex).            |

Plus the externally-supplied `parameters_path` (`(3K, A, R)`, the GT parameters from the param-extraction pipeline).

## 3. Top-level flow

[DatasetCreationPipeline.run()](../pipelines/dataset_creation_pipeline/pipeline.py#L88) executes:

1. Build the **train** dataset *without* normalization to gather statistics.
2. Compute [NormalizationStats.compute_from_dataset()](../pipelines/dataset_creation_pipeline/normalize.py#L150) on the train split (Welford one-pass) and persist them as `meta/normalization_stats.json`.
3. Re-build train / val / test datasets *with* the normalization stats wired in.
4. Build the three `DataLoader`s via [LoaderBuilder.build()](../pipelines/dataset_creation_pipeline/load.py#L88).
5. Persist three metadata files in `meta/`:
   - `dataset_creation_config.json` — full `asdict(config)` with the input config dict-encoded.
   - `crop.json` — global crop and per-split regions.
   - `patch.json` — per-split `PatchGridInfo` dictionaries.

## 4. Splits and crop arithmetic

[Cropper](../pipelines/dataset_creation_pipeline/crop.py#L11) translates **global pixel coordinates** to **local slices** within the pre-processing crop:

$$
s_a = \text{slice}(a_\text{split} - a_\text{global},\ a_\text{end-split} - a_\text{global}), \qquad
s_r = \text{slice}(r_\text{split} - r_\text{global},\ r_\text{end-split} - r_\text{global}).
$$

[Cropper.load_split()](../pipelines/dataset_creation_pipeline/crop.py#L25) `mmap`-loads each artifact (no copy), slices them with the same `(s_a, s_r)` and returns a dict `{inputs, parameters, tomogram}`.

## 5. Patch grid

[Patcher.build()](../pipelines/dataset_creation_pipeline/patch.py#L57) computes a regular $n_v \times n_h$ patch grid that fully covers the spatial extent $(H, W)$ for a chosen `(patch_size, stride)`.

Number of patches per axis:

$$
n_v = \begin{cases} 1 & H \le p_h \\ \left\lceil \dfrac{H - p_h}{s} \right\rceil + 1 & \text{otherwise} \end{cases},\qquad
n_h \text{ analogous on } W.
$$

Required padding so the grid lands exactly on the borders:

$$
\text{pad}_v = (p_h + (n_v-1)\,s) - H,\qquad \text{pad}_v^{\text{top}}=\lfloor \text{pad}_v/2 \rfloor,\ \text{pad}_v^{\text{bot}}=\text{pad}_v-\text{pad}_v^{\text{top}}.
$$

The pad mode is `"symmetric"` (reflective) when `use_reflective_padding=True`, otherwise `"constant"`. Padding is **applied lazily** inside [Patcher.extract()](../pipelines/dataset_creation_pipeline/patch.py#L96) so the cube stays as a memory-mapped view; only the requested patch is copied and padded with `np.pad`.

`PatchGridInfo` exposes `number_of_patches = n_v * n_h`, which is what `TomoPatchDataset.__len__` returns.

## 6. Channel representation pipeline

[InputConfig](../configuration/dataset_config.py#L37) selects which complex sources are converted to real channels and how. Three sources can be combined:

| Source           | Toggle               | Representation enum (per-pass channels)                                                      |
|------------------|----------------------|----------------------------------------------------------------------------------------------|
| Master SLC       | `use_master`         | `master_representation` (1 pass)                                                             |
| Slave SLCs       | `use_slaves`         | `slaves_representation` ($N-1$ passes)                                                       |
| Interferograms   | `use_interferograms` | `interferograms_representation` ($N-1$ pseudo-passes)                                        |

Interferograms are computed **on-the-fly per patch** as

$$
I_k = s_k \cdot \overline{s_m}\quad (k=2,\dots,N).
$$

Each `Representation` (defined in `tools.representation`) maps a complex tensor to a stack of real channels. The slot-kind table — used later for grouped normalization — is

| Representation       | Slot kinds                                       | Channels per pass |
|----------------------|--------------------------------------------------|-------------------|
| `REAL_IMAG`          | `raw_re_im`, `raw_re_im`                         | 2                 |
| `MAG_REAL_IMAG`      | `log_mag`, `norm_re_im`, `norm_re_im`            | 3                 |
| `MAG_ANGLE`          | `log_mag`, `phase`                               | 2                 |
| `MAG_RI_ANGLE`       | `log_mag`, `norm_re_im`, `norm_re_im`, `phase`   | 4                 |
| `ANGLE_ONLY`         | `phase`                                          | 1                 |
| `MAG_ONLY`           | `log_mag`                                        | 1                 |

Total input channels follow

$$
C_\text{in} = c_m + (N-1)\cdot(c_s + c_i),\qquad c_\bullet=\text{channels per pass for source }\bullet.
$$

## 7. `TomoPatchDataset`

[TomoPatchDataset.__getitem__](../pipelines/dataset_creation_pipeline/load.py#L50) for index `idx`:

1. `complex_patch = grid.extract(self.inputs, idx)` — shape `(N_passes, p_h, p_w)` complex.
2. `target_patch  = grid.extract(self.targets, idx)` — shape `(N_h, p_h, p_w)` complex.
3. `converted = input_config.build_tensor(complex_patch[None,...])[0]` — applies the master / slaves / interferograms conversion to produce real channels of shape `(C_in, p_h, p_w)`.
4. `gt_t = grid.extract(self.gt_parameters, idx)` — shape `(3K, p_h, p_w)`.
5. **Target construction** depends on `target_mode`:
   - `TargetMode.RAW`: target is `|target_patch|` (magnitude tomogram).
   - `TargetMode.GAUSSIAN_FIT`: target is the analytical reconstruction from the GT parameters,

     $$ T(h,a,r) = \sum_{k=1}^{K} A_k(a,r)\,\exp\!\left(-\frac{(h - \mu_k(a,r))^2}{2\sigma_k(a,r)^2 + 10^{-8}}\right), $$

     evaluated on the configured `x_axis` (length `N_h`).
6. **Optional normalization** — if `norm_stats` is provided, `Normalizer.normalize_input(input_tensor)` and (when output stats exist) `normalize_output(gt_t)` are applied.

Return tuple: `(input_tensor, target_tensor, gt_t)`.

## 8. Normalization

[NormalizationStats.compute_from_dataset()](../pipelines/dataset_creation_pipeline/normalize.py#L150) iterates the **training** dataset once:

- Optionally subsamples to `max_samples` indices (RNG seed 42).
- Per-channel **Welford online update** of `(count, mean, M2)` for inputs and (if `output_mode != DISABLED`) outputs:

  $$
  n \leftarrow n_a + n_b,\quad
  \delta = m_b - m_a,\quad
  m \leftarrow m_a + \delta\,\frac{n_b}{n},\quad
  M_2 \leftarrow M_{2,a} + M_{2,b} + \delta^2 \frac{n_a n_b}{n}.
  $$

- Variance estimator: $\sigma^2 = M_2 / \max(n-1, 1)$, with a floor of `1e-8` mapped to `1.0` to keep the resulting normalization a no-op for nearly-constant channels.

### 8.1 Modes

`InputNormalizationMode`:

- `PER_CHANNEL`: each input channel keeps its own (mean, std).
- `GROUPED`: channels sharing the same `(source/kind)` key (e.g. `pass/log_mag`, `ifg/phase`) are pooled using the parallel-Welford merge so all phase channels share one mean/std, all log-magnitude channels share another, etc. This is the central trick that makes the normalization configuration-invariant.

`OutputNormalizationMode`:

- `DISABLED` (default).
- `PER_CHANNEL` or `GROUPED` with role keys generated by [_build_output_channel_to_group()](../pipelines/dataset_creation_pipeline/normalize.py#L94) → `["a","mu","sig","p3",...]` repeated per Gaussian.

The resulting stats are persisted as

```json
{
  "input_mode" : "per_channel|grouped",
  "output_mode": "...",
  "input_stats" : {"channels":[{"name":"pass/log_mag","mean":-3.21,"std":1.4}, ...]},
  "output_stats": null | {"channels":[...]}
}
```

### 8.2 `Normalizer`

[Normalizer.normalize_input()](../pipelines/dataset_creation_pipeline/normalize.py#L289) constructs `(mean, std)` tensors with shape `(1,C,1,1)` (4-D batch) or `(C,1,1)` (3-D single sample), broadcasts and applies $z = (x-\mu)/\sigma$. `denormalize_*` are the inverse and are used at inference / evaluation time. Output normalization shape handling additionally supports 1-D parameter vectors.

## 9. Loaders

[LoaderBuilder.build()](../pipelines/dataset_creation_pipeline/load.py#L88) creates three `torch.utils.data.DataLoader`s with shared kwargs `(batch_size, num_workers, pin_memory, drop_last=False)`. Train shuffles when `shuffle_train=True`, val / test never shuffle.

## 10. Configuration recap

[DatasetCreationConfiguration](../configuration/dataset_config.py#L131) bundles:

- `preprocessing_run_directory : Path`
- `split_regions : SplitRegions` — three `CropRegion` objects (train/val/test) in **global** coordinates.
- `parameters_path : Optional[Path]` — output `.npy` from the param-extraction pipeline.
- `patch : PatchConfiguration(size=(64,64), stride=32, use_reflective_padding=True)`.
- `input_config : InputConfig` — see §6.
- `batch_size`, `num_workers`, `shuffle_train`, `pin_memory`.
- `input_normalization_mode`, `output_normalization_mode`.
- `target_mode : TargetMode` — `RAW` or `GAUSSIAN_FIT`.
- `x_axis : Optional[np.ndarray]` — height grid for `GAUSSIAN_FIT` reconstruction.
- `n_gaussians : int` — number of Gaussian components used by `GAUSSIAN_FIT`.

## 11. Failure modes & safeguards

- All artifacts are loaded with `mmap_mode="r"`; per-patch copies are bounded by patch size.
- `InputConfig.build_tensor` raises if **no** channels are produced (master disabled and no slaves).
- Normalization computation downgrades `OutputNormalizationMode` to `DISABLED` (with a warning) if the dataset does not expose GT params.
- `_collapse_to_groups` falls back to `_per_channel_stats` (with warning) if the slot-kind key list does not match the channel count, preventing silent shape mismatches.
- Group-stats merging uses Chan parallel-Welford to remain numerically stable across heterogeneous channel sizes.

## 12. Outputs returned

```python
(train_loader, val_loader, test_loader, datasets={"train":..,"val":..,"test":..})
```

`datasets[name]` exposes:

- `inputs`, `targets`, `gt_parameters` — references to the mmap views.
- `grid : Patcher` — for stitching during inference.
- `input_config`, `norm_stats`, `target_mode`, `x_axis`, `n_gaussians`.
- `passes`, `n_slaves`, `input_channels`, `target_channels`, `gt_channels` — used by the training pipeline to size the model heads.
