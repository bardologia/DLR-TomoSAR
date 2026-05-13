# Inference Pipeline — Detailed Report

> Source: [pipelines/inference_pipeline/](../pipelines/inference_pipeline)

## 1. Purpose

Given a finished training run directory, the inference pipeline:

1. Loads the best (or named) checkpoint, optionally swapping in EMA weights.
2. Re-builds the dataset for one split (`train` / `val` / `test`).
3. Runs the model patch-by-patch and **stitches** the predictions into full-resolution cubes using overlap-add with a configurable window.
4. Computes a battery of **global** and **per-pixel** metrics comparing the prediction to both the GT-Gaussian curves and the raw tomogram.
5. Saves: profile panels (best / worst / random pixels), per-pixel metric maps (MSE, R², peak-error), parameter maps, range / azimuth / elevation slice composites, SSIM-vs-slice plots, GIF walks, a JSON metric dump, and a Markdown report.

The pipeline is implemented in [pipelines/inference_pipeline/pipeline.py](../pipelines/inference_pipeline/pipeline.py) and is configured by [InferenceConfig](../pipelines/inference_pipeline/config.py).

## 2. Configuration surface — [InferenceConfig](../pipelines/inference_pipeline/config.py)

| Field                   | Default              | Notes                                                                                  |
|-------------------------|----------------------|----------------------------------------------------------------------------------------|
| `run_directory`         | required             | Training run directory — must exist (`__post_init__` validates).                       |
| `output_subdir`         | `None`               | Sub-folder inside `<run>/inference/`; default = current timestamp.                    |
| `device`                | `"cuda"`             | Device for inference.                                                                 |
| `use_ema`               | `True`               | Whether to apply EMA shadow weights from the checkpoint before inference.              |
| `checkpoint_name`       | `"best_model.pt"`    | File picked from `<run>/`.                                                            |
| `split`                 | `"test"`             | One of `train`/`val`/`test`.                                                          |
| `batch_size`            | `None`               | Falls back to the dataset config’s batch size.                                        |
| `num_workers`           | `4`                  | DataLoader workers.                                                                   |
| `stitch_window`         | `"hann"`             | One of `hann`, `triangular`, `uniform`.                                               |
| `cube_dtype`            | `"float32"`          | Storage dtype of the stitched cubes.                                                  |
| `save_cubes`            | `True`               | Persist `.npy` cubes alongside metric maps.                                           |
| `n_best/n_worst/n_random_profiles` | `12`/`12`/`12` | Profile-panel selection counts.                                                  |
| `n_range/azimuth/elevation_slices` | `5`/`5`/`5`    | Number of cross-section figures per axis.                                         |
| `gif_axes`              | `["elevation"]`      | Subset of `elevation`/`range`/`azimuth`.                                              |
| `gif_fps/max_frames/dpi`| `12`/`150`/`110`     | Animation parameters.                                                                 |
| `cmap_intensity`        | `"viridis"`          | Colormap for tomogram intensity panels.                                               |
| `cmap_error`            | `"magma"`            | Colormap for error / MSE / peak maps.                                                 |
| `fig_dpi`/`save_dpi`    | `150`/`300`          | Display vs. exported DPI.                                                              |
| `seed`                  | `0`                  | NumPy + torch seed.                                                                   |
| `log_level`             | `"INFO"`             | Logger level.                                                                         |

## 3. Top-level flow

[InferencePipeline.run()](../pipelines/inference_pipeline/pipeline.py#L40):

1. Resolve `output_dir = <run>/inference/<output_subdir or timestamp>` and create `figures/`, `animations/`, `logs/`.
2. Seed `torch` + `numpy`; apply Matplotlib style.
3. `RunDirectoryLoader.load(...)` → `Run` (model + dataset + grid + metadata) — see §4.
4. `Predictor.run_inference()` → `InferenceResult` containing stitched cubes and pixel metrics — see §5.
5. Compute slice indices spaced linearly at the 10 %–90 % quantiles of each axis.
6. `compute_global_metrics(...)` over **all** elevation / range / azimuth slices; persist `metrics.json`.
7. Generate the figure set (§6) and the GIFs (§7).
8. `assemble_report(...)` writes `<output_dir>/report.md`.

## 4. Run loading — [RunDirectoryLoader](../pipelines/inference_pipeline/loader.py)

Discovers and parses these files inside `run_directory`:

- `meta/run_summary.json` → `model_name`, `in_channels`, `out_channels`, `x_axis_length`.
- `docs/trainer_config.json` → reconstructs `TrainerConfig` (and notably `gaussian` for `n_gaussians = out_channels // 3`).
- `meta/dataset_creation_config.json` → reconstructs `DatasetCreationConfiguration`.
- `meta/normalization_stats.json` → `NormalizationStats` (input + optional output).
- `meta/crop.json` and `meta/patch.json` → `global_crop`, per-split `CropRegion`s, per-split `PatchGridInfo`.
- `<checkpoint_name>` → `model.load_state_dict(...)`. If `use_ema and ema_state_dict`, calls `ema.apply_to(model)` so subsequent inference uses the shadow weights.

It then rebuilds the requested split’s `TomoPatchDataset` and returns a `Run` dataclass with everything `Predictor` and the plotters need (`model`, `loader`, `grid`, `x_axis`, `n_gaussians`, `has_noise_head`, `azimuth_offset`, `range_offset`, `dataset_config`, `checkpoint_meta`).

## 5. Stitched prediction — [Predictor.run_inference()](../pipelines/inference_pipeline/predictor.py)

Iterates the dataloader in `torch.no_grad()`. For every patch:

1. **Forward** → `pred_params` of shape `(B, 3K, p_h, p_w)`.
2. **Reconstruct** the predicted curves and the GT curves on `x_axis`:

   $$ C(h,a,r) = \sum_{k=1}^{K} A_k(a,r)\,\exp\!\left(-\frac{(h - \mu_k(a,r))^2}{2\sigma_k(a,r)^2 + 10^{-8}}\right). $$

   GT params are first denormalized via `norm_stats.denormalize_output(...)` if output normalization is active.

3. **Per-pixel metrics** (computed on each patch independently against both targets — GT-Gaussian curves and raw tomogram magnitudes):

   $$
   \text{MSE}(a,r) = \frac{1}{N_h}\sum_h \big(\hat C - C\big)^2,\qquad
   \text{MAE}(a,r) = \frac{1}{N_h}\sum_h |\hat C - C|,
   $$

   $$
   R^2(a,r) = 1 - \frac{\sum_h(\hat C - C)^2}{\sum_h(C - \bar C)^2 + 10^{-8}},\qquad
   \cos(a,r) = \frac{\sum_h \hat C\,C}{\|\hat C\|\,\|C\| + 10^{-8}}.
   $$

   The peak-location error is

   $$ \Delta_\text{peak}(a,r) = |\arg\max_h \hat C(h,a,r)\ -\ \arg\max_h C(h,a,r)|. $$

### 5.1 Overlap-add stitching — [Stitcher](../pipelines/inference_pipeline/stitching.py)

A 2-D blending window $W(p_h, p_w)$ is built with [_make_window()](../pipelines/inference_pipeline/stitching.py):

- `hann`: $W = w_h \otimes w_w$ with $w_n = 0.5\,(1 - \cos(2\pi n / (N-1)))$.
- `triangular`: triangular outer product.
- `uniform`: all ones.

For each batched patch at position $(p_a, p_r)$ in the grid, the accumulator updates per-element:

$$
S(h, a, r) \mathrel{+}= W(a-p_a,\ r-p_r)\,\hat{C}(h, a-p_a, r-p_r),\qquad
N(a, r) \mathrel{+}= W(a-p_a, r-p_r),
$$

where the patch coordinates account for the `pad_top`/`pad_left` offsets recorded in `PatchGridInfo`. After all batches:

$$
\bar S(h,a,r) = S(h,a,r) / (N(a,r) + 10^{-8}),
$$

and the central crop removes the (`pad_top`, `pad_left`, `pad_bot`, `pad_right`) padding so the output cube exactly matches the split’s `(N_h, A_split, R_split)` extent. The same accumulator pattern is also used for **all** per-pixel metric maps, ensuring overlap regions are weighted-averaged identically to the curves.

### 5.2 Persisted artifacts

`Predictor` writes `<output_dir>/cubes/`:

- `pred_curves.npy`, `gt_curves.npy`, `raw_curves.npy` — `(N_h, A_split, R_split)`.
- `params_pred.npy`, `params_gt.npy` — `(3K, A_split, R_split)`.
- `pixel_mse.npy`, `pixel_mae.npy`, `pixel_r2.npy`, `pixel_cosine.npy`, `pixel_peak_err_idx.npy` (vs GT).
- `pixel_mse_raw.npy`, `pixel_r2_raw.npy`, `pixel_cosine_raw.npy` (vs raw tomogram).

The `InferenceResult` namedtuple-style object also exposes `azimuth_offset`, `range_offset` (from `split_region`) so plots can render absolute pixel coordinates.

## 6. Global metrics — [compute_global_metrics()](../pipelines/inference_pipeline/metrics.py)

Computes:

- **Curve-level**: mean / std / median MSE, MAE, RMSE, R², PSNR vs both GT and raw.
- **Per-pixel summary statistics**: 5/25/50/75/95-percentiles of every per-pixel metric map.
- **Peak-location error**: distribution converted to elevation units via `x_axis_step`.
- **SSIM per slice** along all three axes via [compute_ssim_slices()](../pipelines/inference_pipeline/metrics.py): for every slice index `i` in each axis it computes

  $$ \text{SSIM}\!\big(\hat S_i,\ S_i\big),\qquad \hat S_i,\,S_i \in \mathbb{R}^{N_x\times N_y}, $$

  using `skimage.metrics.structural_similarity` against both GT and raw cubes. Keys are written as `ssim_<axis>_<slice_index>` and `ssim_<axis>_raw_<slice_index>`.

The full dict is dumped via [write_metrics_json()](../pipelines/inference_pipeline/report.py).

[select_pixels_by_metric()](../pipelines/inference_pipeline/metrics.py): for the per-pixel MSE map, returns `n_best` lowest, `n_worst` highest, and `n_random` (RNG-seeded) `(a, r)` pixel indices used to drive the profile panels.

## 7. Figures — [plots.py](../pipelines/inference_pipeline/plots.py)

The pipeline calls these in order:

| Function                         | Output                                                     |
|----------------------------------|------------------------------------------------------------|
| `plot_profile_panel`             | `profiles_best.png`, `profiles_worst.png`, `profiles_random.png` — multi-panel overlay of raw / GT / pred profiles with individual Gaussian components and per-pixel metric annotations. |
| `plot_pixel_metric_map`          | `pixel_mse_map.png` (log-scale), `pixel_r2_map.png` (RdYlGn, 2–98 % clipping), `pixel_peak_map.png`. |
| `plot_metric_histogram`          | `metric_histograms.png` — 6-curve log-scale histogram of MSE/R²/cosine vs both GT and raw. |
| `plot_param_maps`                | `param_maps.png` — spatial maps of all `3K` Gaussian parameters; if GT params exist, the figure is rendered side-by-side. |
| `plot_tomogram_slice`            | One file per range / azimuth slice: 6-panel figure (raw / GT / pred + |pred−GT| + |pred−raw|) annotated with the slice’s SSIM. |
| `plot_elevation_intensity_slice` | Same layout for elevation slices (`(A,R)` planes at fixed $h$). |
| `plot_ssim_curves`               | `ssim_range.png`, `ssim_azimuth.png`, `ssim_elev.png` — line plots of SSIM vs slice index for pred×GT and pred×raw. |

All file paths are accumulated into `figure_paths` and forwarded to the report.

## 8. Animations — [animation.py](../pipelines/inference_pipeline/animation.py)

For each axis in `cfg.gif_axes`, [make_walk_gif()](../pipelines/inference_pipeline/animation.py):

- Down-samples the chosen axis to ≤ `gif_max_frames` evenly-spaced indices.
- Renders 6-panel figures (raw / GT / pred / |pred−GT| / |pred−raw|, plus a thumbnail) for each frame.
- Encodes the sequence into a GIF with `imageio` at `fps` frames per second.

Failures are caught and logged as warnings — they do not abort the pipeline.

## 9. Report — [assemble_report()](../pipelines/inference_pipeline/report.py)

Produces a structured Markdown file with:

1. Run summary (model, channels, split, EMA, patch geometry, preprocessing dir).
2. Inference configuration table.
3. Checkpoint metadata (epoch, best val loss, etc.) extracted from the checkpoint dict.
4. Headline metrics with definitions ($R^2$, SSIM, peak-error, …).
5. Full metric table (key/value).
6. Embedded profile panels.
7. Spatial diagnostics (pixel maps, histograms, param maps).
8. Tomogram slice galleries with their SSIM annotations.
9. SSIM-vs-slice plots.
10. Animations.
11. Final artifact listing.

## 10. Failure modes & safeguards

- `InferenceConfig.__post_init__` validates `split`, `stitch_window`, and `gif_axes`, and raises `FileNotFoundError` on a missing run directory.
- `RunDirectoryLoader` re-uses the **exact** dataset config from training, so patch geometry and channel layout cannot drift between training and inference.
- The stitcher applies a `+ 1e-8` floor in the weight division so border pixels with low coverage do not produce `inf` / `NaN`.
- GIF generation is wrapped in a `try/except` per axis.
- All cube-shaped outputs accept memory-mapped backing arrays, so very large scenes can be processed without keeping the full prediction in RAM.

## 11. Outputs returned

`InferencePipeline.run()` returns the path to `report.md`. Side-effects in `<output_dir>/`:

- `metrics.json` — global metrics dict.
- `figures/` — every PNG produced.
- `animations/` — every GIF.
- `cubes/` — `.npy` cubes (if `save_cubes`).
- `logs/` — Rich console log files.
- `report.md` — the full Markdown report.
