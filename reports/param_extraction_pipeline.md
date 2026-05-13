# Parameter Extraction Pipeline — Detailed Report

> Source: [pipelines/param_extraction_pipeline/](../pipelines/param_extraction_pipeline)

## 1. Purpose

Given a **3-D tomogram cube** $T \in \mathbb{C}^{N_h \times N_a \times N_r}$ (height × azimuth × range) produced by the pre-processing pipeline, the parameter-extraction stage fits an analytical multi-Gaussian model to every $(a, r)$ elevation profile and saves the resulting parameter cube. These parameters constitute the **ground-truth labels** consumed by the training pipeline.

The pipeline is implemented in [pipelines/param_extraction_pipeline/pipeline.py](../pipelines/param_extraction_pipeline/pipeline.py) and configured by [ExtractionConfig](../configuration/param_extraction_config.py).

## 2. Mathematical model

Each elevation profile is modelled as a sum of $K$ Gaussians (see [GaussianModel.multi_gaussian](../pipelines/param_extraction_pipeline/gaussian_model.py#L9)):

$$
\hat{f}(h \mid \theta) = \sum_{k=1}^{K} A_k \, \exp\!\left( -\frac{(h - \mu_k)^2}{2\sigma_k^2 + \varepsilon} \right),
\qquad \theta = (A_1,\mu_1,\sigma_1,\dots,A_K,\mu_K,\sigma_K) \in \mathbb{R}^{3K}.
$$

Numerical safety: the exponent is clipped to $[-100, 0]$ so that very narrow Gaussians do not produce `NaN`s, and a regulariser $\varepsilon = 10^{-10}$ is added inside $2\sigma^2$ to prevent division by zero.

The **initial parameter guess** in [estimate_initial_parameters](../pipelines/param_extraction_pipeline/gaussian_model.py#L23) is a greedy peak picker:

1. Smooth the absolute profile with a length-5 uniform filter.
2. Set $\sigma_\text{guess} = (h_\text{max} - h_\text{min}) / (4K)$.
3. Repeat $K$ times: pick the highest sample $(\hat{h}_k, \hat{A}_k)$, append $(\hat{A}_k, \hat{h}_k, \sigma_\text{guess})$, then zero-out the working profile in $|h - \hat{h}_k| < 2\sigma_\text{guess}$.

## 3. Configuration surface

[ExtractionConfig](../configuration/param_extraction_config.py) provides:

- `processed_data_path` — the pre-processing run directory; helper [discover_tomogram_path()](../configuration/param_extraction_config.py) reads `data/dataset.json` and returns the path of the **`full_tomogram`** artifact.
- `discover_height_range()` — returns the configured `(h_min, h_max)` (parsed from the meta file produced by stage 1).
- `output_directory`, `output_prefix`, `output_suffix` — define the output `.npy` filename via `parameters_npy_path`.
- `parameter_workers` — number of worker processes used in the parallel chunked extraction.
- `fit_settings : FitSettings` — chooses the optimisation backend and its hyper-parameters:
  - `number_of_gaussians : int` — value of $K$.
  - `max_fit_iterations : int` — `maxfev` for SciPy `curve_fit` / `maxiter` for `minimize`.
  - `fitting_method : str` — either `"adaptive"` or `"mle"` (used for logging).
  - `fit_config : FitMode.Adaptive | FitMode.MLE` — the polymorphic carrier of method-specific options.

### 3.1 `FitMode.Adaptive`

Drives [FittingMethods.fit_curve()](../pipelines/param_extraction_pipeline/fitting.py#L46), a least-squares solver via `scipy.optimize.curve_fit` (Trust-Region / Levenberg-Marquardt with bounds):

| Field                  | Meaning                                                                                       |
|------------------------|------------------------------------------------------------------------------------------------|
| `threshold_factor`     | Multiplicative threshold on the profile maximum below which samples are zeroed (denoising).    |
| `truncation_index`     | Sample beyond which the profile is forced to zero (range/cutoff).                              |
| `initial_guess`        | Optional fixed `θ₀`; if `None`, the greedy peak picker is used.                                |
| `lower_bounds`/`upper_bounds` | Optional per-parameter bounds; defaults to $A \ge 0$, $\mu \in [h_\text{min}, h_\text{max}]$, $\sigma \in [10^{-6}, h_\text{max}-h_\text{min}]$. |

### 3.2 `FitMode.MLE`

Drives [FittingMethods.fit_mle()](../pipelines/param_extraction_pipeline/fitting.py#L131): a **Poisson negative log-likelihood** minimisation via `L-BFGS-B`. Treating the (non-negative) intensity profile $y$ as Poisson-distributed with rate $\mu(\theta) = \hat f(h\mid\theta)$, the cost minimised per profile is

$$
\mathcal{L}(\theta) = \sum_h \big[ \mu(h;\theta) - y(h)\,\log(\mu(h;\theta) + \epsilon) \big].
$$

| Field           | Meaning                                                                       |
|-----------------|--------------------------------------------------------------------------------|
| `epsilon`       | Floor inside the `log` term to avoid `log(0)`.                                |
| `ftol`, `gtol`  | Convergence tolerances on function and gradient norm respectively.            |
| `lower_bounds`/`upper_bounds`/`initial_guess` | As in `Adaptive`; bounds become a list of `(lo, hi)` pairs for L-BFGS-B. |

Both backends share a fallback strategy: on `RuntimeError`/`ValueError`/`np.linalg.LinAlgError` the parameters of the **previous successfully-fit pixel** are reused, and the failure is counted.

## 4. Top-level execution flow

[ParamExtractionPipeline.run()](../pipelines/param_extraction_pipeline/pipeline.py#L46) performs two stages:

```
_stage_extract_and_save()                ── data/<prefix>_<suffix>.npy
metadata_manager.save_run_metadata()     ── output_directory/param_extraction_meta.json
```

Stage 1 delegates to [ParameterExtractor.run()](../pipelines/param_extraction_pipeline/fitting.py#L351) which returns a NumPy array of shape $(3K,\ N_a,\ N_r)$ stacked along the range axis.

## 5. Parallel chunked extraction

[ParameterExtractor._parallel_extraction()](../pipelines/param_extraction_pipeline/fitting.py#L292):

1. **Range chunking** — split the $N_r$ range bins into `parameter_workers` contiguous chunks of size `chunk_size = N_r // workers`; the last worker absorbs the remainder.
2. **Function dispatch** — pick the worker function from the static `REGISTRY`:

   ```python
   {FitMode.Adaptive: fit_curve, FitMode.MLE: fit_mle}
   ```

3. **Progress queue** — a `multiprocessing.Manager().Queue()` carries one `1` from each worker every time a range bin is finished.
4. **Listener thread** — a daemon thread drains the queue and advances a Rich progress bar (`logger.track`).
5. **Pool execution** — `mp.Pool(processes=workers).starmap(target_function, tasks)` runs the chunks in parallel; each worker `mmap`s the tomogram (so memory stays low and pages are shared).

After joining the pool, results from all workers are concatenated:

| Quantity            | Aggregation                                                       |
|---------------------|--------------------------------------------------------------------|
| `fitted_slices`     | `extend` — list of `(3K, N_a)` arrays, one per range bin.         |
| `total_failed`      | sum of `failed_fits` across workers.                              |
| `total_attempted`   | sum of `attempted_fits` across workers.                           |
| `average_quality`   | $\bar{R^2} = (\sum q_i) / (\sum n_i)$ (NaN-safe).                  |

## 6. Per-profile inner loop

For each range bin and each azimuth pixel (in [fit_curve](../pipelines/param_extraction_pipeline/fitting.py#L92) and [fit_mle](../pipelines/param_extraction_pipeline/fitting.py#L173)):

1. `absolute_profile = |T[:, a, r]|` cast to `float64`.
2. (Adaptive only) **Threshold + truncation**:

   $$ p(h) = \begin{cases} |T(h)| & |T(h)| > \tau \cdot \max(|T|) \\ 0 & \text{otherwise} \end{cases},
      \quad p(h)=0 \ \forall h\ge h_\text{trunc}. $$

3. Build `initial_parameters` from `configured_initial` or the greedy estimator.
4. **Skip rule** — if `max(profile) < 1e-7`, write the initial parameters directly without optimisation (no signal).
5. Otherwise run the optimiser inside `warnings.catch_warnings()` and `np.errstate(invalid="ignore")` to silence the noisy SciPy convergence warnings.
6. Compute the per-profile coefficient of determination

   $$ R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}, $$

   accumulate into `quality_sum` / `quality_count` if finite.
7. Store the parameters, update `previous_valid_parameters` for the fallback path.

The per-bin output is `(3K, N_a)`. After all bins are processed `np.stack(...,axis=-1)` produces the final cube of shape $(3K, N_a, N_r)$ that is saved with `np.save(..., allow_pickle=False)`.

## 7. Reported quality

`ParameterExtractor.run()` logs:

```
Failed fits: <failed>/<attempted> (<pct>%)
Average fit quality (R²): <value or N/A>
```

These values are **not** persisted — only the JSON sidecar from [ExtractionMetadataManager.save_run_metadata()](../pipelines/param_extraction_pipeline/metadata.py#L17) is written, containing:

- timestamp
- source tomogram path & height range
- output directory / prefix / suffix
- `number_of_gaussians`, `fitting_method`, `max_fit_iterations`
- `asdict(fit_config)` (so the bounds and method-specific kwargs are reproducible)
- `parameter_workers`
- `parameters_npy` (basename of the saved array)

## 8. Output schema

The `.npy` array shape is

$$
\Theta \in \mathbb{R}^{3K \times N_a \times N_r},\qquad \Theta[3k+0,a,r]=A_k,\ \Theta[3k+1,a,r]=\mu_k,\ \Theta[3k+2,a,r]=\sigma_k.
$$

This convention is hard-coded everywhere in the project (loss masks, normalization grouping, model heads). The role assignment `[A, μ, σ]` is mirrored by [Metrics.reconstruct_gaussians()](../pipelines/training_pipeline/metrics.py) and by the [GaussianConfig.params_per_gaussian = 3](../configuration/training_config.py).

## 9. Failure modes & safeguards

- **All-zero / very weak profiles** are short-circuited (no SciPy call), avoiding spurious failures.
- **Per-pixel exceptions** in either backend fall back to the last successful parameters, ensuring spatial continuity.
- **Bound construction** clips $\sigma$ to `1e-6` lower bound, which prevents `curve_fit` from collapsing onto a delta.
- **Memory** — workers `mmap` the tomogram instead of loading it; per-bin slices are `del`-ed and `gc.collect()` is called after stacking.
- **Reproducibility** — when `initial_guess` is fixed in the config, no peak-picking randomness is introduced.

## 10. Outputs returned

```python
{
  "parameters_npy"   : Path,   # data/<prefix>_<suffix>.npy
  "metadata"         : Path,   # output_directory/param_extraction_meta.json
  "output_directory" : Path,
  "source_tomogram"  : Path,
}
```

The `parameters_npy` path is what the dataset pipeline uses as the third volumetric input alongside `inputs` and `full_tomogram`.
