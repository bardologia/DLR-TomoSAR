# Training Pipeline — Detailed Report

> Source: [pipelines/training_pipeline/](../pipelines/training_pipeline)

## 1. Purpose

End-to-end orchestration that takes a `DatasetCreationConfiguration`, builds the requested model architecture, and trains it to predict the parameters $\Theta = \{(A_k, \mu_k, \sigma_k)\}_{k=1}^{K}$ of a multi-Gaussian elevation profile model from real-channel SAR patches. Training combines AMP, gradient accumulation, EMA shadow weights, learning-rate warmup, a configurable LR scheduler, gradient clipping, early stopping, periodic checkpointing, and an 11-term composite loss.

The pipeline is implemented in [pipelines/training_pipeline/pipeline.py](../pipelines/training_pipeline/pipeline.py); the heavy lifting is in [trainer.py](../pipelines/training_pipeline/trainer.py).

## 2. Top-level flow

[TrainingPipeline.run()](../pipelines/training_pipeline/pipeline.py#L60):

1. Pull `gaussian_cfg` from `trainer_config` and propagate `n_default_gaussians` to `dataset_config.n_gaussians`.
2. If `target_mode == GAUSSIAN_FIT` and no `x_axis` is set, build it as `np.linspace(x_min, x_max, N_h)` from the full-tomogram height count.
3. `DatasetCreationPipeline.run()` → `(train_loader, val_loader, test_loader, datasets)`.
4. Compute `in_channels = train_dataset.input_channels`, `out_channels = 3K`.
5. `model = get_model(model_name, in_channels=..., out_channels=..., image_size?=patch_height)`.
6. Persist `trainer_config.json` and `run_summary.json` via [TrainingRunMetadata](../pipelines/training_pipeline/metadata.py#L17).
7. Instantiate `Trainer(model, x_axis, config, run_dir, logger, norm_stats=...)` and call `trainer.train(train, val, test)`.
8. Always call `run_metadata.close()` and `logger.close()` in a `finally`.

`TrainingRunMetadata` provisions `run_<model>_<ts>/` containing `tensorboard/`, `docs/`, `logs/`, `images/`, `meta/`. It mutates `trainer_config.io.*` to point at these subdirectories and creates the `SummaryWriter`.

## 3. `Trainer` — initialization

[Trainer.__init__](../pipelines/training_pipeline/trainer.py#L29) wires together every collaborator (one section per concept):

| Component                  | Purpose                                                                               |
|----------------------------|----------------------------------------------------------------------------------------|
| `Tracker`                  | TensorBoard scalar / histogram / metric / activation logger.                          |
| `AdamW(param_groups, …)`   | Optimizer with per-group `lr`/`wd` from `model.config.get_param_groups(model)`.       |
| `Warmup`                   | Linear LR warmup from `start_factor` → `1.0` over `warmup_steps`.                     |
| `EMA`                      | Shadow weights with decay $\bar\theta\leftarrow d\,\bar\theta+(1-d)\,\theta$.         |
| `EarlyStopping`            | Patience + `min_delta`; optional restore-best.                                        |
| `Scheduler`                | Wraps any of 9 PyTorch LR schedulers behind a unified interface.                      |
| `Metrics`                  | Per-batch curve metrics + Gaussian reconstruction.                                    |
| `Loss`                     | Multi-term composite loss (see §6).                                                   |
| `Checkpoint`               | Saves a single best-by-val-loss `.pt` containing every state dict.                    |
| `ShapeLogger`/`ModelSummary`| One-shot architecture introspection saved as Markdown in `docs/`.                    |
| `GradientClipper`          | Pre-clip norm logging + `clip_grad_norm_`.                                            |
| `OverfitManager`           | Single-batch repeating loader for sanity checks.                                      |
| `ResourceMonitor`          | Background thread polling CPU / GPU / RAM via `tools.ResourceMonitor`.                |

AMP is enabled when `training.use_amp and torch.cuda.is_available()`; in that case `scaler = torch.amp.GradScaler("cuda")`.

## 4. Optimizer parameter groups

`make_param_groups()` delegates to `model.config.get_param_groups(model)`, allowing each architecture (e.g. encoder vs heads) to expose different `lr` and `weight_decay`. Each group is logged with name, LR, WD, and parameter count. AdamW is used universally with `betas=config.optimizer.betas`, `eps=config.optimizer.eps`.

## 5. Warmup × scheduler interaction

[Warmup](../pipelines/training_pipeline/warmup.py) acts on **optimizer steps**, not epochs. The current factor is

$$
f_t = f_0 + (1 - f_0)\cdot \min\!\left(\frac{t}{T_w}, 1\right),
$$

with `f_0 = warmup_start_factor` and `T_w = warmup_steps`. Each `param_group['lr']` is set to `base_lr * f_t`. Once $t > T_w$ the factor is restored to `1.0` exactly once and `warmup_finished = True`.

[Scheduler.step()](../pipelines/training_pipeline/scheduler.py) is called once per epoch from `train()`, and short-circuits whenever `warmup.is_finished()` is `False`. `ReduceLROnPlateau` receives `metric=val_loss`. Per-group LRs are logged to TensorBoard under `lr/<group_name>`.

Available scheduler types (`config.scheduler.type`):

`cosine_annealing`, `step`, `multi_step`, `exponential`, `reduce_on_plateau`, `one_cycle`, `cosine_annealing_warm_restarts`, `linear`, `polynomial`. Each has its own `_create_*` factory reading kwargs from `config.scheduler` via `getattr(..., default)`.

## 6. The composite Loss

[Loss.__call__](../pipelines/training_pipeline/loss.py#L116) computes a curve reconstruction $\hat{C} = \text{reconstruct}(\hat\Theta)$ from predicted parameters using the differentiable Gaussian sum (`Metrics.reconstruct_gaussians`):

$$
\hat{C}(h, a, r) = \sum_{k=1}^{K} A_k\,\exp\!\left(-\frac{(h - \mu_k)^2}{2\sigma_k^2 + 10^{-8}}\right).
$$

It then computes up to 11 weighted components, each gated by an `cfg.use_*` flag and weighted by `cfg.weight_*`:

| Term name           | Formula (per element / per pixel)                                                                            |
|---------------------|---------------------------------------------------------------------------------------------------------------|
| `mse_curve`         | $\mathbb{E}[(\hat C - C)^2]$                                                                                  |
| `l1_curve`          | $\mathbb{E}[|\hat C - C|]$                                                                                    |
| `huber_curve`       | `F.huber_loss(pred, exp, delta)`                                                                              |
| `charbonnier_curve` | $\mathbb{E}\!\left[\sqrt{(\hat C - C)^2 + \epsilon^2}\right]$                                                 |
| `cosine_curve`      | $\mathbb{E}_{a,r}[1 - \cos(\hat C(\cdot,a,r), C(\cdot,a,r))]$                                                 |
| `spectral_coh`      | $\mathbb{E}[1 - \gamma]$ with $\gamma$ from §6.2                                                              |
| `ssim_curve`        | $\mathbb{E}[1 - \text{SSIM}(\hat C_{:,h,\cdot,\cdot},\, C_{:,h,\cdot,\cdot})]$ averaged over $h$               |
| `param_l1`          | $\sum_k w_k |\hat\theta_k - \theta_k^{\text{gt}}|$ with optional $\mu$-sorted matching                        |
| `param_huber`       | Same with Huber elementwise loss                                                                              |
| `smoothness_tv`     | $\mathbb{E}[|\partial_x \hat\Theta|] + \mathbb{E}[|\partial_y \hat\Theta|]$                                   |
| `masked_param`      | See §6.3                                                                                                      |

Components and their weighted versions are returned as detached tensors for logging; the differentiable `total_loss` is the sum.

### 6.1 SSIM term

[Loss._ssim_loss()](../pipelines/training_pipeline/loss.py#L74) reshapes `(B, N_h, H, W)` curves into `(B·N_h, 1, H, W)`, applies a Gaussian kernel `_gaussian_window(window_size, sigma)`, and computes the standard SSIM map:

$$
\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)},\qquad c_i=(k_i \cdot \text{data\_range})^2.
$$

The SSIM is therefore measured per elevation slice in the spatial $(H,W)$ plane.

### 6.2 Spectral coherence

[Metrics.spectral_coherence()](../pipelines/training_pipeline/metrics.py#L52) treats both curves as complex signals and computes a windowed coherence using `F.conv3d` with a length-`win` averaging kernel along the elevation axis:

$$
\gamma = \frac{|\langle \hat{C}\,\overline{C}\rangle|}{\sqrt{\langle|\hat{C}|^2\rangle\,\langle|C|^2\rangle}}\quad \in [0, 1].
$$

The mean over elevation is returned as a `(B, H, W)` map; the loss uses $1-\gamma$.

### 6.3 Masked parameter loss

[MaskedParamLoss](../pipelines/training_pipeline/param_mask.py) defines an **active mask** per Gaussian component $i \ge 1$ (in normalized parameter space):

$$
m_i(a,r) = \mathbb{1}\big[\hat A_i > a_\text{ths}\big]\ \land\ \mathbb{1}\big[\hat \sigma_i \ge w_\text{ths}\,(1 + p)\big].
$$

The first Gaussian always uses an all-ones mask (it is treated as the “primary” and always supervised). Per active component, an elementwise loss (`mse`, `l1`, or `huber` with `delta`) is averaged with the mask weights:

$$
\mathcal{L}_\text{masked} = \sum_{i=1}^{K}\sum_{p\in\{A_i,\mu_i,\sigma_i\}} \frac{\sum_{a,r} m_i(a,r)\, \ell(\hat p, p^\text{gt})}{\sum m_i + 10^{-8}}.
$$

### 6.4 Parameter matching strategies

`cfg.param_match`:

- `"index"` — straight per-channel matching.
- `"sorted_mu"` — within each batch element, sort predicted and GT Gaussians by their $\mu$ value and re-gather along the Gaussian axis before computing the term. This breaks the permutation symmetry of unordered Gaussian mixtures.

Per-role weights are pulled from `cfg.param_weights` (padded with `1.0` if shorter than `params_per_gaussian`).

### 6.5 Logging cadence

If `cfg.log_components_every > 0`, every `step % every == 0` the per-component scalars and `loss_total` are pushed to TensorBoard under `train/components/*` and `train/loss_total`.

## 7. Forward / backward step

[Trainer.forward()](../pipelines/training_pipeline/trainer.py#L138) wraps the model + loss inside `torch.amp.autocast("cuda", enabled=use_amp)`. The model is expected to apply its own amplitude constraint (e.g. softplus on $A$ channels), but `_apply_amplitude_constraint()` is provided as a fallback.

[Trainer.backward()](../pipelines/training_pipeline/trainer.py#L147):

1. `scaler.scale(loss).backward()` (AMP) or plain `loss.backward()`.
2. If this iteration ends an accumulation window:
   - `scaler.unscale_(optimizer)` (AMP).
   - `grad_norm = grad_clipper.step(model.parameters(), global_step)`; pre-clip norm is logged each step, histogram of last 100 norms every 100 steps.
   - Optional debug logging: gradient histograms, optimizer state.
   - `scaler.step + update` or `optimizer.step()`; `optimizer.zero_grad()`.
   - `warmup.step()`; `global_step += 1`; `ema.update(model, step=global_step)`.

`accumulation_steps = config.training.gradient_accumulation_steps` and the loss is divided by it inside the loop so that the effective gradient matches a `batch_size × accumulation_steps` micro-batch.

## 8. Epoch loop

[Trainer.train_epoch()](../pipelines/training_pipeline/trainer.py#L177):

- Sets `model.train()`.
- Optionally registers activation hooks every 10 epochs (debug mode).
- Iterates the loader inside a Rich progress bar.
- Per batch: `_unpack_batch`, move tensors to `device` non-blocking, `forward`, `backward`.
- `loss/train_step` per step, `loss/train_epoch` per epoch, `tracker.log_memory(epoch)`.
- If `clear_cache_every_n_steps > 0`, calls `gc.collect() + torch.cuda.empty_cache()` periodically.

[Trainer.evaluate()](../pipelines/training_pipeline/trainer.py#L237) (called for `validation`, `train` eval, and the three `final_*` evaluations):

1. `model.eval()`; `ema.apply_to(model)` (swap to shadow weights); `try / finally` `ema.restore`.
2. Build a [MetricsAggregator](../pipelines/training_pipeline/metrics_aggregator.py) with `deep` flag and pixel-array configuration.
3. Iterate loader in `torch.no_grad()`, accumulate `total_loss` and per-batch updates.
4. **GT denormalization for deep eval** — if `norm_stats.output_stats` exists, GT params are denormalized before being passed to the aggregator so that the comparison happens in physical units.
5. `agg.finalize(epoch, stage, last_C)` returns the final dict and pushes it to TensorBoard.
6. Optional cache cleanup based on `memory.clear_cache_after_eval`.

## 9. Metrics aggregator

[MetricsAggregator.update()](../pipelines/training_pipeline/metrics_aggregator.py#L70) accumulates online:

- Total squared / absolute curve error and elements (for global MSE / MAE / RMSE).
- Per-pixel `mse`, `mae`, `r2`, optionally `cos_sim`, `spec_coh` — each subsampled (random `pixel_subsample` indices) to bound memory.
- A Welford accumulator for `expected_stats` from which the global $R^2$ is derived using `overall_r2 = 1 - sum_squared / m2_expected`.
- Per-channel running mean / std of predicted parameters (Welford) plus min/max under deep mode.
- Per-channel `(sum_squared, sum_absolute, sum_gt, sum_gt_squared, count)` for **GT-vs-pred** parameter MSE/MAE/R², when `gt_params` are passed (deep mode only).

`finalize()` collapses everything into a flat dict that is forwarded to `metrics.track_results(...)` for TensorBoard logging.

## 10. Checkpointing

[Checkpoint.step()](../pipelines/training_pipeline/checkpoint.py#L18) is called every epoch; it persists the model only if `val_loss < best_val_loss`. The single saved file under `<run_dir>/best_model.pt` includes:

- `model_state_dict`, `optimizer_state_dict`, `lr_scheduler_state_dict`.
- `ema_state_dict`, `early_stopping_state` (best-loss / counter / best-model-state), `warmup_state` (current step / finished).
- `scaler_state_dict` (AMP).
- `epoch`, `global_step`, `best_val_loss`, `best_epoch`, `best_metrics`, `train_losses`, `val_losses`.
- `config` (`to_dict()` if available, else `str(config)`), `x_axis` (CPU tensor).

`Checkpoint.load()` restores every block back into the trainer and returns the saved epoch number.

## 11. Early stopping

[EarlyStopping](../pipelines/training_pipeline/early_stopping.py) tracks the lowest `val_loss` modulo `min_delta`. It increments a counter on plateaus and triggers a stop when `counter >= patience`. If `restore_best=True`, the best CPU snapshot is loaded back into the model on stop.

## 12. EMA

[EMA.update()](../pipelines/training_pipeline/ema.py#L26): for each `requires_grad` parameter,

$$
\bar\theta_i \leftarrow d\,\bar\theta_i + (1-d)\,\theta_i.
$$

`apply_to(model)` swaps in the shadow weights for evaluation (storing the current values in `backup`) and `restore(model)` puts the originals back. State is fully persisted in checkpoints.

## 13. Overfit mode

[OverfitManager.setup_loaders()](../pipelines/training_pipeline/overfit_manager.py): if `training.overfit_enabled`, replaces the loaders with `[single_batch] * min(50, len(train_loader))` so the model attempts to memorise one mini-batch. `check_stop(train_loss)` triggers an early stop when `train_loss < 1e-6`.

## 14. End-of-training: final evaluations

After the epoch loop ends (early-stopping or budget exhaustion):

1. `ShapeLogger.save_markdown(sort_by_layer=True)`.
2. `Checkpoint.load()` → restore best epoch.
3. Evaluate **train**, **val**, **test** with `deep=True` (`final_train`, `final_validation`, `final_test`), each writing the full deep-metric dict to TensorBoard.
4. `resource_monitor.stop()` in `finally`.

The returned tuple `(train_final, val_final, test_final)` propagates back to `TrainingPipeline.run()`.

## 15. Configuration reference

[TrainerConfig](../configuration/training_config.py) groups:

- `training` — `device`, `epochs`, `validation_frequency`, `verbose`, `use_amp`, `gradient_accumulation_steps`, `max_grad_norm`, `overfit_enabled`, `deep_validation`, `log_debug`.
- `optimizer` — `betas`, `eps`.
- `warmup` — `warmup_enabled`, `warmup_steps`, `warmup_start_factor`.
- `scheduler` — `type` and per-type kwargs.
- `ema` — `use_ema`, `ema_decay`.
- `early_stopping` — `patience`, `min_delta`, `restore_best`.
- `loss : LossConfig` — every `use_*` / `weight_*` / `*_delta` / `*_eps` / `*_window_size` / `param_match` / `param_weights` / `masked_param_*` / `log_components_every`.
- `gaussian : GaussianConfig` — `params_per_gaussian = 3`, `n_default_gaussians`, `x_min`, `x_max`, `make_param_names(K)`.
- `memory` — `clear_cache_every_n_steps`, `clear_cache_after_eval`, `clear_cache_after_epoch`, `eval_keep_pixel_arrays`, `eval_pixel_subsample`.
- `resources` — `ResourceMonitor` controls.
- `io` — populated by `TrainingRunMetadata`.

## 16. Outputs

```python
return train_final_results, validation_final_results, test_final_results
```

with each dict shaped like the return of `MetricsAggregator.finalize`. Side-effects:

- `<run>/best_model.pt` — best checkpoint.
- `<run>/tensorboard/` — full scalar / histogram logs.
- `<run>/docs/` — `trainer_config.json`, `model_summary.md`, `shape_logger.md`.
- `<run>/meta/` — `run_summary.json`, dataset-creation metadata.
- `<run>/logs/` — Rich console log files.
