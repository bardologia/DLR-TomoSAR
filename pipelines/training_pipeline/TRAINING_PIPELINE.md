# Training Pipeline — Technical Reference

**Package:** `pipelines.training_pipeline`  
**Location:** `pipelines/training_pipeline/`  
**Role:** End-to-end supervised training of convolutional/transformer neural networks for SAR tomographic Gaussian parameter regression. Orchestrates dataset construction, model instantiation, optimisation, learning-rate scheduling, regularisation, and artefact persistence.

---

## Table of Contents

1. [Overview](#1-overview)  
2. [Architecture](#2-architecture)  
3. [Configuration Layer](#3-configuration-layer)  
4. [Component Responsibilities](#4-component-responsibilities)  
   - 4.1 [TrainingRunMetadata](#41-trainingrunmetadata)  
   - 4.2 [TrainingPipeline](#42-trainingpipeline)  
   - 4.3 [Trainer](#43-trainer)  
   - 4.4 [Loss](#44-loss)  
   - 4.5 [Warmup](#45-warmup)  
   - 4.6 [Scheduler](#46-scheduler)  
   - 4.7 [EMA](#47-ema)  
   - 4.8 [GradientClipper](#48-gradientclipper)  
   - 4.9 [EarlyStopping](#49-earlystopping)  
   - 4.10 [Checkpoint](#410-checkpoint)  
   - 4.11 [OverfitManager](#411-overfitmanager)  
5. [Pipeline Execution Stages](#5-pipeline-execution-stages)  
6. [Mathematical Formulation](#6-mathematical-formulation)  
   - 6.1 [Gaussian Curve Reconstruction](#61-gaussian-curve-reconstruction)  
   - 6.2 [Loss Terms](#62-loss-terms)  
   - 6.3 [Total Weighted Loss](#63-total-weighted-loss)  
   - 6.4 [Warmup Schedules](#64-warmup-schedules)  
   - 6.5 [Learning-Rate Schedules](#65-learning-rate-schedules)  
   - 6.6 [Exponential Moving Average](#66-exponential-moving-average)  
   - 6.7 [Gradient Clipping](#67-gradient-clipping)  
   - 6.8 [Early Stopping Criterion](#68-early-stopping-criterion)  
7. [Curriculum Learning](#7-curriculum-learning)  
8. [Artifact Naming and Directory Layout](#8-artifact-naming-and-directory-layout)  
9. [Inputs and Outputs Summary](#9-inputs-and-outputs-summary)  
10. [Canonical Usage](#10-canonical-usage)  
11. [Public API Reference](#11-public-api-reference)

---

## 1. Overview

The training pipeline accepts pre-processed SAR artifacts and Gaussian parameter maps and produces a trained neural network model together with a complete set of reproducibility artifacts. It operates on top of the **dataset pipeline** (which generates PyTorch `DataLoader` objects) and the **model registry** (which instantiates architectures from a configuration). All state — optimiser moments, EMA shadows, LR scheduler state, early-stopping counters, loss histories — is serialised to a single checkpoint file at every new validation-loss minimum.

The fundamental regression task is:

$$
f_\theta : \mathbb{R}^{C_{\text{in}} \times P_h \times P_w} \rightarrow \mathbb{R}^{C_{\text{out}} \times P_h \times P_w}
$$

where $C_{\text{in}}$ is the number of real-valued SAR input channels, $C_{\text{out}} = K \cdot P$ is the number of Gaussian parameter output channels ($K$ components, $P$ parameters each), and $(P_h, P_w)$ is the patch spatial size.

---

## 2. Architecture

```
TrainerConfig + DatasetConfiguration
           │
           ▼
┌───────────────────────┐
│   TrainingPipeline    │  (pipeline.py)
│  ┌─────────────────┐  │
│  │TrainingRunMeta  │  │ ← creates run directory tree, SummaryWriter
│  └────────┬────────┘  │
│           │            │
│  ┌────────▼────────┐  │
│  │ DatasetPipeline │  │ ← train/val/test DataLoaders + norm_stats
│  └────────┬────────┘  │
│           │            │
│  ┌────────▼────────┐  │
│  │  _build_model   │  │ ← models.get_model(name, in_ch, out_ch)
│  └────────┬────────┘  │
│           │            │
│  ┌────────▼────────┐  │
│  │    Trainer      │  │ (trainer.py)
│  └────────┬────────┘  │
└───────────┼────────────┘
            │
            │  owns
            ├─── Loss              (loss.py)
            ├─── Warmup            (warmup.py)
            ├─── Scheduler         (scheduler.py)
            ├─── EMA               (ema.py)
            ├─── GradientClipper   (gradient_clipper.py)
            ├─── EarlyStopping     (early_stopping.py)
            ├─── Checkpoint        (checkpoint.py)
            ├─── OverfitManager    (overfit.py)
            ├─── ResourceMonitor   (tools)
            └─── Tracker           (tools / TensorBoard)

  Trainer.train()
       │
       ├── epoch loop
       │     ├── _apply_curriculum_swap
       │     ├── train_epoch  ──► _forward ──► _backward
       │     │                      (AMP, GradScaler, gradient clipping, EMA)
       │     ├── evaluate (every N epochs)
       │     │     (EMA apply → eval mode → EMA restore)
       │     ├── Checkpoint.step
       │     ├── Scheduler.step
       │     └── EarlyStopping.__call__
       │
       └── EarlyStopping.restore (best params)
```

---

## 3. Configuration Layer

All behaviour is governed by `TrainerConfig` (defined in `configuration/training_config.py`). The root object is composed of nested dataclasses; no global mutable state exists.

### `TrainerConfig` — root

| Sub-config | Attribute | Description |
|-----------|-----------|-------------|
| `TrainingConfig` | `training` | Epochs, validation frequency, AMP, gradient accumulation, logging. |
| `OptimizerConfig` | `optimizer` | AdamW betas, epsilon, weight decay. |
| `WarmupConfig` | `warmup` | Steps, start factor, mode. |
| `SchedulerConfig` | `scheduler` | Type and hyperparameters for each schedule. |
| `EMAConfig` | `ema` | EMA enable flag, decay, update frequency. |
| `GradientClipperConfig` | `gradient_clipper` | Mode (fixed / adaptive), threshold, window, percentile. |
| `EarlyStoppingConfig` | `early_stopping` | Patience, min delta, restore-best flag. |
| `OverfitConfig` | `overfit` | Enable flag, max steps, stop threshold, batch size. |
| `GaussianConfig` | `gaussian` | $K$ components, x-axis bounds, amplitude ceiling, params-per-Gaussian. |
| `CurriculumConfig` | `curriculum` | Swap epoch, warmup / complete `LossConfig`, reset flags. |
| `IOConfig` | `io` | Log directory, TensorBoard dir, docs dir, `SummaryWriter` handle. |
| `MemoryConfig` | `memory` | Cache-clear frequencies and thresholds. |
| `ResourceConfig` | `resources` | CPU/GPU monitoring intervals. |
| `PermutationMetricsConfig` | `permutation_metrics` | Whether to compute permutation-invariant regression metrics. |

### `LossConfig`

Controls which loss terms are active and their individual weights.

| Field | Type | Description |
|-------|------|-------------|
| `use_mse_curve` | `bool` | Per-pixel MSE on reconstructed Gaussian curves. |
| `weight_mse_curve` | `float` | Unnormalised weight $\alpha_{\text{mse}}$. |
| `use_l1_curve` | `bool` | Mean absolute error on curves. |
| `use_huber_curve` | `bool` | Huber loss on curves (delta configurable). |
| `use_charbonnier_curve` | `bool` | Charbonnier (differentiable L1) on curves. |
| `use_cosine_curve` | `bool` | Cosine distance over the elevation axis. |
| `use_spectral_coherence` | `bool` | 1 − windowed spectral coherence. |
| `use_ssim_curve` | `bool` | 1 − SSIM over configurable spatial axis. |
| `use_param_l1` | `bool` | Weighted per-parameter L1 in normalised space. |
| `use_param_huber` | `bool` | Weighted per-parameter Huber loss. |
| `use_smoothness_tv` | `bool` | Total variation regularisation on predicted parameters. |
| `param_match` | `str` | Gaussian permutation matching strategy (`"none"`, `"hungarian"`, `"greedy"`). |
| `param_weights` | `list[float]` | Per-parameter-role weights $[w_a, w_\mu, w_\sigma]$. |
| `amp_zero_thr` | `float` | Amplitude threshold below which $\mu, \sigma$ are masked from param losses. |

---

## 4. Component Responsibilities

### 4.1 `TrainingRunMetadata`

**File:** `metadata.py`

Creates the run directory tree and a TensorBoard `SummaryWriter`. All other components receive directory paths from this object.

**Directory tree created at construction:**

```
{base_logdir}/{run_name}/
    tensorboard/    ← SummaryWriter events
    docs/           ← model summary, trainer config JSON
    logs/           ← Logger output files
    meta/           ← run_summary.json (from TrainingPipeline)
    checkpoints/    ← (reserved; best_model.pt is written to run root)
```

The `run_name` defaults to `run_{model_name}_{YYYYMMDD_HHMMSS}` if not supplied.

**Key methods:**

| Method | Output |
|--------|--------|
| `save_trainer_config()` | `docs/trainer_config.json` — full `TrainerConfig` as JSON. |
| `save_run_summary(...)` | `meta/run_summary.json` — model name, channel counts, device count. |
| `close()` | Flushes and closes the TensorBoard `SummaryWriter`. |

---

### 4.2 `TrainingPipeline`

**File:** `pipeline.py`

The top-level orchestrator. Constructed once and driven by a single call to `run()`.

**Constructor responsibilities:**

1. Sets global random seeds (`torch.manual_seed`, `torch.cuda.manual_seed_all`).
2. Enables `torch.backends.cudnn.deterministic`.
3. Instantiates `TrainingRunMetadata` (creates directory tree).
4. Instantiates `DatasetPipeline` (does not run it yet).

**`run()` sequence:**

1. Reads the `tomogram_full.npy` artifact to determine `x_axis_length`.
2. Constructs the elevation axis: $\mathbf{x} = \text{linspace}(x_{\min}, x_{\max}, L)$.
3. Calls `DatasetPipeline.run()` → three `DataLoader` objects + normalisation stats.
4. Derives `in_channels` and `out_channels` from the training dataset and Gaussian config.
5. Calls `_build_model(in_channels, out_channels)` via the model registry.
6. Saves `trainer_config.json` and `run_summary.json`.
7. Constructs `Trainer` and calls `trainer.train(...)`.
8. Calls `run_metadata.close()` in a `finally` block.

---

### 4.3 `Trainer`

**File:** `trainer.py`

Owns and coordinates all training sub-components. The training loop is `Trainer.train()`.

**Construction initialises (in order):**

1. Device selection (`cuda` if available, else `cpu`).
2. Model transfer to device.
3. `ModelSummary` computation and Markdown export.
4. `AdamW` optimiser with per-parameter-group learning rates from `model_cfg.get_param_groups`.
5. `Warmup`, `Scheduler`, `EMA`, `EarlyStopping`, `Loss`, `Checkpoint`, `GradientClipper`, `OverfitManager`, `ResourceMonitor`, `PermutationMetrics`.
6. EMA shadow initialisation from model parameters.

**`train()` loop:**

```
for epoch in 0 … epochs-1:
    _apply_curriculum_swap(epoch)
    train_loss = train_epoch(train_loader, epoch)
    if validation_due:
        val_results = evaluate(val_loader, epoch, stage="validation")
        Checkpoint.step(val_loss, epoch)
        new_lrs = Scheduler.step(epoch, metric=val_loss)
        stop = EarlyStopping(val_loss, model, epoch)
    else:
        new_lrs = Scheduler.step(epoch, metric=None)
    _update_optimizer(new_lrs)
    if stop or OverfitManager.check_stop(train_loss): break

EarlyStopping.restore(model)
```

**`train_epoch()` inner loop:**

```
model.train()
for batch_idx, (images, gt_params) in enumerate(train_loader):
    loss, loss_dict = _forward(images, gt_params)   # AMP autocast
    _backward(loss, batch_idx, n_batches)            # GradScaler + clipping + EMA
    warmup.step()
    global_step += 1
```

**`evaluate()` loop:**

```
ema.apply_to(model)
model.eval()
for images, gt_params in loader:
    pred = model(images)           # AMP autocast, no_grad
    loss_dict = criterion(pred, gt_params)
    accumulate losses and permutation metrics
ema.restore(model)
```

---

### 4.4 `Loss`

**File:** `loss.py`

Implements a composite, configurable loss that operates on both reconstructed Gaussian **curves** and directly on **parameters** in normalised space.

**Execution flow in `__call__`:**

1. **Physical clamping:** Denormalise predicted parameters and clamp to valid physical ranges (amplitude ≥ 0, sigma > 0) using `clamp_gaussian_params`.
2. **Re-normalise:** Convert clamped physical params back to normalised space for parameter-space loss terms.
3. **Curve reconstruction:** Build predicted and ground-truth Gaussian PSD curves via `reconstruct_gaussians`.
4. **Term evaluation:** Compute each active loss term.
5. **Weighted aggregation:** Sum weighted terms and normalise by the total weight sum.

**Return value:** `{"total_loss": Tensor, "components": dict, "weighted": dict}`

The `Loss` object also manages a `ParamMatcher` for permutation-invariant Gaussian assignment, called during parameter-space loss evaluation.

---

### 4.5 `Warmup`

**File:** `warmup.py`

Implements a per-step multiplicative warm-up factor applied to all learning rates at the start of training (and optionally after curriculum swap).

**State:** `current_step`, `warmup_finished`.

**`step()`** increments `current_step`, computes and returns `factor()`, logs to TensorBoard, and sets `warmup_finished` when `current_step >= warmup_steps`.

**`factor()`** returns a scalar $f \in [s_0, 1]$ where $s_0 = $ `warmup_start_factor` — see §6.4.

---

### 4.6 `Scheduler`

**File:** `scheduler.py`

Implements epoch-level learning-rate scheduling. Computes a scalar multiplier applied to each parameter group's base learning rate, further scaled by the warmup factor if warmup is not yet finished.

**`step(epoch, metric)`** returns a list of learning rates (one per parameter group). The caller (`Trainer._update_optimizer`) applies them to the optimiser.

**Supported types:** `cosine_annealing`, `cosine_annealing_warm_restarts`, `step`, `multi_step`, `exponential`, `linear`, `polynomial`, `reduce_on_plateau`, `constant` — see §6.5.

---

### 4.7 `EMA`

**File:** `ema.py`

Maintains an exponential moving average of all trainable parameters in a `shadow` dictionary.

**Lifecycle:**

| Method | When called |
|--------|-------------|
| `init(model)` | Once at `Trainer` construction. |
| `update(model, step)` | Every `_ema_every` gradient steps during `_backward`. |
| `apply_to(model)` | At the start of each `evaluate()` call — copies shadow to model params, saves backup. |
| `restore(model)` | At the end of `evaluate()` — restores live params from backup. |

---

### 4.8 `GradientClipper`

**File:** `gradient_clipper.py`

Computes the global $\ell_2$ gradient norm and optionally rescales all gradients before the optimiser step. Supports fixed thresholds and two adaptive strategies based on a rolling history window.

**Modes:** `disabled`, `fixed`, `adaptive_percentile`, `adaptive_mean_std` — see §6.7.

---

### 4.9 `EarlyStopping`

**File:** `early_stopping.py`

Monitors the validation loss and halts training if no improvement exceeds `min_delta` over `patience` consecutive validation epochs.

**`restore_best`:** If enabled, a CPU copy of the model's `state_dict` is kept whenever a new best validation loss is achieved. After stopping, `Trainer.train()` calls `EarlyStopping.restore(model)` to load these weights.

---

### 4.10 `Checkpoint`

**File:** `checkpoint.py`

Saves and loads a **complete training state** whenever a new validation-loss minimum is reached.

**Checkpoint contents:**

| Key | Description |
|-----|-------------|
| `epoch` | Epoch index at save time. |
| `global_step` | Total gradient steps taken. |
| `best_val_loss` | Best observed validation loss. |
| `params` | `model.state_dict()` |
| `opt_state` | `optimizer.state_dict()` |
| `ema_shadow` | EMA shadow parameter dict. |
| `scheduler_state` | `Scheduler.state_dict()` |
| `warmup_state` | `Warmup.state_dict()` |
| `early_stopping_state` | `{best_loss, counter, best_params}` |
| `train_losses` / `val_losses` | Full epoch-level loss histories. |
| `x_axis` | Elevation axis used during the run. |
| `config` | String or dict representation of `TrainerConfig`. |

---

### 4.11 `OverfitManager`

**File:** `overfit.py`

Enables controlled single-batch overfitting for debugging. When `config.overfit.enabled` is `True`:

1. Extracts a single mini-batch of size `overfit.batch_size` from the training loader.
2. Replaces all three loaders with repeated lists of this single batch.
3. `check_stop()` halts when `max_steps` total optimiser steps are taken or when `train_loss < stop_threshold`.

---

## 5. Pipeline Execution Stages

### Stage 0 — Initialisation

```
TrainingPipeline.__init__
    seed(global RNG)
    TrainingRunMetadata(...)     ← mkdir, SummaryWriter
    DatasetPipeline(...)         ← Layout parsed; no data loaded yet
```

### Stage 1 — Data Preparation

```
TrainingPipeline.run()
    ├── read tomogram_full.npy       ← derive x_axis_length
    ├── construct x_axis             ← linspace(x_min, x_max, L)
    └── DatasetPipeline.run()
            ├── Cropper.load_split("train")
            ├── StatsComputer.compute_input_stats
            ├── StatsComputer.compute_output_stats
            ├── Stats.save  → meta/normalization_stats.json
            ├── Cropper.load_split("val"), .load_split("test")
            └── Loader.build → (train_loader, val_loader, test_loader)
```

### Stage 2 — Model Instantiation

```
    _build_model(in_channels, out_channels)
        models.get_model(name, config, ...)
        → model, model_cfg
        ModelSummary.save_markdown → docs/model_summary.md
    save_trainer_config  → docs/trainer_config.json
    save_run_summary     → meta/run_summary.json
```

### Stage 3 — Trainer Construction

```
    Trainer.__init__
        model.to(device)
        AdamW(model_cfg.get_param_groups(model))
        Warmup, Scheduler, EMA, EarlyStopping
        Loss(x_axis, norm_stats, gaussian_cfg, loss_cfg)
        Checkpoint, GradientClipper, OverfitManager
        EMA.init(model)
        ShapeLogger (deferred to first training step)
```

### Stage 4 — Training Loop

```
    Trainer.train(train_loader, val_loader, test_loader)
        OverfitManager.setup_loaders
        ShapeLogger.run (single forward pass, detached)
        ResourceMonitor.start
        for epoch in 0…E-1:
            _apply_curriculum_swap(epoch)
            train_epoch
                for batch in train_loader:
                    _forward → Loss.__call__
                    _backward → GradScaler + GradClipper + EMA.update
                    Warmup.step
            if validation_due:
                EMA.apply_to → evaluate → EMA.restore
                Checkpoint.step
                Scheduler.step(metric=val_loss)
                EarlyStopping(val_loss)
            else:
                Scheduler.step(metric=None)
            _update_optimizer(new_lrs)
            if stop: break
        EarlyStopping.restore(model)
        ResourceMonitor.stop
```

### Stage 5 — Teardown

```
    run_metadata.close()    ← SummaryWriter flush + close
    logger.close()
    return (train_losses, val_losses, best_val_loss)
```

---

## 6. Mathematical Formulation

### 6.1 Gaussian Curve Reconstruction

Given the predicted parameter tensor $\hat{\mathbf{P}} \in \mathbb{R}^{B \times C_{\text{out}} \times H \times W}$ with $C_{\text{out}} = K \cdot 3$ and the elevation axis $\mathbf{x} \in \mathbb{R}^L$, the reconstructed power spectral density is:

$$
\hat{S}(x; b, h, w) = \sum_{k=1}^{K} \hat{a}_k \exp\!\left(-\frac{(x - \hat{\mu}_k)^2}{2 \hat{\sigma}_k^2}\right)
$$

yielding $\hat{\mathbf{S}} \in \mathbb{R}^{B \times L \times H \times W}$.

This reconstruction is applied to **both** predicted and ground-truth parameter maps (the latter under `torch.no_grad()`), and all curve-based loss terms compare $\hat{\mathbf{S}}$ against $\mathbf{S}^*$.

Physical parameters are clamped before reconstruction:

- $\hat{a}_k \geq 0$ (with leaky slope 0.01 below zero)
- $\hat{\sigma}_k > 0$
- $\hat{\mu}_k \in [x_{\min}, x_{\max}]$

### 6.2 Loss Terms

Let $\hat{s} = \hat{\mathbf{S}}$ and $s^* = \mathbf{S}^*$ denote the vectorised curves over all batch/spatial/elevation indices.

**Mean Squared Error:**

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_i (\hat{s}_i - s^*_i)^2
$$

**Mean Absolute Error (L1):**

$$
\mathcal{L}_{\text{L1}} = \frac{1}{N} \sum_i |\hat{s}_i - s^*_i|
$$

**Huber Loss** (with threshold $\delta$):

$$
\mathcal{L}_{\text{Huber}} = \frac{1}{N} \sum_i \ell_\delta(\hat{s}_i - s^*_i), \quad
\ell_\delta(r) = \begin{cases} \tfrac{1}{2} r^2 & |r| \leq \delta \\ \delta(|r| - \tfrac{\delta}{2}) & |r| > \delta \end{cases}
$$

**Charbonnier Loss** (with $\varepsilon$):

$$
\mathcal{L}_{\text{Charbonnier}} = \frac{1}{N} \sum_i \sqrt{(\hat{s}_i - s^*_i)^2 + \varepsilon^2}
$$

**Cosine Distance** (along elevation axis $n=1$):

$$
\mathcal{L}_{\text{cosine}} = \frac{1}{|\mathcal{V}|} \sum_{b,h,w \in \mathcal{V}} \left(1 - \frac{\hat{\mathbf{s}} \cdot \mathbf{s}^*}{\|\hat{\mathbf{s}}\| \|\mathbf{s}^*\|}\right)
$$

where $\mathcal{V}$ is the set of pixels with $\|\mathbf{s}^*\| > 10^{-3}$.

**Spectral Coherence Loss** (with window $W$):

$$
\mathcal{L}_{\text{spec}} = 1 - \frac{1}{M} \sum_i \left|\frac{\sum_{j \in W_i} \hat{s}_j s^*_j}{\sqrt{\sum_{j \in W_i} \hat{s}_j^2 \cdot \sum_{j \in W_i} (s^*_j)^2}}\right|
$$

**Structural Similarity (SSIM)** (applied per spatial slice with Gaussian window $\mathbf{k}$):

$$
\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

$$
C_1 = (k_1 \cdot D)^2, \quad C_2 = (k_2 \cdot D)^2
$$

where $D$ = `ssim_data_range` (dynamic range after local min-max normalisation), and $\mu, \sigma$ are computed via 2-D Gaussian convolution.

$$
\mathcal{L}_{\text{SSIM}} = 1 - \text{SSIM}(\hat{\mathbf{S}}, \mathbf{S}^*)
$$

**Per-Parameter L1** (in normalised space, after Hungarian/greedy matching):

$$
\mathcal{L}_{\text{param-L1}} = \frac{1}{N} \sum_{b,k,p,h,w} w_p \cdot m_{b,k,p,h,w} \cdot |\hat{P}_{b,k,p,h,w} - P^*_{b,k,p,h,w}|
$$

where $w_p \in \{w_a, w_\mu, w_\sigma\}$ and $m_{b,k,p,h,w}$ is a binary mask that suppresses $\mu$ and $\sigma$ losses for components with ground-truth amplitude $a^* \leq $ `amp_zero_thr`.

**Total Variation Regularisation:**

$$
\mathcal{L}_{\text{TV}} = \frac{1}{N} \left(\sum_{b,c,i,j} |\hat{P}_{b,c,i+1,j} - \hat{P}_{b,c,i,j}| + |\hat{P}_{b,c,i,j+1} - \hat{P}_{b,c,i,j}|\right)
$$

### 6.3 Total Weighted Loss

Each active term $\ell \in \mathcal{A}$ has a configured weight $\alpha_\ell$ and an optional normalisation divisor $\nu_\ell$. The effective weight is $\tilde{\alpha}_\ell = \alpha_\ell / \nu_\ell$.

$$
\mathcal{L}_{\text{total}} = \frac{\sum_{\ell \in \mathcal{A}} \tilde{\alpha}_\ell \cdot \mathcal{L}_\ell}{\sum_{\ell \in \mathcal{A}} \tilde{\alpha}_\ell}
$$

The normalisation by the total weight sum ensures that the loss magnitude does not scale with the number of active terms.

**Gradient accumulation** divides the per-micro-batch loss by `accumulation_steps`:

$$
\hat{\mathcal{L}}_{\text{micro}} = \frac{\mathcal{L}_{\text{total}}}{\texttt{accumulation\_steps}}
$$

Gradients from successive micro-batches accumulate before a single optimiser update.

### 6.4 Warmup Schedules

Let $t$ denote the current step, $T_w$ the total warmup steps, $s_0$ the start factor, and $\phi = t / T_w \in [0, 1]$ the fractional progress.

**Linear:**

$$
f(t) = s_0 + (1 - s_0)\,\phi
$$

**Cosine:**

$$
f(t) = s_0 + (1 - s_0) \cdot \frac{1 - \cos(\pi\phi)}{2}
$$

**Exponential:**

$$
f(t) = s_0^{1 - \phi}
$$

**Polynomial** (power $p$):

$$
f(t) = s_0 + (1 - s_0)\,\phi^p
$$

All modes satisfy $f(0) = s_0$ and $f(T_w) = 1$. The effective learning rate during warmup is $\eta_t^{\text{eff}} = \eta_{\text{base}} \cdot r(t) \cdot f(t)$ where $r(t)$ is the epoch-level scheduler multiplier.

### 6.5 Learning-Rate Schedules

Let $\eta_0$ denote the base learning rate and $r(e)$ the multiplier at epoch $e$.

**Cosine Annealing:**

$$
r(e) = \eta_{\min}/\eta_0 + \frac{1}{2}\left(1 - \eta_{\min}/\eta_0\right)\left(1 + \cos\!\left(\frac{\pi e}{T_{\max}}\right)\right)
$$

**Cosine Annealing with Warm Restarts** ($T_0$, multiplier $T_{\text{mult}}$):

The period at restart $n$ is $T_i = T_0 \cdot T_{\text{mult}}^n$; within a period, the cosine schedule restarts from $\eta_0$.

**Step Decay:**

$$
r(e) = \gamma^{\lfloor e / s \rfloor}
$$

**Exponential:**

$$
r(e) = \gamma^e
$$

**Linear:**

$$
r(e) = s_{\text{start}} + (s_{\text{end}} - s_{\text{start}}) \cdot \min\!\left(1, \frac{e}{T}\right)
$$

**Polynomial:**

$$
r(e) = \left(1 - \min\!\left(1, \frac{e}{T}\right)\right)^p
$$

**Reduce on Plateau:** If the validation metric does not improve by more than `threshold` over `patience` epochs, the scheduler multiplies the current learning rate by `factor`:

$$
\eta_{e+1} = \begin{cases} \eta_e \cdot \gamma_{\text{plateau}} & \text{if stagnant for patience epochs} \\ \eta_e & \text{otherwise} \end{cases}
$$

### 6.6 Exponential Moving Average

The shadow parameter $\bar{\theta}$ is maintained for all trainable parameters $\theta$:

$$
\bar{\theta}_{t+1} = \beta \cdot \bar{\theta}_t + (1 - \beta)\,\theta_t
$$

where $\beta$ = `ema_decay` $\in [0, 1)$. The EMA shadow is applied during evaluation by copying $\bar{\theta}$ into the model, then restoring the live $\theta$ afterwards. This ensures that checkpointed validation losses reflect EMA performance.

### 6.7 Gradient Clipping

Let $\|\mathbf{g}\|_2$ denote the global gradient norm across all parameters:

$$
\|\mathbf{g}\|_2 = \sqrt{\sum_{p} \sum_{i} \left(\frac{\partial \mathcal{L}}{\partial p_i}\right)^2}
$$

**Fixed clipping** (threshold $\tau$):

$$
\mathbf{g}' = \mathbf{g} \cdot \min\!\left(1,\, \frac{\tau}{\|\mathbf{g}\|_2 + \varepsilon}\right)
$$

**Adaptive percentile** (rolling window of size $W$, percentile $q$):

$$
\tau_t = P_q\!\left(\{\|\mathbf{g}\|_{2,t-W}, \ldots, \|\mathbf{g}\|_{2,t-1}\}\right)
$$

**Adaptive mean + $k\sigma$:**

$$
\tau_t = \overline{\|\mathbf{g}\|}_W + k \cdot \hat{\sigma}_W
$$

In all adaptive modes, clipping is skipped until the history buffer contains at least $W$ samples.

### 6.8 Early Stopping Criterion

Let $v_e$ denote the validation loss at epoch $e$, $v^* = \min_{e' < e} v_{e'}$ the best observed loss, and $c$ the stagnation counter.

At each validation epoch:

$$
c_{e+1} = \begin{cases} 0 & v_e < v^* - \delta_{\min} \\ c_e + 1 & \text{otherwise} \end{cases}
$$

Training halts when $c \geq P$ (patience). If `restore_best` is enabled, the best model state — stored on CPU when $v^*$ is updated — is restored after stopping.

---

## 7. Curriculum Learning

The curriculum system supports a two-phase training strategy:

| Phase | Loss Config | Active from |
|-------|------------|-------------|
| **Warmup** | `curriculum.warmup` | Epoch 0 |
| **Complete** | `curriculum.complete` | Epoch `swap_epoch` |

At epoch `swap_epoch`, `Trainer._apply_curriculum_swap` performs the following atomically:

1. Replaces `criterion.loss_cfg` with `curriculum.complete`.
2. Updates `criterion.matcher.strategy` to the new `param_match` strategy.
3. Optionally resets: early stopping counter, LR scheduler (with epoch offset), warmup, optimiser moments (Adam $m_t$, $v_t$).

This allows a first phase with simpler / cheaper loss terms (e.g. curve MSE only) before introducing more expensive terms (SSIM, parameter matching) once the model has reached a reasonable initialisation.

---

## 8. Artifact Naming and Directory Layout

### Training Run (output)

```
{base_logdir}/{run_name}/
    best_model.pt               ← complete training state (Checkpoint)
    tensorboard/
        events.out.tfevents.*   ← TensorBoard scalars and histograms
    docs/
        trainer_config.json     ← serialised TrainerConfig
        model_summary.md        ← layer-wise parameter table
        shape_log.md            ← per-layer tensor shape trace
    logs/
        {model_name}_metadata.log
        dataset_pipeline.log
    meta/
        run_summary.json        ← in_channels, out_channels, x_axis_length
        normalization_stats.json← from DatasetPipeline
        dataset_creation_config.json
        crop.json
        patch.json
```

### TensorBoard Scalars Logged

| Tag | Logged at |
|----|-----------|
| `loss/train` | End of each epoch |
| `loss/val` | Each validation epoch |
| `lr/{group_name}` | Each gradient step |
| `lr/warmup_factor` | Each warmup step |
| `train/grad_norm_before_clip` | Each gradient step |
| `train/grad_norm_after_clip` | Each gradient step |
| `train/grad_clip_ratio` | Each gradient step |
| `train/grad_clip_threshold` | Each gradient step |
| `train/grad_norm_dist` | Every `log_histogram_freq` steps |
| `early_stop/counter` | Each validation epoch |
| `early_stop/best_val_loss` | On improvement |
| `loss_components/{stage}/*` | Each eval epoch |
| `loss_weighted/{stage}/*` | Each eval epoch |
| `permutation/{stage}/*` | Each eval epoch |
| `debug/ema_divergence` | Each update (debug mode) |

---

## 9. Inputs and Outputs Summary

### Inputs

| Source | Type | Description |
|--------|------|-------------|
| `TrainerConfig` | Python dataclass | All optimiser, loss, scheduler, and regularisation hyperparameters. |
| `DatasetConfiguration` | Python dataclass | Dataset split definitions, patch settings, normalisation config. |
| `train_loader` | `DataLoader` | Normalised, augmented `(input, gt_params)` patch batches. |
| `val_loader` | `DataLoader` | Normalised `(input, gt_params)` patch batches (no augmentation). |
| `test_loader` | `DataLoader` | As val_loader. |
| `norm_stats` | `Normalizer` | Per-channel `(loc, scale)` for denormalisation inside the loss. |
| `x_axis` | `ndarray (L,)` | Elevation sample coordinates for curve reconstruction. |

### Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| `best_model.pt` | `{run_dir}/` | Complete training state at best validation loss. |
| `train_losses` | return value | List of per-epoch training losses. |
| `val_losses` | return value | List of per-epoch validation losses (NaN for skipped epochs). |
| `best_val_loss` | return value | Scalar: minimum observed validation loss. |
| `docs/trainer_config.json` | run dir | Full configuration for reproducibility. |
| `docs/model_summary.md` | run dir | Parameter count per layer. |
| `docs/shape_log.md` | run dir | Forward-pass tensor shapes. |
| `tensorboard/` | run dir | All training scalars and histograms. |

---

## 10. Canonical Usage

```python
from pathlib import Path
from configuration.training_config import TrainerConfig
from configuration.dataset_config  import (
    DatasetConfiguration, InputConfig, OutputConfig, PatchConfiguration
)
from pipelines.training_pipeline import TrainingPipeline
from tools.split_regions import SplitRegions
from configuration.processing_config import CropRegion

# --- Dataset configuration ---
dataset_config = DatasetConfiguration(
    preprocessing_run_directory = Path("/runs/preproc/run_001"),
    split_regions               = SplitRegions({
        "train" : CropRegion(0,   800, 0, 1000),
        "val"   : CropRegion(800, 950, 0, 1000),
        "test"  : CropRegion(950, 1024, 0, 1000),
    }),
    parameters_path = Path("/runs/params/run_001/params/parameters.npy"),
    patch           = PatchConfiguration(size=(64, 64), stride=32),
    input_config    = InputConfig(use_interferograms=True),
    output_config   = OutputConfig(),
    batch_size      = 16,
    num_workers     = 8,
    n_gaussians     = 2,
)

# --- Trainer configuration (abbreviated) ---
trainer_config = TrainerConfig()   # uses defaults; override fields as needed
trainer_config.io.logdir = "/runs/train"

# --- Run ---
pipeline = TrainingPipeline(
    trainer_config = trainer_config,
    dataset_config = dataset_config,
    model_name     = "unet",
    seed           = 42,
    run_name       = "unet_K2_run01",
)

train_losses, val_losses, best_val_loss = pipeline.run()
```

**To resume from a checkpoint:**

```python
# Inside Trainer (called automatically if checkpoint_path exists):
start_epoch = trainer.checkpoint.load(trainer, str(checkpoint_path))
```

**To run in overfit-debug mode:**

```python
trainer_config.overfit.enabled        = True
trainer_config.overfit.max_steps      = 200
trainer_config.overfit.stop_threshold = 1e-5
```

---

## 11. Public API Reference

### `TrainingPipeline` (`pipeline.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(trainer_config, dataset_config, model_name, model_config, seed, run_name)` | Seeds RNG, creates metadata, instantiates DatasetPipeline. |
| `run` | `(probe_config=None) → (list, list, float)` | Full execution: data, model build, training loop, teardown. |

### `Trainer` (`trainer.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `train` | `(train_loader, val_loader, test_loader) → (list, list, float)` | Main epoch loop. |
| `train_epoch` | `(train_loader, epoch) → float` | Single training epoch; returns average loss. |
| `evaluate` | `(loader, epoch, stage) → dict` | Validation / test evaluation with EMA. |
| `maybe_run_loss_probe` | `(train_loader, probe_config)` | Optional pre-training loss-scale analysis. |

### `Loss` (`loss.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(pred_params, gt_params) → dict` | Compute all active terms; return `{total_loss, components, weighted}`. |
| `reconstruct_gaussians` | `(params) → Tensor` | Build PSD curves from parameter tensor. |

### `Warmup` (`warmup.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `factor` | `() → float` | Current warmup multiplier without advancing state. |
| `step` | `() → float` | Advance counter, log, return factor. |
| `reset` | `()` | Restart warmup from step 0. |
| `is_finished` | `() → bool` | True when warmup is complete or disabled. |

### `Scheduler` (`scheduler.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `step` | `(epoch, metric=None) → list[float]` | Compute new LRs; applies warmup factor if active. |
| `reset` | `(epoch_offset)` | Reset plateau state and offset epoch counter (used after curriculum swap). |

### `EMA` (`ema.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `init` | `(model)` | Initialise shadow from model params. |
| `update` | `(model, step)` | EMA step: $\bar{\theta} \leftarrow \beta\bar{\theta} + (1-\beta)\theta$. |
| `apply_to` | `(model)` | Copy shadow to model; save backup of live params. |
| `restore` | `(model)` | Restore live params from backup. |

### `GradientClipper` (`gradient_clipper.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `maybe_clip` | `(model, global_step) → float` | Compute norm, clip if threshold available, log. |
| `record` | `(grad_norm, global_step)` | Append norm to history; log histogram periodically. |

### `EarlyStopping` (`early_stopping.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(val_loss, model, epoch) → bool` | Update counter; return `True` if training should stop. |
| `restore` | `(model)` | Load best-known model weights into model. |
| `reset` | `()` | Clear counter and best state (used after curriculum swap). |

### `Checkpoint` (`checkpoint.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `step` | `(val_loss, epoch, trainer) → bool` | Save if new minimum; return `True` if saved. |
| `save` | `(trainer, path, epoch)` | Serialise full training state. |
| `load` | `(trainer, path) → int` | Deserialise and restore all state; returns resume epoch. |

### `OverfitManager` (`overfit.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `setup_loaders` | `(train, val, test) → (train, val, test)` | If enabled, replaces loaders with single-batch repeats. |
| `check_stop` | `(train_loss) → bool` | Returns `True` when max steps or loss threshold reached. |

### `TrainingRunMetadata` (`metadata.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `save_trainer_config` | `() → Path` | Writes `docs/trainer_config.json`. |
| `save_run_summary` | `(model_name, in_ch, out_ch, x_len, ...) → Path` | Writes `meta/run_summary.json`. |
| `close` | `()` | Flushes and closes TensorBoard writer. |
