# Tuning Pipeline — Technical Reference

**Package:** `pipelines.tuning_pipeline`  
**Location:** `pipelines/tuning_pipeline/`  
**Role:** Automated, distributed hyperparameter optimisation of SAR tomographic regression networks using Optuna's Tree-structured Parzen Estimator (TPE). Supports two execution modes: a **two-phase sequential strategy** (optimise learning/regularisation parameters first, then architecture) and a **single-phase joint strategy** (optimise all parameters simultaneously). Both modes run multi-GPU parallel trials against a shared persistent SQLite study.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Configuration Layer](#3-configuration-layer)
4. [Component Responsibilities](#4-component-responsibilities)
   - 4.1 [TuningPipeline](#41-tuningpipeline)
   - 4.2 [Phase1Tuner](#42-phase1tuner)
   - 4.3 [Phase2Tuner](#43-phase2tuner)
   - 4.4 [SinglePhaseTuner](#44-singlephastuner)
   - 4.5 [TrialPipeline](#45-trialpipeline)
   - 4.6 [TrialTrainer](#46-trialtrainer)
   - 4.7 [\_sample\_params](#47-_sample_params)
5. [Two-Phase Tuning Strategy](#5-two-phase-tuning-strategy)
6. [Optuna Study Design](#6-optuna-study-design)
   - 6.1 [Sampler — Tree-structured Parzen Estimator](#61-sampler--tree-structured-parzen-estimator)
   - 6.2 [Pruner — Median Pruner](#62-pruner--median-pruner)
   - 6.3 [Persistent SQLite Storage](#63-persistent-sqlite-storage)
7. [Multi-GPU Distributed Execution](#7-multi-gpu-distributed-execution)
8. [Search Space Specification](#8-search-space-specification)
9. [Trial Execution Flow](#9-trial-execution-flow)
10. [Mathematical Formulation](#10-mathematical-formulation)
    - 10.1 [TPE Acquisition Function](#101-tpe-acquisition-function)
    - 10.2 [Median Pruner](#102-median-pruner)
    - 10.3 [Objective](#103-objective)
11. [Artifact Layout and Outputs](#11-artifact-layout-and-outputs)
12. [Inputs and Outputs Summary](#12-inputs-and-outputs-summary)
13. [Canonical Usage](#13-canonical-usage)
14. [Public API Reference](#14-public-api-reference)

---

## 1. Overview

The tuning pipeline wraps the full training pipeline as an Optuna objective and optimises hyperparameters over a configurable number of short-epoch trials. Its design is based on four principles:

1. **Decoupled search spaces.** Each model config class declares two orthogonal search spaces — `tunable_lr_params()` (learning rate, weight decay, regularisation) and `tunable_arch_params()` (layer widths, normalisations, activations) — enabling independent optimisation in each phase.

2. **Sequential two-phase optimisation.** Phase 1 identifies optimal training dynamics parameters; Phase 2 then freezes those results and searches over architectural choices. This reduces the joint search space dimensionality from $D = D_{\text{lr}} + D_{\text{arch}}$ to $\max(D_{\text{lr}}, D_{\text{arch}})$ effective dimensions per study, improving sample efficiency.

3. **Intermediate-value pruning.** Each trial reports its validation loss at every epoch. The Median Pruner terminates unpromising trials early, redirecting compute budget to regions of parameter space that show lower validation loss trajectories.

4. **Multi-GPU parallelism via subprocess workers.** The scheduler process spawns one independent OS-level worker process per GPU. All workers share the same SQLite-backed Optuna study, coordinating via the database with the `constant_liar` strategy to reduce redundancy.

---

## 2. Architecture

```
main/tune.py  (entry point)
       │
       ├─ _scheduler()
       │      for each model_name in CONFIG_REGISTRY:
       │          ┌─ Phase 1 ─────────────────────────────────────────────┐
       │          │  optuna.create_study (TPE + MedianPruner, SQLite)     │
       │          │  _spawn_workers(model, phase=1, n_per_gpu=[...])      │
       │          │      subprocess.Popen(_worker, --gpu, --phase 1)×N    │
       │          │  _wait_workers → poll until all terminate             │
       │          │  load best trial → decode indexed_categorical params  │
       │          │  save phase1_best.json                                │
       │          └───────────────────────────────────────────────────────┘
       │          ┌─ Phase 2 ─────────────────────────────────────────────┐
       │          │  optuna.create_study (TPE + MedianPruner, SQLite)     │
       │          │  _spawn_workers(model, phase=2, n_per_gpu=[...])      │
       │          │      subprocess.Popen(_worker, --gpu, --phase 2)×N    │
       │          │  _wait_workers                                        │
       │          │  load best trial → merge p1+p2 params                │
       │          │  save best_config.json                                │
       │          └───────────────────────────────────────────────────────┘
       │          append → tuning_results.json
       │
       └─ _worker()
              os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
              TuningPipeline(model_name, model_config_cls,
                             base_trainer_config, base_dataset_config,
                             tune_cfg, log_dir, logger)
                  │
                  ├─ run_phase1(study, n_trials)
                  │       Phase1Tuner
                  │           study.optimize(_objective, n_trials)
                  │               for each trial:
                  │                   _sample_params(trial, tunable_lr_params())
                  │                   copy base_trainer_config / base_dataset_config
                  │                   override: epochs, scheduler.epochs, early_stop_patience, logdir
                  │                   TrialPipeline.run()
                  │                       TrainingPipeline.__init__ + run()
                  │                           TrialTrainer._trial_callback(val_loss, epoch)
                  │                               trial.report(val_loss, epoch)
                  │                               trial.should_prune() → raise TrialPruned
                  │                   return best_val_loss
                  │
                  ├─ run_phase2(study, n_trials, best_phase1_params)
                  │       Phase2Tuner
                  │           freeze Phase-1 best params in model_config
                  │           study.optimize(_objective, n_trials)
                  │               for each trial:
                  │                   _sample_params(trial, tunable_arch_params())
                  │                   apply frozen Phase-1 params + sampled arch params
                  │                   TrialPipeline.run() → return best_val_loss
                  │
                  └─ run_single_phase(study, n_trials)
                          SinglePhaseTuner
                              _sample_params(lr_params) ∪ _sample_params(arch_params)
                              TrialPipeline.run() → return best_val_loss
```

---

## 3. Configuration Layer

All tuning behaviour is governed by the `TuningConfig` dataclass hierarchy (`configuration/tuning_config.py`).

### `Phase1TuneConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_trials` | `int` | `100` | Total number of trials for Phase 1 (distributed across all GPUs). |
| `n_epochs` | `int` | `30` | Number of training epochs per trial. Kept short relative to full training to maximise trial throughput. |
| `early_stop_patience` | `int` | `8` | Early stopping patience (in epochs) applied within each trial. |
| `lr_low` | `float` | `1e-5` | Lower bound of the learning rate search range (informational; actual bounds are declared in `tunable_lr_params()`). |
| `lr_high` | `float` | `1e-2` | Upper bound of the learning rate search range. |
| `wd_low` | `float` | `1e-6` | Lower bound of the weight decay range. |
| `wd_high` | `float` | `1e-1` | Upper bound of the weight decay range. |
| `pruner_n_startup_trials` | `int` | `8` | Number of completed trials before the median pruner starts pruning. |
| `pruner_n_warmup_steps` | `int` | `8` | Number of initial epochs within each trial that are exempt from pruning. |

### `Phase2TuneConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_trials` | `int` | `100` | Total Phase-2 trials. |
| `n_epochs` | `int` | `30` | Training epochs per trial. |
| `early_stop_patience` | `int` | `10` | Early stopping patience per trial. |
| `pruner_n_startup_trials` | `int` | `8` | Startup trials before pruning activates. |
| `pruner_n_warmup_steps` | `int` | `8` | Warm-up epochs per trial before pruning activates. |

### `SinglePhaseTuneConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_trials` | `int` | `500` | Total single-phase trials. Higher than two-phase due to larger joint search space. |
| `n_epochs` | `int` | `60` | Training epochs per trial. Longer budget as phase isolation cannot reduce variance. |
| `early_stop_patience` | `int` | `10` | Early stopping patience per trial. |
| `pruner_n_startup_trials` | `int` | `20` | Startup trials before pruning activates. |
| `pruner_n_warmup_steps` | `int` | `20` | Warm-up epochs per trial before pruning activates. |

### `TuningConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `phase1` | `Phase1TuneConfig` | default-constructed | Phase-1 configuration. |
| `phase2` | `Phase2TuneConfig` | default-constructed | Phase-2 configuration. |
| `single_phase` | `SinglePhaseTuneConfig` | default-constructed | Single-phase configuration. |
| `study_storage_dir` | `str` | `/ste/.../logs/tuning` | Root directory for SQLite study databases and logs. |
| `n_gpus` | `int` | `4` | Number of GPUs to use for distributed trial execution. |

---

## 4. Component Responsibilities

### 4.1 `TuningPipeline`

**File:** `pipeline.py`

The central coordinator. Holds shared references (model name, config class, base trainer/dataset configs, tuning config, log directory, logger) and delegates each mode to the appropriate tuner.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `run_phase1` | `(study, n_trials)` | Instantiates `Phase1Tuner` with phase-1 config and calls `tuner.run(study, n_trials)`. |
| `run_phase2` | `(study, n_trials, best_phase1_params)` | Instantiates `Phase2Tuner` with frozen Phase-1 params and calls `tuner.run`. |
| `run_single_phase` | `(study, n_trials)` | Instantiates `SinglePhaseTuner` and calls `tuner.run`. |

The `log_dir` passed to each tuner is scoped by phase:

```
{log_dir}/phase1/   ← Phase1Tuner log root
{log_dir}/phase2/   ← Phase2Tuner log root
{log_dir}/single_phase/  ← SinglePhaseTuner log root
```

---

### 4.2 `Phase1Tuner`

**File:** `phase1_tuner.py`

Optimises the LR/WD parameter group and regularisation hyperparameters. Architecture hyperparameters are left at model-config defaults.

**`_objective(trial)` sequence:**

1. Call `_sample_params(trial, model_config_cls.tunable_lr_params())` to obtain `{param_name: value}`.
2. Instantiate `model_config = model_config_cls()` and `setattr` all sampled values.
3. `deepcopy` both base configs to avoid cross-trial mutation.
4. Override in trainer config: `training.epochs`, `scheduler.epochs`, `early_stopping.patience`, `io.logdir = .../trial_{N:04d}`.
5. Instantiate `TrialPipeline` and call `pipeline.run()`.
6. Return `best_val_loss`.
7. On `TrialPruned`: re-raise. On any other exception: log error, raise `TrialPruned` (treat failed trial as pruned, not crashed).

**`run(study, n_trials)`:** Calls `study.optimize(self._objective, n_trials=n_trials, gc_after_trial=True)`. The `gc_after_trial=True` flag triggers garbage collection after each trial to prevent GPU memory accumulation across the long sequence of trial executions.

---

### 4.3 `Phase2Tuner`

**File:** `phase2_tuner.py`

Optimises architectural hyperparameters, with Phase-1 best learning/regularisation parameters frozen.

**`_objective(trial)` sequence:**

1. Call `_sample_params(trial, model_config_cls.tunable_arch_params())` to sample architecture decisions.
2. Instantiate `model_config = model_config_cls()`.
3. Apply `best_phase1_params` first (frozen): for each key in `best_phase1_params`, `setattr(model_config, k, v)`.
4. Apply `arch_sampled` on top: for each key in `arch_sampled`, `setattr(model_config, k, v)`.
5. Proceed identically to Phase1Tuner (deepcopy configs, override epochs, instantiate `TrialPipeline`, return `best_val_loss`).

The key design choice: Phase-2 trials are evaluated with the **same LR/WD schedule** as the best Phase-1 trial. This ensures that architectural choices are assessed under near-optimal training dynamics, rather than being confounded by a poor LR.

---

### 4.4 `SinglePhaseTuner`

**File:** `single_phase_tuner.py`

Optimises all hyperparameters jointly in a single Optuna study. Intended for cases where the two-phase decomposition assumption (independence of LR and architecture) does not hold, or when the available compute budget permits exhaustive joint search.

**`_objective(trial)` sequence:**

1. Sample both spaces simultaneously:
   ```python
   all_params = {
       **_sample_params(trial, model_config_cls.tunable_lr_params()),
       **_sample_params(trial, model_config_cls.tunable_arch_params()),
   }
   ```
2. Apply all sampled parameters to a freshly instantiated `model_config`.
3. Proceed identically to Phase1Tuner.

The combined search space has dimensionality $D = D_{\text{lr}} + D_{\text{arch}}$, which motivates the larger default trial budget (500 vs 100 per phase).

---

### 4.5 `TrialPipeline`

**File:** `trial_pipeline.py`

A minimal subclass of `TrainingPipeline` that injects Optuna trial awareness at the trainer level. The only override is `_make_trainer`, which instantiates `TrialTrainer` instead of the base `Trainer`:

```python
class TrialPipeline(TrainingPipeline):
    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _make_trainer(self, model, model_cfg, x_axis, norm_stats):
        return TrialTrainer(
            ...,
            trial = self._trial,
        )
```

All other pipeline stages (data loading, normalisation, model construction, checkpoint saving) are inherited unchanged from `TrainingPipeline`. The return value of `run()` is `(run_directory, model, best_val_loss)`.

---

### 4.6 `TrialTrainer`

**File:** `trial_trainer.py`

A minimal subclass of `Trainer` that adds the Optuna pruning callback. The only override is `_trial_callback`, called by the base `Trainer` at the end of each validation step:

```python
def _trial_callback(self, val_loss: float, epoch: int) -> None:
    self._trial.report(val_loss, epoch)
    if self._trial.should_prune():
        raise optuna.exceptions.TrialPruned()
```

This method:
1. Reports the intermediate value to the Optuna study database.
2. Queries whether the trial should be pruned (based on the Median Pruner's decision).
3. If pruning is requested, raises `TrialPruned`, which unwinds the training loop cleanly and is caught by the tuner's `_objective`.

The base `Trainer` must call `self._trial_callback(val_loss, epoch)` at each validation step for this mechanism to operate. All checkpoint-saving, EMA updates, and logging proceed normally before the callback is invoked.

---

### 4.7 `_sample_params`

**File:** `phase1_tuner.py` (imported by `phase2_tuner.py` and `single_phase_tuner.py`)

A free function that translates a search space specification dictionary into Optuna API calls:

```python
def _sample_params(trial: optuna.Trial, space: dict) -> dict:
```

**Supported parameter types:**

| `"type"` | Optuna API | Notes |
|----------|-----------|-------|
| `"float"` | `trial.suggest_float(name, low, high, log=...)` | `log=True` samples on a log scale, appropriate for LR and WD. |
| `"categorical"` | `trial.suggest_categorical(name, choices)` | Choices must be hashable; used for string-valued hyperparameters (activation, normalisation). |
| `"indexed_categorical"` | `trial.suggest_categorical(name + "__idx", range(len(choices)))` then `choices[idx]` | Used for list-valued hyperparameters (e.g., `features = [64, 128, 256, 512]`) that cannot be directly stored as Optuna categorical values. The integer index is persisted in the study; the value is reconstructed on retrieval. |

The `indexed_categorical` → `__idx` encoding requires a decoding step in the scheduler after Phase-1 concludes. The scheduler iterates `raw_p1_params`, detects `k.endswith("__idx")` keys, looks up the corresponding spec from `tunable_lr_params()`, and writes the decoded list value to `phase1_best.json`.

---

## 5. Two-Phase Tuning Strategy

The sequential two-phase approach decomposes the joint optimisation problem:

$$
\theta^* = \operatorname{argmin}_{\theta \in \Theta_{\text{lr}} \times \Theta_{\text{arch}}} \mathcal{L}_{\text{val}}(f_\theta)
$$

into two sequential problems:

$$
\theta^*_{\text{lr}} = \operatorname{argmin}_{\theta_{\text{lr}} \in \Theta_{\text{lr}}} \mathcal{L}_{\text{val}}\!\left(f_{\theta_{\text{lr}},\, \theta_{\text{arch}}^{(0)}}\right)
$$

$$
\theta^*_{\text{arch}} = \operatorname{argmin}_{\theta_{\text{arch}} \in \Theta_{\text{arch}}} \mathcal{L}_{\text{val}}\!\left(f_{\theta^*_{\text{lr}},\, \theta_{\text{arch}}}\right)
$$

where $\theta_{\text{arch}}^{(0)}$ denotes the default model config values. This is a greedy coordinate-wise optimisation and is exact only under the independence assumption:

$$
\mathcal{L}_{\text{val}}(\theta_{\text{lr}}, \theta_{\text{arch}}) \approx g(\theta_{\text{lr}}) + h(\theta_{\text{arch}})
$$

In practice, this assumption is approximately valid because the optimal learning dynamics (LR warmup, cosine schedule, weight decay) are relatively insensitive to specific architectural choices of moderate width, and architecture quality is primarily determined by representational capacity rather than exact LR values. The two-phase strategy reduces the total number of trials required from $\mathcal{O}(n_{\text{joint}})$ to $\mathcal{O}(n_1 + n_2)$ while maintaining close-to-optimal results.

**Phase 1 search space (example — UNet):**

| Parameter | Type | Range | Scale |
|-----------|------|-------|-------|
| `encoder_lr` | float | $[10^{-5}, 10^{-2}]$ | log |
| `bottleneck_lr` | float | $[10^{-5}, 10^{-2}]$ | log |
| `decoder_lr` | float | $[10^{-5}, 10^{-2}]$ | log |
| `output_head_lr` | float | $[10^{-5}, 10^{-2}]$ | log |
| `encoder_wd` | float | $[10^{-6}, 10^{-1}]$ | log |
| `bottleneck_wd` | float | $[10^{-6}, 10^{-1}]$ | log |
| `decoder_wd` | float | $[10^{-6}, 10^{-1}]$ | log |
| `output_head_wd` | float | $[10^{-6}, 10^{-1}]$ | log |
| `dropout` | float | $[0.0, 0.5]$ | linear |

**Phase 2 search space (example — UNet):**

| Parameter | Type | Choices |
|-----------|------|---------|
| `features` | indexed\_categorical | `[32,64,128,256]`, `[64,128,256,512]`, `[48,96,192,384]`, `[64,128,256,512,1024]` |
| `bottleneck_factor` | categorical | `1`, `2`, `4` |
| `activation` | categorical | `"relu"`, `"leaky_relu"`, `"gelu"`, `"silu"` |
| `normalization` | categorical | `"batch"`, `"instance"`, `"group"` |
| `upsample_mode` | categorical | `"convtranspose"`, `"bilinear"` |

---

## 6. Optuna Study Design

### 6.1 Sampler — Tree-structured Parzen Estimator

The TPE sampler (`optuna.samplers.TPESampler`) is used for both Phase-1 and Phase-2 studies.

**Configuration:**

| Option | Value | Rationale |
|--------|-------|-----------|
| `n_startup_trials` | `8` (Phase 1/2) / `20` (single) | Random exploration before model fitting. |
| `multivariate` | `True` | Models joint parameter dependencies; superior to independent marginals for correlated hyperparameters (e.g., LR and WD). |
| `constant_liar` | `True` | In a parallel study, pending (not yet completed) trials are treated as having a constant "liar" loss equal to the current best, discouraging redundant parameter proposals across GPU workers. |
| `seed` | `42` | Reproducibility across runs. |

### 6.2 Pruner — Median Pruner

`optuna.pruners.MedianPruner` terminates a trial at epoch $t$ if its reported validation loss exceeds the median of all completed trial losses at step $t$.

**Configuration:**

| Option | Value | Effect |
|--------|-------|--------|
| `n_startup_trials` | `8` / `20` | Minimum number of completed trials before any pruning decision is made. |
| `n_warmup_steps` | `8` / `20` | Minimum number of epochs within a trial before it can be pruned. |

The combination of `n_startup_trials` and `n_warmup_steps` prevents premature pruning during the early phase of the study when the reference distribution is poorly estimated and during LR warmup when most trials exhibit high losses.

### 6.3 Persistent SQLite Storage

All study data (trial parameters, intermediate values, trial states, pruning decisions) are stored in a SQLite file:

```
{study_storage_dir}/{tag}/optuna.db
```

Study names follow the pattern `{model_name}_phase{1|2}_{tag}`. Studies are created with `load_if_exists=True`, enabling crash recovery: if a worker dies, its in-progress trials are automatically marked `FAILED` by Optuna's heartbeat mechanism and the study resumes cleanly when workers are restarted.

---

## 7. Multi-GPU Distributed Execution

The entry point `main/tune.py` implements a **scheduler–worker** process model.

### Scheduler (`_scheduler`)

The scheduler runs on the head node. For each model and each phase, it:

1. Creates the Optuna study (once, idempotently via `load_if_exists=True`).
2. Distributes the total trial budget across $N_{\text{GPU}}$ workers using `_distribute_trials`:

$$
n_i = \left\lfloor \frac{N_{\text{trials}}}{N_{\text{GPU}}} \right\rfloor + \mathbb{1}[i < N_{\text{trials}} \bmod N_{\text{GPU}}]
$$

This ensures the load is balanced with at most 1 extra trial for the first few workers.

3. Spawns one `subprocess.Popen` per GPU, passing `--worker --gpu {i} --phase {p} --n-trials {n_i} --study-name {name} --storage-url {url}`.
4. Polls all processes at 5-second intervals with `proc.poll()`.
5. After all processes terminate, loads the best trial from the study and saves results.

Worker stdout/stderr are redirected to per-GPU log files at `{run_dir}/{model_name}/phase{p}_gpu{i}.log`.

### Worker (`_worker`)

Each worker:

1. Sets `os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)` to isolate the GPU.
2. Sets thread counts (`MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `NUMEXPR_NUM_THREADS` = 4) to prevent CPU over-subscription.
3. Loads the shared study via `optuna.load_study(study_name, storage)`.
4. Instantiates `TuningPipeline` and calls `run_phase1` or `run_phase2`.
5. `study.optimize` internally calls `_objective` for `n_trials` trials. Multiple workers calling `optimize` on the same study concurrently is fully supported by Optuna's SQLite backend.

### Trial Isolation

Each trial runs in its own sub-directory:

```
{log_dir}/{model_name}/phase{p}/trial_{N:04d}/
```

A deep copy of both `trainer_config` and `dataset_config` is made at the start of each `_objective` call to ensure complete isolation between trials — no state leaks from one trial to the next.

---

## 8. Search Space Specification

Search spaces are declared as class methods on model config classes (in `configuration/models_config.py`):

```python
@classmethod
def tunable_lr_params(cls) -> dict[str, dict]: ...

@classmethod
def tunable_arch_params(cls) -> dict[str, dict]: ...
```

Each returns a dictionary mapping parameter names to specification dicts. The specification dict has a mandatory `"type"` key and type-specific fields:

**Float parameter:**
```python
"learning_rate": {
    "type": "float",
    "low": 1e-5,
    "high": 1e-2,
    "log": True    # optional, default False
}
```

**Categorical parameter:**
```python
"activation": {
    "type": "categorical",
    "choices": ["relu", "leaky_relu", "gelu", "silu"]
}
```

**Indexed categorical parameter** (for non-hashable values such as lists):
```python
"features": {
    "type": "indexed_categorical",
    "choices": [
        [32, 64, 128, 256],
        [64, 128, 256, 512],
        [48, 96, 192, 384],
        [64, 128, 256, 512, 1024],
    ]
}
```

Optuna stores `"features__idx"` (an integer 0–3) in the trial; the tuner reconstructs `"features"` as the actual list during Phase-2 warm-start and the scheduler decodes it for `phase1_best.json`.

All models in `CONFIG_REGISTRY` must implement both class methods for the tuning pipeline to function. The registry maps model name strings to config classes.

---

## 9. Trial Execution Flow

For a single Optuna trial $t$ in Phase 1:

```
Phase1Tuner._objective(trial)
    │
    ├─ _sample_params(trial, tunable_lr_params())
    │       → {lr_1: v_1, lr_2: v_2, ..., dropout: v_d}
    │
    ├─ model_config = model_config_cls()
    │   setattr(model_config, k, v) for all sampled params
    │
    ├─ trainer_cfg = deepcopy(base_trainer_config)
    │   trainer_cfg.training.epochs         = n_epochs (30)
    │   trainer_cfg.scheduler.epochs        = n_epochs (30)
    │   trainer_cfg.early_stopping.patience = early_stop_patience (8)
    │   trainer_cfg.io.logdir               = .../trial_0042/
    │
    ├─ dataset_cfg = deepcopy(base_dataset_config)
    │
    ├─ TrialPipeline(trainer_cfg, dataset_cfg, model_name, model_config,
    │                seed=trial.number, run_name="phase1_trial_0042",
    │                trial=trial)
    │
    └─ pipeline.run()
           ├─ TrainingPipeline._setup → create dirs, logger
           ├─ TrainingPipeline._build_dataset → PatchDataset, DataLoader
           ├─ TrainingPipeline._build_model → model.to(device)
           ├─ TrainingPipeline._run_training
           │       for epoch in range(n_epochs):
           │           train one epoch
           │           val_loss = validate()
           │           TrialTrainer._trial_callback(val_loss, epoch)
           │               trial.report(val_loss, epoch)       ← write to SQLite
           │               if trial.should_prune():            ← query SQLite
           │                   raise TrialPruned               ← unwind
           │           early_stopping.step(val_loss)
           │           if early_stopped: break
           └─ return (run_dir, model, best_val_loss)
```

The objective function returns `best_val_loss` (the minimum validation loss achieved across all epochs in the trial). Optuna stores this as the trial's final value, minimises it, and uses it as the optimisation target.

---

## 10. Mathematical Formulation

### 10.1 TPE Acquisition Function

TPE models the objective $f(\theta)$ implicitly by building two density estimates from the history of completed trials:

- $\ell(\theta)$: kernel density estimate (KDE) over parameters of trials with $f(\theta) < f^*_\gamma$ (the $\gamma$-th quantile of observed losses, typically $\gamma = 0.25$).
- $g(\theta)$: KDE over parameters of all remaining trials (those with $f(\theta) \geq f^*_\gamma$).

The next trial's parameters are chosen to maximise the Expected Improvement (EI) surrogate:

$$
\theta^{(t+1)} = \operatorname{argmax}_\theta \frac{\ell(\theta)}{g(\theta)}
$$

This ratio is maximised by drawing candidates from $\ell(\theta)$ and selecting the one with the highest $\ell/g$ ratio, which requires evaluating only the two KDEs rather than a full probabilistic model.

With `multivariate=True`, Optuna constructs a **multivariate TPE** where the density estimates model the joint distribution $p(\theta_1, \ldots, \theta_D)$ rather than independent marginals $\prod_i p(\theta_i)$. This captures parameter correlations (e.g., the interaction between encoder and decoder learning rates) at the cost of requiring more data to estimate accurately.

With `constant_liar=True`, pending trials (running on other GPU workers) are assigned a "liar" value equal to the current best loss, inserting phantom completed trials into the density estimate. This discourages the sampler from proposing parameter values that are already being evaluated by another worker.

### 10.2 Median Pruner

Let $\{f_i(t)\}_{i=1}^{M}$ be the validation losses at epoch $t$ for all completed trials (those that ran past epoch $t$). The median reference value is:

$$
\bar{f}(t) = \text{median}\!\left(\{f_i(t)\}_{i=1}^{M}\right)
$$

Trial $j$ is pruned at epoch $t$ if:

$$
f_j(t) > \bar{f}(t) \quad \text{and} \quad n_{\text{completed}} \geq n_{\text{startup}} \quad \text{and} \quad t \geq n_{\text{warmup}}
$$

Pruning is applied only once sufficient reference trials are available (`n_startup_trials`) and only after an initial training phase where the learning rate warmup is active (`n_warmup_steps`).

### 10.3 Objective

The objective function $f(\theta)$ evaluated for a given set of hyperparameters $\theta$ is the **best validation loss** achieved during the trial's training run:

$$
f(\theta) = \min_{e=1,\ldots,E} \mathcal{L}_{\text{val}}^{(e)}(\theta)
$$

where $\mathcal{L}_{\text{val}}^{(e)}$ is the validation loss at epoch $e$, defined by the training pipeline's loss function (weighted Gaussian parameter regression loss). The outer minimisation is implicit: it is the value stored as `best_val_loss` by the `Trainer`.

The study is configured as a **minimisation** study (`direction="minimize"`):

$$
\theta^* = \operatorname{argmin}_\theta f(\theta)
$$

---

## 11. Artifact Layout and Outputs

### Per-Run (within each trial)

```
{log_base_dir}/{tag}/{model_name}/
    phase1/
        trial_0000/   ← full TrainingPipeline output (model, logs, meta)
        trial_0001/
        ...
    phase2/
        trial_0000/
        ...
    phase1_gpu0.log   ← stdout/stderr of GPU 0 Phase-1 worker
    phase1_gpu1.log
    phase1_gpu2.log
    phase1_gpu3.log
    phase2_gpu0.log
    ...
    phase1_best.json   ← best Phase-1 trial params (decoded)
    best_config.json   ← merged best Phase-1 + Phase-2 params + val losses
```

### Study Level

```
{log_base_dir}/{tag}/
    optuna.db          ← SQLite study database (all trials, all models)
    tuning_results.json ← summary: one entry per model
    tune_scheduler.log  ← scheduler process log
```

### `phase1_best.json` schema

```json
{
  "encoder_lr": 0.000312,
  "bottleneck_lr": 0.000891,
  "decoder_lr": 0.000234,
  "output_head_lr": 0.000512,
  "encoder_wd": 0.0000234,
  "bottleneck_wd": 0.0000891,
  "decoder_wd": 0.0000456,
  "output_head_wd": 0.0000123,
  "dropout": 0.157
}
```

Note: `indexed_categorical` values are decoded to the actual list (e.g., `"features": [64, 128, 256, 512]`), not the integer index, before being written to this file.

### `best_config.json` schema

```json
{
  "model"          : "UNet",
  "phase1_val_loss": 0.003421,
  "phase2_val_loss": 0.002987,
  "params"         : {
    "encoder_lr"      : 0.000312,
    "features__idx"   : 1,
    "activation"      : "gelu",
    "normalization"   : "batch",
    "upsample_mode"   : "bilinear",
    "bottleneck_factor": 2
  }
}
```

### `tuning_results.json` schema

```json
[
  {
    "model"          : "UNet",
    "status"         : "DONE",
    "phase1_val_loss": 0.003421,
    "phase2_val_loss": 0.002987,
    "best_config"    : "/ste/.../UNet/best_config.json"
  },
  {
    "model"          : "UNetMultiHead",
    "status"         : "PARTIAL",
    "phase1_val_loss": 0.004102,
    "phase2_val_loss": null,
    "best_config"    : null
  }
]
```

Status is `"DONE"` if both phases completed without failures, `"PARTIAL"` if Phase-1 succeeded but Phase-2 had worker failures.

---

## 12. Inputs and Outputs Summary

### Inputs

| Artifact | Source | Description |
|----------|--------|-------------|
| `TrainerConfig` | `_build_base_configs` (tune.py) | Base training configuration. Overridden per trial: `epochs`, `scheduler.epochs`, `early_stop_patience`, `io.logdir`. |
| `DatasetConfiguration` | `_build_base_configs` (tune.py) | Dataset/DataLoader configuration. Deep-copied per trial; not mutated. |
| `model_config_cls` | `CONFIG_REGISTRY[model_name]` | Model config class. Must implement `tunable_lr_params()` and `tunable_arch_params()`. |
| `TuningConfig` | Instantiated in `_worker` | All tuning hyperparameters. |
| `optuna.Study` | Loaded in `_worker` from SQLite | Shared Optuna study; provides sampler state and trial history. |
| Pre-processed dataset | Filesystem | SAR tomographic `.npy` arrays (same as training pipeline). |

### Outputs

| Artifact | Type | Description |
|----------|------|-------------|
| `optuna.db` | SQLite | Complete Optuna study (all trials, parameters, losses, pruning decisions). |
| `phase1_best.json` | JSON | Decoded best Phase-1 hyperparameters per model. |
| `best_config.json` | JSON | Merged best Phase-1 + Phase-2 hyperparameters + validation losses per model. |
| `tuning_results.json` | JSON | Per-model tuning summary with status and best losses. |
| `phase{p}_gpu{i}.log` | text | Per-GPU worker stdout/stderr. |
| Trial run directories | filesystem | Complete `TrainingPipeline` output per trial (checkpoint, logs, meta). Useful for post-hoc analysis of any trial's trained model. |

---

## 13. Canonical Usage

### Two-Phase Tuning (via `main/tune.py`)

```bash
# Tune all models, distributing Phase-1 and Phase-2 trials across 4 GPUs
python main/tune.py

# Tune a single model
python main/tune.py --model UNet
```

The script runs in scheduler mode by default. It creates an Optuna SQLite study, spawns GPU workers, waits for their completion, and saves results.

### Programmatic API (within Python)

```python
import optuna
from configuration.tuning_config import TuningConfig
from pipelines.tuning_pipeline.pipeline import TuningPipeline
from tools.logger import Logger

# Assume base configs and model_config_cls are already built
tune_cfg = TuningConfig()
logger   = Logger(log_dir="/runs/tuning/exp01", name="tuner")

pipeline = TuningPipeline(
    model_name          = "UNet",
    model_config_cls    = UNetConfig,
    base_trainer_config = trainer_cfg,
    base_dataset_config = dataset_cfg,
    tune_cfg            = tune_cfg,
    log_dir             = "/runs/tuning/exp01/UNet",
    logger              = logger,
)

# Phase 1
study_p1 = optuna.create_study(
    study_name = "UNet_phase1",
    storage    = "sqlite:////runs/tuning/exp01/optuna.db",
    direction  = "minimize",
    sampler    = optuna.samplers.TPESampler(multivariate=True, constant_liar=True),
    pruner     = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=8),
    load_if_exists = True,
)
pipeline.run_phase1(study_p1, n_trials=100)

best_p1 = study_p1.best_trial.params  # decode indexed_categorical if needed

# Phase 2
study_p2 = optuna.create_study(
    study_name = "UNet_phase2",
    storage    = "sqlite:////runs/tuning/exp01/optuna.db",
    direction  = "minimize",
    sampler    = optuna.samplers.TPESampler(multivariate=True, constant_liar=True),
    pruner     = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=8),
    load_if_exists = True,
)
pipeline.run_phase2(study_p2, n_trials=100, best_phase1_params=best_p1)
```

### Applying Best Parameters to a Full Training Run

After tuning, load the best config and use it for a full-duration training run via the standard training pipeline:

```python
import json
from configuration.models_config import UNetConfig

with open("/runs/tuning/exp01/UNet/best_config.json") as f:
    result = json.load(f)

model_cfg = UNetConfig()
for k, v in result["params"].items():
    if hasattr(model_cfg, k):
        setattr(model_cfg, k, v)

# Proceed with full TrainingPipeline using model_cfg
```

---

## 14. Public API Reference

### `TuningPipeline` (`pipeline.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(model_name, model_config_cls, base_trainer_config, base_dataset_config, tune_cfg, log_dir, logger)` | Store references; no I/O. |
| `run_phase1` | `(study: optuna.Study, n_trials: int)` | Run Phase-1 optimisation over `tunable_lr_params()`. |
| `run_phase2` | `(study: optuna.Study, n_trials: int, best_phase1_params: dict)` | Run Phase-2 optimisation over `tunable_arch_params()` with frozen Phase-1 params. |
| `run_single_phase` | `(study: optuna.Study, n_trials: int)` | Run joint optimisation over both parameter spaces simultaneously. |

### `Phase1Tuner` (`phase1_tuner.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `(study: optuna.Study, n_trials: int)` | Call `study.optimize(_objective, n_trials, gc_after_trial=True)`. |

### `Phase2Tuner` (`phase2_tuner.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `(study: optuna.Study, n_trials: int)` | Call `study.optimize(_objective, n_trials, gc_after_trial=True)`. |

### `SinglePhaseTuner` (`single_phase_tuner.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `(study: optuna.Study, n_trials: int)` | Call `study.optimize(_objective, n_trials, gc_after_trial=True)`. |

### `TrialPipeline` (`trial_pipeline.py`)

Inherits all public methods from `TrainingPipeline`. The only override is the internal `_make_trainer` factory. The public interface is identical to `TrainingPipeline.run()`.

### `TrialTrainer` (`trial_trainer.py`)

Inherits all public methods from `Trainer`. The only addition is the internal `_trial_callback(val_loss, epoch)` method.

### `_sample_params` (`phase1_tuner.py`)

| Function | Signature | Description |
|----------|-----------|-------------|
| `_sample_params` | `(trial: optuna.Trial, space: dict) → dict` | Sample all parameters in `space` using the appropriate Optuna suggest method. Returns `{name: value}`. |
