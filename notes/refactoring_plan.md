# DLR-TomoSAR Refactoring Plan

Goal: reduce file count by merging components that work together or sequentially (multiple classes per file), extract logic repeated across pipelines into shared components, and remove dead code. Optimized for readability, maintainability, ease of adding features, and trackability.

Scope: all pipelines, main entry points, configuration, tools, models, scripts. Webui is untouched except for the launch.js sync required by configuration changes.

Net effect: ~198 Python files to ~125 files; ~31300 lines to ~28000 lines. No behavior changes.

---

## Constraints and ground rules

1. Every phase leaves the repository runnable. Verification after each phase: `python -m compileall .`, `--help` on every `main/` entry point, plus one fast end-to-end check (`main/overfit_test.py` on a small model).
2. Public class names and CLI flags stay stable. Benchmark and cross-validation spawn subprocesses through `main/` scripts via `GpuQueue` job commands (`--model`, `--fold`); entry script names and flags must not change.
3. Config dataclass field names stay stable. `ConfigCli.load_resolved()` reads resolved configs from old runs (`main/benchmark.py:40`, `main/cross_validate.py:40`, `main/tune.py:382`); renaming fields breaks resume and post-hoc comparison.
4. The webui derives panel layout from config dataclass blank-line blocks plus `FIELD_TAXONOMY`/`STACK_PAIRS` in `webui/launch.js`. Any configuration merge must preserve the blank-line block structure and update launch.js in the same commit.
5. `processing_pipeline/tomogram_worker.py` stays a separate module: `run_pyrat` must be importable at top level for `ProcessPoolExecutor` pickling.
6. Policy update required: the vault `CLAUDE.md` rule "Components are separated into distinct classes and distinct files" is superseded by this plan. Amend to: "Related components that work together or sequentially live in one module; unrelated components in distinct modules." Same amendment in `coding_style.md` if it is ever extended with file-layout rules.

---

## Phase 0 — Safety net

- Create branch `refactor/file-consolidation`.
- `git init` is already done; commit per phase with one commit per merge group so every move is trackable in history.
- Record the baseline: `python -m compileall .` and `--help` of all 12 entry points must pass before starting.

---

## Phase 1 — Dead code removal (verified)

Confirmed dead (zero importers, checked against `tools/__init__.py` re-export path):

| File | Lines | Evidence |
|---|---|---|
| `tools/live_monitor.py` | 43 | Re-exported by `tools/logger.py:28` and `tools/__init__.py:3`, never consumed |
| `tools/gaussian_mask.py` | 43 | No importers anywhere |
| `tools/gaussian_model.py` | 40 | No importers; scripts reference a different, stale `pipelines.param_extraction_pipeline.gaussian_model` |
| `scripts/_debug_toy.py` | 45 | Debug scratch file |

NOT dead, despite zero direct module imports (consumed through `from tools import ...`):
`tools/tracker.py` (trainer.py:10,46), `tools/resource_monitor.py` (trainer.py:74), `tools/permutation_metrics.py` (trainer.py:75), `tools/detach.py` (config_cli.py:9,42). These stay.

Stale scripts: `scripts/validate_gpu_vs_cpu.py`, `scripts/tomo_compare_ngauss_ref.py:19`, `scripts/tune_adam.py:16` import from the non-existent `pipelines.param_extraction_pipeline` (renamed to `param_pipeline`). Decide per script: fix the import path (one line) or delete if obsolete. Default: fix the path, do not delete reference scripts.

Also remove the unused re-exports from `tools/logger.py:28-34` once Phase 2 settles the final tools layout.

---

## Phase 2 — Shared foundation (new `pipelines/shared/`)

The duplication that exists in 5+ pipelines, extracted once. Four small modules:

### 2a. `pipelines/shared/io.py` (~60 lines)
- `save_json(data, path, logger=None)` — replaces the 15+ verbatim copies of `path.parent.mkdir(...)` + `json.dump(..., indent=4, default=str)` (training/metadata.py:65-69, dataset/metadata.py:32-58, processing/metadata.py:77-81, param/metadata.py:33-34, inference/metrics.py, tune.py:175-234, overfit_test.py:89-90).
- `save_text_metadata(entries, path, logger=None)` — the `k: v` line writer from processing/metadata.py.
- `ensure_dir(path)` — the 30+ `mkdir(parents=True, exist_ok=True)` call sites.

### 2b. `pipelines/shared/metadata.py` (~80 lines)
- `MetadataBase` class: holds `(config, logger)`, provides `_save_json`, `_save_text`, timestamp helper, directory-structure creation hook.
- The five per-pipeline metadata classes become thin subclasses (75-85 percent of their bodies is this shared pattern). Per-pipeline metadata classes then merge into their pipeline.py files (Phase 3); they no longer justify standalone files.

### 2c. `pipelines/shared/plotting.py` (~120 lines)
- `SCIENTIFIC_RC` — single canonical dict (currently two 95-percent-identical copies: inference/plots.py:17 vs param/plots.py:19; reconcile the font-size and pdf.fonttype differences, keep the stricter publication settings).
- `PlotBase` class: `_apply_style()`, `_save()` (identical at inference/plots.py:72 and param/plots.py:83), `_shared_clim()` (identical logic at inference/plots.py:64 and param/plots.py:90 with renamed kwargs), `_cmap_with_bad()` (param/plots.py:98), `_triple_panel()` (from inference PlotTools, also used by animation.py).
- Both `Ploter` and `FittingResultPlotter` inherit from `PlotBase`.

### 2d. `pipelines/shared/orchestration.py` (~250 lines)
The benchmark and cross-validation pipelines are structural twins (config_factory 95 percent shared, workers 85 percent, training stages 85 percent, inference stages 80 percent, pipeline orchestrators 75 percent). Formalize the three emergent patterns:
- `GpuQueue`, `GpuJob`, `GpuJobResult` — moved verbatim from `benchmark_pipeline/gpu_queue.py` (already shared de facto by both pipelines).
- `ExperimentStage` base: the five-step pattern implemented five times (cache check via resume flag, split cached/pending, build jobs, `queue.run(jobs)`, merge + write JSON + log table). Abstract hooks: `iteration_items()`, `make_job(item)`, `checkpoint_dir(item)`.
- `ExperimentCollector` base: directory iteration + metadata/checkpoint/metrics loading (TrialCollector and FoldCollector pattern).

Deliberately NOT extracted (analyzed, rejected):
- inference vs param `metrics.py` — different domains (curve quality vs reconstruction quality), only tiny percentile helpers overlap; those go into `shared/io.py` or stay local.
- A universal `PipelineBase` for all eight `pipeline.py` orchestrators — the shape similarity is superficial; a base class would add indirection without removing meaningful code. The shared metadata/io components already capture the real duplication.

### 2e. Tools consolidation (`tools/` 20 files to 10)
- Merge `crop_region.py` + `split_regions.py` into `tools/regions.py`. Add two methods to `CropRegion` that centralize crop arithmetic currently duplicated in dataset/crop.py:28-42 and processing/tomogram.py:37-59: `local_slices(region, global_crop)` and `subdivide_by_azimuth(crop, max_width)`.
- Merge `gaussian_mixture.py` + `gaussian_utils.py` + inference's `reconstruction.py` (GaussianReconstructor) into `tools/gaussians.py` — Gaussian math is a cross-pipeline concern (training loss, inference predictor/plots, param metrics/best_k, physics check).
- Move single-consumer tools into their consumer: `shape_logger.py` + `model_summary.py` into `training_pipeline` docs module; `param_matcher.py` into training loss module; `loss_scale_probe.py` into `training_pipeline` (benchmark workers reach it through the training pipeline anyway); `representation.py` into `configuration/`.
- Inline `Detacher` (detach.py, only consumer config_cli.py:9) into `config_cli.py`.
- Keep in `tools/`: `logger.py`, `tracker.py`, `resource_monitor.py`, `permutation_metrics.py`, `config_cli.py`, `regions.py`, `gaussians.py`, `tomo_geometry.py`, `__init__.py`. Update `tools/__init__.py` re-exports atomically with all importers.

---

## Phase 3 — Per-pipeline file merges

Merges are between components that work together or sequentially. Line counts are the sums of the current files; code bodies move verbatim. Each pipeline keeps its `pipeline.py` as the orchestrator entry.

### 3a. training_pipeline: 19 files to 8

| New file | Absorbs | Lines |
|---|---|---|
| `pipeline.py` | TrainingPipeline + SingleTrainRunner (single_run.py) + TrainingRunMetadata (metadata.py) | ~290 |
| `trainer.py` | Trainer + TrainStep (train_step.py) + MetricAggregator (metric_aggregator.py) | ~480 |
| `loss.py` | Loss + LossComponents (loss_components.py) + PhysicsComponents (physics_components.py) + ParamMatcher (from tools) | ~570 |
| `callbacks.py` | Warmup + Scheduler + EarlyStopping + EMA + GradientClipper | ~550 |
| `control.py` | CurriculumController (curriculum.py) + OverfitManager (overfit.py) + Checkpoint (checkpoint.py) | ~150 |
| `docs.py` | TrainingDocs (training_docs.py) + ShapeLogger + ModelSummary (from tools) + LossScaleProbe (from tools) | ~380 |
| `__init__.py` | exports unchanged | |

Rationale: callbacks all share the init/step/reset/state_dict interface and are reset together by the curriculum controller; loss components are only ever called by Loss; TrainStep and MetricAggregator exist solely inside Trainer's epoch loop; control groups the three classes that decide when training stops, swaps, or saves.

### 3b. tuning_pipeline: 8 files to 4

| New file | Absorbs | Lines |
|---|---|---|
| `pipeline.py` | TuningPipeline | 57 |
| `tuners.py` | BaseTuner + Phase1Tuner + Phase2Tuner + ParamSampler | ~180 |
| `trial.py` | TrialPipeline (trial_pipeline.py) + TrialTrainer (trial_trainer.py) | ~45 |

The inheritance reuse of training_pipeline is already correct (TrialTrainer overrides one callback, TrialPipeline overrides one factory); no logic changes.

### 3c. inference_pipeline: 14 files to 8

| New file | Absorbs | Lines |
|---|---|---|
| `pipeline.py` | InferencePipeline + InferenceMetadata (metadata.py) + Result (types.py) | ~220 |
| `loader.py` | RunLoader + Run + ModelWrapper (wrapper.py) | ~320 |
| `predictor.py` | Predictor + CubeStitcher (stitching.py) | ~360 |
| `metrics.py` | unchanged | 478 |
| `plots.py` | Ploter on PlotBase; PlotTools statics fold into shared/plotting | ~750 |
| `figures.py` | FigureComposer + Animator (animation.py) | ~510 |
| `report.py` | Report + ReportPayloadBuilder, unchanged | 415 |

`reconstruction.py` leaves for `tools/gaussians.py` (Phase 2e). `GridInfo` import in stitching code points at dataset_pipeline's new `spatial.py`.

### 3d. param_pipeline: 11 files to 6

| New file | Absorbs | Lines |
|---|---|---|
| `pipeline.py` | ParamExtractionPipeline + ExtractionMetadataManager (metadata.py) + ParameterIO (artifact_io.py) | ~170 |
| `fitting.py` | ParameterExtractor + PeakInitialiser (peak_init.py) + BestKSelector (best_k.py) | ~240 |
| `sigma.py` | SigmaFittingExtractor (sigma_fitting.py) + SigmaScan/SigmaAdamKernel/PmapSigmaAdamKernel (sigma_kernels.py) | ~520 |
| `metrics.py` | unchanged | 143 |
| `plots.py` | FittingResultPlotter on PlotBase | ~480 |

### 3e. dataset_pipeline: 13 files to 5

| New file | Absorbs | Lines |
|---|---|---|
| `pipeline.py` | DatasetPipeline + MetadataWriter (metadata.py) | ~220 |
| `spatial.py` | Layout (layout.py) + Cropper (crop.py) + GridInfo/Patcher (patch.py); crop arithmetic delegates to `CropRegion.local_slices` | ~230 |
| `normalization.py` | Stats (stats.py) + StatsComputer (stats_computer.py) + Normalizer (normalizer.py) | ~465 |
| `datasets.py` | PatchDataset (dataset.py) + MultiRegionDataset (multi_region.py) + Loader (loader.py) + SpatialAugmenter (augmentation.py) | ~285 |

These groups mirror the sequential flow in pipeline.py: layout/crop/patch, then stats compute/save/apply, then dataset/augment/load. Update consumers: `training_pipeline/pipeline.py`, `inference_pipeline/loader.py` (imports 6 symbols), `inference_pipeline` stitching code (GridInfo).

### 3f. processing_pipeline: 7 files to 6

| New file | Absorbs | Lines |
|---|---|---|
| `artifacts.py` | ArtifactRegistry + MetadataManager (metadata.py, renamed ProcessingMetadata) | ~145 |
| `pipeline.py`, `tomogram.py`, `interferogram.py`, `tomogram_worker.py` | unchanged (worker isolation, see constraint 5); tomogram.py delegates subdivision to `CropRegion.subdivide_by_azimuth` | |

### 3g. benchmark_pipeline: 14 files to 7

| New file | Absorbs | Lines |
|---|---|---|
| `pipeline.py` | BenchmarkPipeline | 118 |
| `stages.py` | OverfitStage + SizeMatchStage + TrainingStage + InferenceStage + ComparisonStage, all on `ExperimentStage` base | ~350 after base extraction (from ~500) |
| `sizing.py` | SizeMatcher (size_matcher.py) + WidthScaler (width_scaler.py) | ~165 |
| `config_factory.py` | unchanged | 210 |
| `workers.py` | unchanged | 135 |
| `results.py` | TrialCollector (trial_collector.py) + ComparisonReport (comparison_report.py) | ~455 |

`gpu_queue.py` moves to `pipelines/shared/orchestration.py` (Phase 2d).

### 3h. cross_validation_pipeline: 9 files to 5

| New file | Absorbs | Lines |
|---|---|---|
| `pipeline.py` | CrossValidationPipeline | 108 |
| `stages.py` | FoldTrainingStage + FoldInferenceStage + CVReportStage, on `ExperimentStage` base | ~190 after base extraction (from ~270) |
| `folds.py` | FoldPlanner (fold_planner.py) + FoldCollector (fold_collector.py) + FoldConfigFactory (config_factory.py) + the three worker classes (workers.py) | ~215 |
| `cv_report.py` | unchanged (already reuses ComparisonReport) | 224 |

### 3i. physics_pipeline: unchanged (2 files, already minimal).

---

## Phase 4 — Models: 22 files to 18, ~500 lines removed

### 4a. `models/blocks.py` (new, ~200 lines)
Single home for the building blocks currently defined in one architecture file and imported by others:
- `ConvBlock` (UNet.py:11-47), `ResidualConvBlock` (ResUNet.py:12-68), `PixelMLP` (UNet_multihead.py:11-27), `match_spatial_size` (duplicated verbatim at UNet.py:51-59 and ResUNet.py:72-80 — keep one), `Encoder`/`Decoder` (UNet.py:63-97,101-139), `PatchEmbedding` (defined three times: TransUNet.py, UNETR.py, SwinUNet.py — unify where the implementations match; Swin keeps its own if structurally different).
- Also receive the nn-level helpers currently in `configuration/models_config.py:1-100` (`DropPath`, `build_activation`, `build_norm2d`, `build_upsample`, `initialize_weights`) — they are model code, not configuration.

### 4b. Family merges with head strategies
- `unet.py`: UNet + UNetMultihead + UNetPerGaussian + UNetSkip (renamed `ResUNetMaxpool` — it uses ResidualConvBlock and is misnamed today). The multihead/pergaussian variants duplicate ~85 percent of the base; refactor to a shared encode-bottleneck-decode body with a pluggable output-head (standard, three grouped heads, per-gaussian heads). 4 files to 1.
- `resunet.py`: ResUNet + ResUNetMultihead + ResUNetPerGaussian, same head-strategy refactor (~77 percent duplication today). 3 files to 1.
- The 13 remaining standalone architectures keep their own files (genuinely distinct), importing from blocks.py.
- `models/__init__.py` registry: keys unchanged so `models_config.py` and existing checkpoints/resolved configs continue to resolve.

---

## Phase 5 — Configuration

### 5a. Shared field groups (~100 lines saved, lower risk than full dedup)
- `seed` appears in 6 configs, `gpus`/`n_gaussians`/`batch_size` in 4 each, `height_range` in 5, `device` in 3. Extract two small dataclasses in a new `configuration/common_config.py`: `RuntimeConfig` (device, seed) and `ComputeConfig` (gpus, num_workers, batch_size), composed into the entry configs.
- Constraint 3 applies: field names and nesting visible to `ConfigCli` flags must keep their resolved paths, or a one-time alias map is added to `ConfigCli`. If aliasing is too invasive, restrict this step to NEW nesting only for configs without persisted runs, and document the duplication as intentional for the rest. This sub-phase is the only one where partial execution is acceptable.

### 5b. models_config.py decomposition (1529 lines)
- Move the ~100 lines of nn helpers to `models/blocks.py` (Phase 4a).
- Split the remainder by family into `configuration/model_configs/` (one module per family: unet, resunet, transformer, misc), with `models_config.py` reduced to a registry that re-exports everything under the same names. Imports elsewhere stay unchanged.

### 5c. Webui sync (constraint 4)
- Every config file touched: preserve blank-line block grouping; update `FIELD_TAXONOMY`/`STACK_PAIRS` in `webui/launch.js` in the same commit; verify the affected panel renders by launching the webui once.

---

## Phase 6 — main/ entry points

- `main/_bootstrap.py` (new, ~40 lines): the 6-line `repo_root` sys.path preamble (identical in all 12 scripts) plus an `EnvironmentPinner` class for the env-var pinning block currently copy-pasted in 6 scripts (single_train.py:13-23, single_infer.py:14-23, batch_inference.py:14-23, overfit_test.py:15-24, benchmark.py:13-32, cross_validate.py:13-32, including the worker variant that sets `CUDA_VISIBLE_DEVICES`). Ordering constraint: pinning must run before torch is imported, so each entry's first line stays `import _bootstrap` (or `from _bootstrap import pin`), nothing heavier.
- Logger lifecycle: add a context-manager form to `tools/logger.py` and use it in the four scripts that currently never call `close()` (batch_inference, extract_params, pre_process, compare_runs).
- Per the config contract (memory): no module-level values appear in `main/` scripts; the bootstrap module holds only mechanics, all tunables remain in config dataclasses with ConfigCli overrides.
- `main/tune.py` (403 lines) is the only oversized entry: extract its orchestrator class body into `pipelines/tuning_pipeline/pipeline.py` (where the other entries keep their logic), leaving tune.py as a thin wiring script like the rest.

---

## Phase 7 — Scripts triage (low priority)

- Fix the three stale `pipelines.param_extraction_pipeline` imports (Phase 1).
- `scripts/tomo_compare_ngauss_ref.py` (1498 lines) reimplements metric/error computation (~lines 347-480) that belongs near `param_pipeline/metrics.py`; extract only if those metrics are needed by maintained code — otherwise leave reference scripts frozen. Scripts are one-off experiments by design; do not force them into pipelines.

---

## Execution order and risk

| Phase | Risk | Reason |
|---|---|---|
| 1 dead code | minimal | verified zero importers |
| 2 shared foundation | low | additive, then atomical import switch |
| 3a-3f pipeline merges | low | verbatim moves + import updates |
| 3g-3h orchestration base | medium | behavior-preserving but restructures the stage run loop; verify with a 2-model benchmark and a 2-fold CV |
| 4 models | medium | head-strategy refactor changes forward() structure; verify state_dict key compatibility with an existing checkpoint before and after |
| 5 configuration | medium-high | resolved-config and webui coupling; partial execution acceptable (5a) |
| 6 main/ | low | mechanical |
| 7 scripts | minimal | optional |

Order: 1, 2, 3a-f, 3g-h, 6, 4, 5, 7. Each merge group is one commit ("merge X into Y, no logic change"), import updates included, so `git log --follow` keeps history trackable.

Checkpoint compatibility check for Phase 4: load an existing run checkpoint into the refactored model and compare `state_dict().keys()`; if head refactoring renames parameters, add a key-mapping shim in `Checkpoint.load` rather than invalidating old runs.

---

## File count summary

| Area | Before | After |
|---|---|---|
| pipelines/training | 19 | 8 |
| pipelines/tuning | 8 | 4 |
| pipelines/inference | 14 | 8 |
| pipelines/param | 11 | 6 |
| pipelines/dataset | 13 | 5 |
| pipelines/processing | 7 | 6 |
| pipelines/benchmark | 14 | 7 |
| pipelines/cross_validation | 9 | 5 |
| pipelines/physics | 2 | 2 |
| pipelines/shared | 0 | 5 |
| models | 22 | 18 |
| tools | 20 | 10 |
| configuration | 13 | ~18 (family split adds files but removes 1529-line monolith) |
| main | 12 | 13 (adds _bootstrap) |
| scripts | 16 | 15 |
| Total (excl. webui) | ~180 | ~130 |

Duplication removed: metadata/IO boilerplate (~200 lines), plotting base (~190), experiment orchestration (~230), model blocks and variants (~500), entry-point boilerplate (~150), dead code (~170). Total ~1400-1600 lines, with the structural duplication between benchmark and cross-validation eliminated at the source.
