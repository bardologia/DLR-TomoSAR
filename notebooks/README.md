# Pipeline confirmation notebooks

Visual confirmation suites for the inner workings of every pipeline. Each notebook isolates one mechanism, drives it with small seeded synthetic inputs built in-notebook, and plots inputs against outputs so correctness is verifiable by eye.

## Conventions

- One directory per pipeline, notebooks numbered in reading order.
- Cell 1 states what the notebook confirms and which modules it exercises; cell 2 bootstraps the repository root onto `sys.path`, seeds numpy and torch, and sets matplotlib defaults. A closing markdown cell states the expected visual outcome.
- Fully self-contained: no dependence on the `/ste/rnd` data mount, trained checkpoints, or a GPU. Run from each notebook's own directory with the Dune environment kernel.
- Notebooks are authored against the real APIs (verified by smoke imports and live calls during authoring) but ship unexecuted; run them to produce the figures.

## Environment notes

- `h5py` was installed into the Dune environment (3.16.0, an optional `processing` extra) so `pipelines.processing_pipeline.tomogram` imports; processing notebooks 08-10 need it.
- `jax` is not installed, so `pipelines.param_pipeline.sigma` is unimportable; param notebooks 02-06 reproduce its algorithms (peak initialisation, sigma Adam scan, best-K scoring) faithfully in NumPy and state so in their introductions.
- Steps that are inseparable from disk IO or PyRat (SLC loading, `tomo.fusartomo`, `RunLoader`, `BenchmarkPipeline.run`, full `DatasetPipeline` orchestration) are demonstrated through their mathematical core or constituent components instead; each such notebook documents the substitution.

## Index

### dataset_pipeline
01 channel representations, 02 patch grid geometry, 03 patch extraction padding, 04 input tensor assembly, 05 output parameter selection, 06 normalization strategies, 07 stats computer grouping, 08 normalize-denormalize roundtrip, 09 spatial augmentation, 10 region splitting and `__getitem__` routing.

### processing_pipeline
01 synthetic SLC stack, 02 interferogram formation, 03 DEM phase deramping, 04 amplitude clipping, 05 tomographic focusing and beamforming, 06 Capon vs Bartlett, 07 boxcar covariance filter, 08 crop subdivision, 09 subsection concatenation, 10 artifact registry and metadata.

### training_pipeline
01 curve reconstruction, 02 pointwise loss terms, 03 structural curve terms, 04 parameter loss matching, 05 physics moment terms, 06 physics signal terms, 07 composite loss landscape, 08 warmup and schedulers, 09 early stopping, EMA, gradient clipper, 10 trainer single-batch overfit, 11 control and curriculum.

### inference_pipeline
01 pipeline overview, 02 predictor forward pass, 03 Gaussian reconstruction, 04 cube stitcher tiling, 05 CPU worker matching, 06 curve metrics, 07 Gaussian parameter metrics, 08 profile and slice plots, 09 parameter and metric figures, 10 report assembly.

### param_pipeline
01 Gaussian mixture forward model, 02 peak initialisation, 03 sigma-only Adam fit, 04 sigma vs SNR, 05 best-K selection, 06 K sensitivity, 07 R-squared metric, 08 activity and separation metrics, 09 example fit plots, 10 spatial distribution plots.

### physics_pipeline
01 signal model and synthetic profiles, 02 tomographic geometry and steering, 03 covariance construction, 04 Capon and beamforming spectra, 05 moment computations, 06 total power term, 07 coherence resynthesis term, 08 covariance matching term, 09 capon cycle term.

### tuning_pipeline
01 search space definition, 02 parameter sampling, 03 config override application, 04 objective evaluation path, 05 sampler convergence, 06 trial pruning, 07 result aggregation and best trial, 08 two-phase flow, 09 trial distribution across workers.

### cross_validation_pipeline
01 azimuth partition, 02 fold assignment and leakage, 03 spatial fold maps, 04 disjoint training regions, 05 deterministic assignment, 06 per-fold config overrides, 07 fold balance statistics, 08 metric aggregation, 09 report assembly.

### benchmark_pipeline
01 architecture registry, 02 backbone instantiation, 03 shared dataset contract, 04 width scaling, 05 size matching, 06 metric collection (mock), 07 ranking and tables, 08 timing and throughput, 09 GPU queue scheduling.
