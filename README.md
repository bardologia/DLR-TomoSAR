<div align="center">

# DLR-TomoSAR

### Deep Per-Pixel Gaussian-Mixture Decomposition of Tomographic SAR Spectra

*A configuration-driven, physics-informed deep-learning framework that decomposes multi-baseline SAR tomographic reflectivity profiles into Gaussian scatterer components — from raw SLC stacks to trained models, benchmarks, and validated physics, with every stage runnable from a single web console.*

<br>

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![Architectures](https://img.shields.io/badge/architectures-21-1565C0)
![Loss terms](https://img.shields.io/badge/loss_terms-15-8E24AA)
![Tests](https://img.shields.io/badge/tests-1022-2E7D32)
![Optuna](https://img.shields.io/badge/Optuna-HPO-0E4C92)
![Domain](https://img.shields.io/badge/domain-TomoSAR_remote_sensing-6A1B9A)

</div>

---

## Overview

Tomographic Synthetic Aperture Radar (**TomoSAR**) extends conventional interferometry by exploiting multiple, spatially separated radar acquisitions to resolve the vertical distribution of backscattered energy within each ground range–azimuth resolution cell. The resulting per-pixel *reflectivity profile* along elevation is frequently **multi-modal**: distinct physical scatterers — the ground surface, a vegetation canopy, dihedral structures — contribute separate, overlapping peaks. Recovering these components is the central inverse problem of TomoSAR analysis and underpins forest-height estimation, layered-scatterer separation, and elevation-model refinement.

**DLR-TomoSAR** casts this decomposition as a **dense, per-pixel regression** solved by an image-to-parameter neural network: given a tomogram represented as a multi-channel image, the network predicts at every pixel the parameters of a $K$-component Gaussian mixture that reconstructs the elevation profile. Where classical estimation fits each pixel by iterative optimisation, the network amortises the inverse problem across an entire scene in a single forward pass — while the framework retains GPU-batched classical fitting for ground-truth generation and a dedicated physics pipeline that validates the learned decomposition against Capon spectral estimates, covariance structure, and profile moments.

The framework is **end-to-end and self-contained**: raw SLC stack processing, dataset construction, training, inference, classical parameter extraction, physics validation, architecture benchmarking, cross-validation, and hyperparameter search are each a self-sufficient pipeline with its own entry point and dataclass configuration — all orchestrated, monitored, and launched from a built-in web control console.

---

## Highlights

| | |
|---|---|
| **21 architectures** | CNN, residual, attention, dense, pyramid, and transformer backbones behind one registry and a uniform configuration interface |
| **15-term composite objective** | Curve-space, parameter-space, and physics-informed loss terms, freely weighted and combinable, with a two-phase loss curriculum |
| **Physics-informed training** | Moment matching, coherence re-synthesis, covariance matching, Capon cycle-consistency, and total-power conservation derived from the tomographic signal model |
| **Full SAR processing chain** | SLC stack to interferograms (DEM deramping) to Capon/Bartlett beamforming to tomographic cubes |
| **Classical baseline** | GPU-batched Gaussian fitting (peak initialisation, $\sigma$-only refinement via SciPy or JAX, best-$K$ selection by $R^2$) for ground-truth generation |
| **Rigorous evaluation** | Parameter-count-matched architecture benchmarking, azimuth-stratified $k$-fold cross-validation, two-phase Optuna hyperparameter search |
| **Web control console** | Standard-library HTTP server with live SSE job streaming, typed configuration editing, KaTeX-rendered theory, run browsing, and resource monitoring |
| **1022 tests** | Full pytest suite including a state-dict regression baseline that pins parameter keys and shapes of all 21 models, guaranteeing checkpoint compatibility |

---

## 1. Problem Formulation

A TomoSAR stack focuses the complex backscatter as a function of elevation $z$, yielding for each ground pixel a one-dimensional reflectivity profile $y(z)$. Modelling each scattering contribution as a Gaussian peak provides a compact, physically interpretable parameterisation: the **mean** $\mu_k$ localises a scatterer in elevation, the **amplitude** $a_k$ quantifies its relative reflectivity, and the **standard deviation** $\sigma_k$ describes its vertical extent (volumetric vegetation versus a sharp surface return).

For an input image with $C_\text{in}$ channels and spatial dimensions $H \times W$, the model outputs a tensor of shape $(B, 3K, H, W)$, giving at each pixel

$$
\hat{\mathbf{p}} = [\,a_1, \mu_1, \sigma_1,\; a_2, \mu_2, \sigma_2,\; \dots,\; a_K, \mu_K, \sigma_K\,].
$$

For $N$ sample points $\{x_n\}_{n=1}^{N}$ along the elevation axis, the reconstructed profile is the Gaussian superposition

$$
\hat{y}(x_n) = \sum_{k=1}^{K} a_k \exp\!\left( -\frac{(x_n - \mu_k)^2}{2\sigma_k^2 + \epsilon} \right),
$$

and the model is trained by comparing $\hat{y}$ against the measured reflectivity profile $y$ — and, through the physics terms, against quantities derived directly from the complex SAR data.

The complex tomographic input is mapped to real channels through a configurable `Representation` scheme (magnitude, phase, or combined encodings) with per-channel normalisation strategies selected per physical quantity.

---

## 2. Model Zoo — 21 Architectures

All backbones share a common interface (input channels derived from the chosen representation, output channels equal to $3K$) and are exposed through a single model registry with per-architecture dataclass configuration: features, activation, normalisation, upsampling mode, and discriminative learning rates and weight decays for the encoder, bottleneck, decoder, and output-head parameter groups.

| Family | Registry keys |
|---|---|
| U-Net | `unet` · `unet_multihead` · `unet_pergaussian` · `unet_skip` |
| Residual U-Net | `resunet` · `resunet_multihead` · `resunet_pergaussian` |
| Attention / nested skips | `attention_unet` · `unetplusplus` · `u2net` |
| Transformer hybrids | `swin_unet` · `transunet` · `unetr` · `segformer` |
| Modern convolutional | `convnext_unet` · `deeplabv3plus` · `fpn` |
| Dense / multi-resolution | `dense_unet` · `multires_unet` · `hrnet` |
| Lightweight | `linknet` |

The *multi-head* variants predict amplitude, mean, and $\sigma$ through separate heads; the *per-Gaussian* variants devote an independent branch to each mixture component. Per-architecture design notes live in `notes/models/`.

---

## 3. Composite Training Objective

The loss is a weighted combination of fifteen complementary terms, each individually switchable and normalised by empirically calibrated factors so that weights are comparable across terms:

**Curve-space** — agreement of the reconstructed profile with the measured spectrum:
MSE · $L_1$ · Huber · Charbonnier · cosine similarity · spectral coherence · SSIM along the elevation axis.

**Parameter-space** — direct supervision of the mixture parameters against classically fitted ground truth, with permutation-invariant component matching (Hungarian assignment) to resolve label ambiguity:
per-component $L_1$ · per-component Huber.

**Physics-informed** — consistency with quantities derived from the tomographic signal model and acquisition geometry (wavelength, baselines, $k_z$):
total-power conservation against the Capon spectrum · first- and second-moment matching (centroid and spread) · steering-based coherence re-synthesis · covariance matching · Capon cycle-consistency.

**Regularisation** — total-variation smoothness over the predicted parameter fields.

A **loss curriculum** swaps the active term set at a configured epoch — typically a stabilising warm-up objective followed by the complete composite — optionally resetting the optimiser, schedule, and early-stopping state at the transition. The training entry point can fan the curriculum out into parallel trials for direct comparison.

### Training engineering

Training uses **AdamW** with per-group discriminative learning rates, a **linear-warmup to cosine-annealing** schedule, automatic mixed precision (bfloat16), gradient accumulation, fixed or adaptive gradient clipping with NaN/Inf detection, an **exponential moving average** of the weights, and early stopping with best-state restoration. Checkpoints capture the *complete* training state — model, optimiser, scheduler, EMA shadow, early-stopping and warm-up state — so any run is exactly resumable.

---

## 4. End-to-End Pipeline

Nine entry points in `main/` drive eight self-contained pipelines, each governed by its own configuration group:

```
  raw SLC stack
       |
       v
  [1] pre_process       interferogram formation, DEM deramping, Capon/Bartlett
       |                beamforming, cropping, height-range selection -> tomographic cubes
       v
  [2] extract_params    GPU-batched classical Gaussian fitting: peak initialisation,
       |                sigma refinement (SciPy/JAX), best-K selection by R^2 -> ground truth
       v
  [3] train             backbone + 15-term composite loss + curriculum + EMA
       |                + warmup/cosine + early stopping (optionally fanned into trials)
       v
  [4] infer             prediction, patch stitching, per-pixel metrics,
       |                profile/slice/error visualisation, report generation
       v
  [5] physics_check     learned decomposition vs Capon spectra: moments,
       |                coherence, covariance, cycle-consistency
       v
  [6] evaluate at scale benchmark (21 architectures, size-matched, multi-GPU)
                        cross_validate (azimuth-stratified k-fold)
                        tune (two-phase Optuna search) - compare_runs
```

| Entry point | Pipeline | Purpose |
|---|---|---|
| `main/pre_process.py` | `processing_pipeline` | SLC stack to interferograms to beamformed tomographic cubes |
| `main/extract_params.py` | `param_pipeline` | Classical GPU-batched Gaussian fitting for ground-truth generation |
| `main/train.py` | `training_pipeline` | Single run or curriculum-trial fan-out |
| `main/infer.py` | `inference_pipeline` | Prediction, stitching, metrics, figures, and reports over one or more runs |
| `main/physics_check.py` | `physics_pipeline` | Agreement of fitted quantities with the tomographic signal model |
| `main/benchmark.py` | `benchmark_pipeline` | All-architecture comparison with parameter-count matching and multi-GPU worker dispatch |
| `main/cross_validate.py` | `cross_validation_pipeline` | $k$-fold cross-validation with azimuth-based, leakage-free fold assignment |
| `main/tune.py` | `tuning_pipeline` | Two-phase Optuna search: learning rates / weight decay, then architecture parameters |
| `main/compare_runs.py` | shared | Comparative summary tables across completed runs |

Dataset construction (channel representation, patch extraction, region-based splitting, normalisation) is handled by the shared `dataset_pipeline`; cross-pipeline orchestration, run I/O, metadata, and plotting live in `pipelines/shared/`.

---

## 5. Web Control Console

The repository ships a zero-dependency web console (`webui/`) that turns the entire framework into an interactive control surface:

```bash
webui/run.sh            # serves on http://localhost:8765
```

- **Launch** any of the nine entry points with typed, validated configuration overrides and a live command preview; jobs stream their stdout to the browser in real time over server-sent events, with stop control.
- **Configuration** pages are generated live by AST-parsing the dataclasses in `configuration/` — every field, type, and default, always in sync with the code.
- **Theory** pages render the signal model, mixture target, loss terms, and optimiser as KaTeX equations; **architecture** pages document all 21 backbones with selection guidance; **pipeline** pages map the staged flow to launchable scripts.
- **Run browser** and **cube explorer** inspect completed runs and tomographic volumes; integrated **TensorBoard** lifecycle management and a **resource watchdog** track GPU, CPU, and memory throughout.

The backend is standard-library only (`http.server` + SSE) — no additional packages to install. See `webui/README.md` for the full architecture and API reference.

---

## 6. Configuration System

All behaviour is governed by dataclass configuration groups in `configuration/` — one module per stage, with defaults embedded in the dataclasses. Every entry point wraps its config in `ConfigCli`, which exposes each leaf field as a dotted-path command-line override:

```bash
python main/train.py --help-config                      # list every field, type, and default
python main/train.py --loss.use_moments true --trainer.use_amp true
python main/train.py --detach                           # run detached (nohup-style)
```

Defaults live in the dataclasses, single-run variations are expressed as CLI overrides, and batch experiments apply programmatic override functions — the same mechanism used by the benchmark, cross-validation, and tuning pipelines and by the web console's launch page.

---

## 7. Repository Structure

```
DLR-TomoSAR/
├── main/             9 stage entry points (pre_process, extract_params, train, infer,
│                     physics_check, benchmark, cross_validate, tune, compare_runs)
├── pipelines/        8 self-contained pipelines + shared orchestration, I/O, plotting
├── models/           21 registered architectures, shared blocks, model registry
├── configuration/    dataclass configuration groups for every stage
├── tools/            shared utilities: ConfigCli, Logger, MetricTracker, ResourceMonitor,
│                     Gaussian mixture ops, tomographic geometry, region splitting,
│                     permutation metrics, markdown reporting
├── webui/            web control console (stdlib HTTP + SSE, KaTeX, live job streaming)
├── scripts/          state-dict baseline generation, parameter sweeps, SAR simulation,
│                     tomogram rendering and comparison utilities
├── tests/            1022 tests across 16 suites, incl. state-dict regression baseline
├── notebooks/        87 confirmation notebooks across 9 per-pipeline suites
├── notes/            per-architecture design notes and refactoring documentation
├── presentations/    generated presentation suites and the users' guide (LaTeX/PDF)
├── pyproject.toml    installable package, pinned dependencies, pytest configuration
└── requirements.txt  pinned mirror of pyproject.toml
```

---

## 8. Installation

```bash
pip install -e .                    # core framework
pip install -e ".[processing]"     # + h5py for SLC/HDF5 data loading
pip install -e ".[sigma]"          # + JAX-accelerated sigma optimisation
pip install -e ".[dev]"            # + pytest, pyflakes
```

Requires **Python 3.11+** and builds on the PyTorch 2.11 ecosystem with NumPy, SciPy, scikit-image, Matplotlib, Optuna, TensorBoard, and Rich. A CUDA-capable GPU is recommended: mixed-precision training, multi-GPU benchmarking, and GPU-batched parameter fitting assume CUDA availability.

---

## 9. Usage

```bash
# [1] Form tomographic cubes from the SLC stack
python main/pre_process.py

# [2] Generate ground-truth parameters via classical GPU-batched fitting
python main/extract_params.py

# [3] Train (architecture and all hyperparameters via configuration/ or overrides)
python main/train.py --loss.use_huber_curve true --loss.use_moments true

# [4] Inference: stitch predictions, compute metrics, render figures and a report
python main/infer.py

# [5] Validate the learned decomposition against the physics
python main/physics_check.py

# [6] Evaluate at scale
python main/benchmark.py        # size-matched comparison of all 21 architectures
python main/cross_validate.py   # azimuth-stratified k-fold
python main/tune.py             # two-phase Optuna search
```

Or launch and monitor everything from the web console: `webui/run.sh`.

---

## 10. Verification and Reproducibility

- **Test suite** — 1022 tests across 16 suites cover every pipeline, all configuration dataclasses, the loss terms, callbacks, orchestration, and shared numerical utilities. Run with `pytest`.
- **State-dict regression baseline** — `tests/state_dict_baseline.json` pins the parameter keys and shapes of all 21 registered models; any refactor must leave this test green, guaranteeing that existing checkpoints continue to load. The baseline is regenerated only for intentional architecture changes via `scripts/generate_state_dict_baseline.py`.
- **Confirmation notebooks** — 87 self-contained notebooks across 9 per-pipeline suites visually verify each stage on seeded synthetic inputs, from channel representations and beamforming through loss behaviour, fold assignment, and benchmark scheduling.
- **Total-state checkpointing** — every training run is exactly resumable; configuration snapshots and pipeline metadata are persisted alongside every run.
- **Pinned environment** — dependencies are pinned to exact versions in `pyproject.toml` and mirrored in `requirements.txt`.

---

## 11. Selected References

1. Reigber, A. & Moreira, A. *First Demonstration of Airborne SAR Tomography Using Multibaseline L-Band Data.* IEEE TGRS (2000).
2. Fornaro, G., Serafino, F. & Soldovieri, F. *Three-Dimensional Focusing with Multipass SAR Data.* IEEE TGRS (2003).
3. Lombardini, F. & Reigber, A. *Adaptive Spectral Estimation for Multibaseline SAR Tomography with Airborne L-Band Data.* IGARSS (2003).
4. Zhu, X. X. & Bamler, R. *Tomographic SAR Inversion by L1-Norm Regularization — The Compressive Sensing Approach.* IEEE TGRS (2010).
5. Ronneberger, O., Fischer, P. & Brox, T. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI (2015).
6. Cao, H. et al. *Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation.* ECCV Workshops (2022).
7. Hatamizadeh, A. et al. *UNETR: Transformers for 3D Medical Image Segmentation.* WACV (2022).

---

## 12. Citation

```bibtex
@software{dlr_tomosar,
  title  = {DLR-TomoSAR: Deep Per-Pixel Gaussian-Mixture Decomposition
            of Tomographic SAR Spectra},
  author = {{DLR-TomoSAR contributors}},
  year   = {2026},
  note   = {Configuration-driven, physics-informed deep-learning framework
            for TomoSAR reflectivity-profile decomposition}
}
```

---

<div align="center">
<sub>21 architectures · 15-term physics-informed objective · classical &amp; learned inversion · 1022 tests · one console</sub>
</div>
