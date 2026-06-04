<div align="center">

# DLR-TomoSAR

### Deep Per-Pixel Gaussian-Mixture Decomposition of Tomographic SAR Spectra

*A configuration-driven deep-learning framework that learns to decompose multi-baseline synthetic-aperture-radar tomographic reflectivity profiles into superpositions of Gaussian components, enabling scatterer separation, vertical-structure characterisation, and elevation-model refinement.*

<br>

![Python](https://img.shields.io/badge/python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-deep_learning-EE4C2C?logo=pytorch&logoColor=white)
![Architectures](https://img.shields.io/badge/architectures-10-1565C0)
![Optuna](https://img.shields.io/badge/Optuna-HPO-0E4C92)
![GDAL](https://img.shields.io/badge/GDAL-geospatial_I%2FO-5CAE58)
![Domain](https://img.shields.io/badge/domain-TomoSAR_remote_sensing-6A1B9A)

</div>

---

## Abstract

Tomographic Synthetic Aperture Radar (**TomoSAR**) extends conventional interferometry by exploiting multiple, spatially-separated radar acquisitions to resolve the vertical distribution of backscattered energy within a single ground range–azimuth resolution cell. The resulting per-pixel *reflectivity profile* along the elevation axis is frequently **multi-modal**: distinct physical scatterers — the ground surface, a vegetation canopy, dihedral structures — contribute separate, overlapping peaks. Recovering these components is the central inverse problem of TomoSAR analysis and underpins applications such as forest-height estimation, layered-scatterer separation, and the refinement of digital elevation models.

**DLR-TomoSAR** casts this decomposition as a **dense, per-pixel regression** task solved by an image-to-parameter neural network. Given a TomoSAR tomogram represented as a multi-channel image, the network predicts, at every pixel, the parameters of a $K$-component Gaussian mixture that reconstructs the elevation profile. The framework provides ten interchangeable convolutional and transformer-based segmentation backbones, a richly configurable composite training objective, an optional heteroscedastic-uncertainty head, and a complete, reproducible pipeline spanning data generation, training, inference, classical parameter fitting, and benchmarking.

---

## 1. Scientific Motivation

A TomoSAR stack focuses the complex backscatter as a function of elevation $z$, yielding for each ground pixel a one-dimensional reflectivity profile $y(z)$. Because multiple scattering mechanisms may coexist within a single resolution cell, this profile is in general a superposition of contributions that must be *separated* to be interpreted. Modelling each contribution as a Gaussian peak — characterised by its amplitude, elevation centre, and spread — provides a compact, physically-interpretable parameterisation:

- the **mean** $\mu_k$ localises a scatterer in elevation (e.g. ground level versus canopy top),
- the **amplitude** $a_k$ quantifies its relative reflectivity, and
- the **standard deviation** $\sigma_k$ describes its vertical extent (e.g. volumetric vegetation versus a sharp surface return).

Classical estimation of these parameters is performed pixel-by-pixel through iterative optimisation, which is computationally expensive over large scenes and sensitive to initialisation. DLR-TomoSAR learns this mapping directly, amortising the inverse problem across an entire image in a single forward pass while retaining the option of high-fidelity classical fitting for ground-truth generation.

---

## 2. Problem Formulation

The network maps a multi-channel tomographic image to a dense field of mixture parameters. For an input image with $C_\text{in}$ channels and spatial dimensions $H \times W$, the model outputs a tensor of shape $(B, 3K, H, W)$ (or $(B, 3K+1, H, W)$ when the heteroscedastic-noise head is enabled), where $K$ is the number of Gaussian components. At each pixel the predicted parameters are

$$
\hat{\mathbf{p}} = [\,a_1, \mu_1, \sigma_1,\; a_2, \mu_2, \sigma_2,\; \dots,\; a_K, \mu_K, \sigma_K\,].
$$

For $N$ sample points $\{x_n\}_{n=1}^{N}$ along the elevation axis, the reconstructed profile at each pixel is the Gaussian superposition

$$
\hat{y}(x_n) = \sum_{k=1}^{K} a_k \exp\!\left( -\frac{(x_n - \mu_k)^2}{2\sigma_k^2 + \epsilon} \right),
\qquad \epsilon = 10^{-8},
$$

and the model is trained by comparing $\hat{y}$ against the measured reflectivity profile $y$.

### Input Channel Representations

The complex-valued tomographic data is mapped to real channels through a configurable `Representation` scheme — magnitude only, phase only, or combined magnitude/phase and real/imaginary encodings — with per-channel normalisation strategies (percentile min–max, robust IQR, z-score, or fixed $\pi$-division) selected per physical quantity.

---

## 3. Method

### 3.1 Model Architectures

Ten image-to-parameter backbones are exposed through a single model registry and selected by key. All share a common interface (input channels derived from the chosen representation; output channels equal to $3K$, optionally $+1$ for the noise head) and configurable activation, normalisation, dropout, upsampling, and weight-initialisation strategies.

| Key | Architecture | Distinguishing characteristic |
|---|---|---|
| `unet` | U-Net | Canonical encoder–decoder with skip connections |
| `unet_multihead` | U-Net (multi-head) | Separate prediction heads for amplitude, mean, and $\sigma$ |
| `unet_pergaussian` | U-Net (per-Gaussian) | Independent branch per Gaussian component |
| `resunet` | ResU-Net | Residual blocks throughout the backbone |
| `attention_unet` | Attention U-Net | Attention gates on the skip connections |
| `unetplusplus` | U-Net++ | Nested, dense skip pathways |
| `linknet` | LinkNet | Lightweight factorised decoder |
| `swin_unet` | Swin-UNet | Shifted-window transformer encoder |
| `transunet` | TransUNet | Hybrid ViT-encoder / CNN-decoder |
| `unetr` | UNETR | Vision-transformer backbone with U-Net decoding |

Most backbones partition their parameters into encoder / bottleneck / decoder / output-head groups, each assigned an independent learning rate and weight decay.

### 3.2 Composite Training Objective

The loss is a configurable, weighted combination of complementary terms, enabling a curriculum that first stabilises global shape before refining detail:

- **Curve-space losses** on the reconstructed profile — MSE, $L_1$, Huber, Charbonnier, cosine similarity, spectral coherence, and SSIM along the elevation axis.
- **Parameter-space losses** — per-component penalties on amplitude, mean, and $\sigma$, with permutation-invariant component matching to resolve label ambiguity in multi-component decomposition.
- **Regularisation** — total-variation smoothness over the predicted parameter fields.
- **Heteroscedastic mode** — when the noise head is active, a Gaussian negative-log-likelihood with a per-pixel predicted noise standard deviation, providing calibrated uncertainty.

A complete, formal description of the objective, optimiser, schedules, and metrics is given in the **[Training Pipeline — Technical Reference](docs/training_pipeline_reference.md)**.

### 3.3 Optimisation & Training Engineering

Training is performed with **AdamW** under discriminative per-group learning rates, a **linear-warmup → cosine-annealing** schedule, automatic mixed precision, gradient accumulation, configurable gradient clipping (fixed or adaptive), an exponential moving average of the weights, and early stopping with best-state restoration. The training state is checkpointed in full — model, optimiser, scheduler, EMA shadow, early-stopping and warmup state — so that any run is exactly resumable.

---

## 4. End-to-End Pipeline

The framework is organised as a sequence of self-contained, individually-runnable stages, each driven by a dedicated entry point in `main/` and a corresponding configuration group.

```
  raw SAR stack
       │
       ▼
  [1] pre-processing      tomogram formation (beamforming) · cropping · height-range selection
       │
       ▼
  [2] dataset preparation channel representation · patch extraction · region-based split · augmentation · normalisation
       │
       ▼
  [3] training            backbone + composite loss + EMA + warmup/cosine + early stopping
       │
       ▼
  [4] inference           patch stitching · per-pixel metrics · profile/slice/animation visualisation · report
       │
       ▼
  [5] parameter fitting   GPU-batched classical Gaussian fitting (ground-truth generation)
       │
       ▼
  [6] benchmarking / HPO  multi-architecture comparison · Optuna hyperparameter search
```

| Stage | Entry point(s) | Purpose |
|---|---|---|
| Pre-processing | `main/pre_process.py` | Form tomograms from the SAR stack; crop, select height range, clip amplitude. |
| Training | `main/single_train.py`, `main/batch_train.py` | Single run, or multi-GPU parallel experiments. |
| Overfit test | `main/overfit_test.py` | Capacity check on a single repeated batch. |
| Inference | `main/single_infer.py`, `main/batch_inference.py` | Predict, stitch patches, compute metrics, render figures and reports. |
| Parameter extraction | `main/extract_params.py` | Classical GPU-accelerated Gaussian fitting for ground truth. |
| Benchmarking | `main/benchmark.py` | Compare all architectures on a shared dataset. |
| Hyperparameter search | `main/tune.py` | Optuna optimisation over learning rates and architecture parameters. |

---

## 5. Repository Structure

```
DLR-TomoSAR/
├── main/                       # stage entry points (pre-process, train, infer, extract, benchmark, tune)
├── models/                     # 10 architectures + model registry (UNet, ResUNet, Attention, Swin, TransUNet, UNETR, …)
├── configuration/              # dataclass configuration for every stage
│   ├── processing_config.py    #   tomogram formation, cropping, height range
│   ├── dataset_config.py       #   channels, patches, augmentation, splits
│   ├── norm_config.py          #   per-channel normalisation strategies
│   ├── training_config.py      #   trainer, optimiser, scheduler, warmup, EMA, loss
│   ├── inference_config.py     #   prediction, stitching, metrics, visualisation
│   ├── param_extraction_config.py  # classical Gaussian-fitting settings
│   ├── models_config.py        #   per-architecture configuration
│   └── tuning_config.py        #   hyperparameter search ranges
├── tools/                      # Gaussian mixture math, representation enum, permutation metrics,
│                               # rich logging, resource/live monitoring, region splitting, model summary
├── notebooks/                  # pipeline inspection, normalisation studies, parameter-distribution analysis
├── docs/
│   └── training_pipeline_reference.md   # formal, equation-level training specification
├── coding_style.md             # engineering conventions / developer profile
├── requirements.txt
└── README.md
```

---

## 6. Installation

```bash
pip install -r requirements.txt
```

The framework builds on the PyTorch ecosystem and the scientific-Python stack (NumPy, SciPy, scikit-image, Matplotlib, h5py), with **GDAL** for geospatial raster I/O and convex-optimisation solvers (cvxpy, cvxopt, osqp, ecos) supporting the classical fitting routines. A CUDA-capable GPU is recommended; mixed-precision training and GPU-batched parameter fitting assume CUDA availability.

---

## 7. Usage

```bash
# [1] Form tomograms and pre-process the SAR stack
python main/pre_process.py

# [2]+[3] Train a model (architecture and all hyperparameters set in configuration/)
python main/single_train.py

# [4] Run inference: stitch predictions, compute metrics, render figures and a report
python main/single_infer.py

# [5] Generate ground-truth parameters via classical GPU-accelerated fitting
python main/extract_params.py

# [6] Benchmark all architectures / search hyperparameters
python main/benchmark.py
python main/tune.py
```

All behaviour is governed by the dataclass configuration objects in `configuration/`; there are no command-line flags. Edit the relevant configuration group before launching a stage.

---

## 8. Documentation

- **[Training Pipeline — Technical Reference](docs/training_pipeline_reference.md)** — a complete, equation-level specification of the objective, optimiser and parameter groups, warmup and cosine-annealing schedules, EMA, early stopping, mixed precision, gradient accumulation, the full evaluation-metric suite, checkpointing, and the epoch loop.
- **`coding_style.md`** — the engineering conventions and design philosophy underpinning the codebase (modular object-oriented design, defensive numerical practice, exhaustive observability, total-state checkpointing).
- **`notebooks/`** — inspection notebooks documenting each pipeline stage, the input-normalisation strategy, and the empirical distribution of the target parameters.

---

## 9. Engineering Principles

The codebase is written to research-software standards emphasising reproducibility and observability: highly modular, single-responsibility object-oriented design with a central-orchestrator pattern; defensive numerical practice (clamping, $\epsilon$-stabilised denominators); total-state checkpointing for exact resumption; structured hierarchical logging with live resource monitoring; and explicit management of hardware resources and thread contention.

---

## 10. Citation

```bibtex
@software{dlr_tomosar,
  title  = {DLR-TomoSAR: Deep Per-Pixel Gaussian-Mixture Decomposition
            of Tomographic SAR Spectra},
  author = {{DLR-TomoSAR contributors}},
  year   = {2026},
  note   = {Configuration-driven deep-learning framework for TomoSAR
            reflectivity-profile decomposition}
}
```

---

## 11. Selected References

1. Reigber, A. & Moreira, A. *First Demonstration of Airborne SAR Tomography Using Multibaseline L-Band Data.* IEEE TGRS (2000).
2. Fornaro, G., Serafino, F. & Soldovieri, F. *Three-Dimensional Focusing with Multipass SAR Data.* IEEE TGRS (2003).
3. Zhu, X. X. & Bamler, R. *Tomographic SAR Inversion by $L_1$-Norm Regularization — The Compressive Sensing Approach.* IEEE TGRS (2010).
4. Ronneberger, O., Fischer, P. & Brox, T. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI (2015).
5. Hatamizadeh, A. et al. *UNETR: Transformers for 3D Medical Image Segmentation.* WACV (2022).
6. Cao, H. et al. *Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation.* ECCV Workshops (2022).

---

<div align="center">
<sub>Configuration-driven · ten architectures · classical &amp; learned inversion · fully reproducible</sub>
</div>
