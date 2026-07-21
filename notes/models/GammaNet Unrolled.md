---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - GammaNet
  - Unrolled Physics Network
family: unrolled
registry_key: gamma_net
summary: Unrolled LISTA-style proximal-gradient inversion of the tomographic forward model, with learned steps, thresholds, and a per-pixel 1D prox; its own isolated model family and training pipeline.
---

# GammaNet Unrolled

`GammaNet` (`models/unrolled/gamma_net.py`) is an unrolled physics network in the spirit of γ-Net (Qian et al., *γ-Net: Superresolving SAR Tomographic Inversion via Deep Learning*, IEEE TGRS 2022) and LISTA/ISTA-Net: the compressive-sensing TomoSAR inversion is unrolled into $L$ proximal-gradient iterations whose step sizes, thresholds, and proximal operators are learned. Unlike every model in the [[Model Zoo]], the tomographic forward model is encoded **exactly** — the steering operator is built per pixel from the kz field — so the network only learns the prior, not the physics. It is deliberately isolated: its own model family (`UNROLLED_MODEL_REGISTRY`), its own training pipeline (`pipelines/unrolled/`), and its own entry point (`main/training/train_unrolled.py`); nothing in the backbone, autoencoder, JEPA, or benchmark stacks depends on it.

---

## Summary

Per pixel $p$, the forward model maps the reflectivity profile $\mathbf{s}_p \in \mathbb{R}_{\ge 0}^N$ on the elevation grid to complex coherence measurements over $T$ tracks through the steering operator $\mathbf{A}_p$ with entries $e^{i k_{z,t}(p) z_n} \, dz$ — the same operator the physics losses apply per track (`PhysicalLoss.synthesise_track`); `TomoOperator` builds it once per forward pass as a per-pixel steering tensor and applies forward and adjoint as batched matrix products reused across all iterations. GammaNet inverts it by unrolling:

- matched-filter initialisation $\mathbf{s}^0 = \max(0, \mathbf{A}^H \mathbf{y} / T)$,
- $L$ iterations of gradient step → learned per-pixel 1D convolutional prox along $z$ → nonnegative soft-threshold,
- learned per-iteration step sizes $\alpha_l$ and thresholds $\theta_l$ (softplus-reparameterised to stay positive).

The gradient is normalised by $T N \, dz^2$. Note the code applies the adjoint **without** the $dz$ weight (the $dz$-weighted-inner-product adjoint the operator test locks), so the iteration's gradient operator is $\tfrac{1}{dz}\mathrm{Re}[\mathbf{A}^H\mathbf{A}]$ with norm bound $T N \, dz$ — the normaliser therefore leaves a residual factor of at most $1/dz$ rather than being a strict Lipschitz bound. On the project geometry (dz ≈ 0.4–0.6 m, five-track subset including the $k_z = 0$ primary) the measured normalised operator norm is 0.34–0.53, so `step_init = 1.0` starts stable with 2–3× margin, and the per-iteration steps are learned anyway. The network is track-count- and grid-length-agnostic (the operator is built from the batch's kz map and x-axis at run time) and tiny — about 1.4k parameters at defaults — because the physics carries the inductive bias.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{s}^l$ | Profile estimate at iteration $l$ (length $N$, per pixel) |
| $\mathbf{y}$ | Complex coherence measurements over $T$ tracks (per pixel) |
| $\mathbf{A}$ | Per-pixel steering operator, $A_{tn} = e^{i k_{z,t} z_n}\, dz$ |
| $\alpha_l, \theta_l$ | Learned step size and soft-threshold of iteration $l$ |
| $\mathcal{P}_l$ | Learned prox of iteration $l$: residual half-resolution conv bottleneck along $z$ |
| $L_{\text{lip}}$ | Gradient normaliser $T N \, dz^2$ (see Summary: bounds the true-adjoint operator; the implemented $dz$-less adjoint retains a $\le 1/dz$ residual factor) |
| $K_z$ | Per-pixel vertical wavenumber map (from the [[TomoSAR track acquisition parameters|geometry field]]) |

---

## Architecture

$$
\begin{aligned}
\mathbf{s}^0 &= \max\!\left(0, \tfrac{1}{T}\,\mathrm{Re}\!\left[\mathbf{A}^H \mathbf{y}\right]\right) \\
\mathbf{r}^l &= \mathbf{s}^l + \alpha_l \, \frac{\mathrm{Re}\!\left[\mathbf{A}^H (\mathbf{y} - \mathbf{A}\mathbf{s}^l)\right]}{L_{\text{lip}}} \\
\mathbf{s}^{l+1} &= \max\!\left(0, \, \mathcal{P}_l(\mathbf{r}^l) - \theta_l\right)
\end{aligned}
$$

The nonnegative soft-threshold replaces the signed soft-threshold of ISTA because reflectivity power is nonnegative; the prox $\mathcal{P}_l$ is the learned prior over profile shape, applied identically at every pixel: a residual half-resolution bottleneck (stride-2 windows projected by a pointwise conv, i.e. a GEMM, → act → Conv1d at half resolution → linear upsample back). The window-GEMM form exists because cuDNN's deterministic algorithms make the equivalent strided Conv1d ~1.7× slower, and the half-resolution hidden field halves the dominant activation traffic; a full-stride ConvTranspose1d variant was rejected for failing the 60-step overfit contract (peak error 32 m) — the physics-native counterpart of the [[PixelMLP]] no-spatial-context philosophy.

---

## Training Pipeline

`UnrolledTrainingPipeline` (`pipelines/unrolled/training/pipeline.py`) reuses the backbone dataset stack (`DatasetPipeline` with `build_geometry_field=True`) and trains on **synthesised measurements**: ground-truth Gaussian parameters are denormalised, rendered to profiles on the elevation grid (`GaussianCurve.reconstruct`), power-normalised, and pushed through the exact forward operator to produce the per-pixel coherence vector; optional complex Gaussian noise (`measurement_noise_std`) breaks the inverse-crime purity. The loss is a power-masked L1/MSE over the normalised profile, with peak-position MAE (metres) monitored. The dataset's input channel stack is untouched — only ground-truth parameters and the kz field are consumed.

This phase-1 setup validates the unrolled inversion machinery end to end on fit-consistent measurements. The phase-2 step — feeding **measured** covariance estimated from the SLC stack instead of synthesised coherence — requires the preprocessing pipeline to persist per-pixel multilooked covariance, and is left as future work.

Entry point: `main/training/train_unrolled.py`, configured by `UnrolledEntryConfig` (`configuration/training/unrolled.py`) via `ConfigCli`. Following the exposure convention, the training knobs live in a slim `UnrolledTrainingConfig` containing exactly the fields the trainer reads. The trainer reuses the shared runtime components: `Warmup` + `Scheduler` (linear LR warmup into cosine annealing), `WeightEma` (validation, the best checkpoint, and the final test run all use the averaged weights when `use_ema` is on), and `VramReservation` (fill before the first epoch, refill after each validation). AMP is intentionally unsupported — the forward operator is complex-valued. Run layout: `logdir/<run>/{docs,meta,checkpoints,logs,training_summary.json}` with the model architecture persisted through `UnrolledModelConfigIO`. The script is registered in the webui: a Train-group variant ("Unrolled") in `ScriptCatalog`, an `ENTRY_OVERRIDES` mapping in `project_paths.py`, and a `launch_layout` page claiming every entry-config field; it is deliberately absent from `TensorboardManager.TRAINING_LOGDIRS` because the trainer writes no TensorBoard event files.

---

## Design Rationale

**The physics-native comparison point.** The [[Model Zoo]] asks which generic vision prior suits the task; GammaNet asks the complementary question — how far does the exact forward model plus a minimal learned prior go? If it matches capacity-matched backbones on profile recovery, the vision networks are mostly re-learning known physics; if it trails, the data-driven priors capture structure (spatial context, GT-fit idiosyncrasies) the forward model alone cannot.

**Isolation as a requirement.** The family deliberately shares only leaf utilities with the rest of the codebase (`PhysicalLoss` operators, `GaussianCurve`, the dataset pipeline, `ConfigCli`, `Logger`), so it can evolve — or be deleted — without touching the backbone, autoencoder, or benchmark stacks. It participates in the state_dict baseline as its own `unrolled` family.

---

## Parameter Reference

See [[Configuration Layer]] → `GammaNetConfig` (`n_iterations`, `prox_hidden`, `prox_kernel_size`, `step_init`, `threshold_init`) and `UnrolledEntryConfig` (`curve_loss`, `measurement_noise_std`, `power_floor`).

---

## Provenance

Formulation adapted from γ-Net (Qian, Zhu et al., IEEE TGRS 2022, unrolled CS TomoSAR inversion), LISTA (Gregor & LeCun, ICML 2010, learned step/threshold unrolling), and ISTA-Net (Zhang & Ghanem, CVPR 2018, learned proximal operators).

Deviations from γ-Net proper, all deliberate: γ-Net solves for the **complex single-look reflectivity** $\boldsymbol{\gamma}$ from stacked SLC measurements with one fixed steering matrix $\mathbf{R}$, which lets it *learn* per-layer weight matrices under the coupling structure $\mathbf{W}_2 = \mathbf{I} - \mathbf{W}_1\mathbf{R}$; here the operator varies per pixel (per-pixel $k_z$), so learned matrices are infeasible and the exact operator with learned scalar steps is used instead. γ-Net's two shrinkage accelerations — **support selection** (top-$\rho\%$ magnitude entries bypass thresholding each layer) and the **piecewise-linear thresholding function** (its replacement for soft-thresholding, credited with ~10 dB NMSE and faster convergence) — are not carried over; the residual CNN prox before the nonnegative threshold is this implementation's learned-shrinkage counterpart. γ-Net's downstream BIC model-order selection is likewise out of scope (peak/parameter extraction is handled by this project's own machinery). γ-Net converged at ~12 layers (Fig. 6 of the paper); the default here is `n_iterations = 8` with 12/16 reachable through `tunable_arch_params`. Behavioural contract verified by `tests/models_unrolled/test_gamma_net.py`: forward/adjoint operator identity ($\langle \mathbf{A}\mathbf{s}, \mathbf{y}\rangle = \langle \mathbf{s}, \mathbf{A}^H\mathbf{y}\rangle$), matched-filter peak localisation, output nonnegativity, gradient flow to both parameter groups; a 60-step synthetic overfit drives peak error below 0.5 m.

---

## Related Notes

- [[Model Zoo]] — The generic-backbone counterpart this family is compared against
- [[PixelMLP]] — The learned no-context control; GammaNet is its physics-encoded sibling
- [[TomoSAR track acquisition parameters]] — Source of the per-pixel kz geometry field
- [[Configuration Layer]] — GammaNetConfig, UnrolledEntryConfig
- [[DLR-TomoSAR Index]] — Master index
