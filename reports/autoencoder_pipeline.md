# Autoencoder Pipeline — Detailed Report

> Source: [pipelines/autoencoder_pipeline/](../pipelines/autoencoder_pipeline)

## 1. Purpose

A **self-supervised** pipeline that learns a low-dimensional latent representation of individual elevation profiles extracted from the dataset patches. The encoder is regularised with a VICReg-style loss combining reconstruction, variance, covariance, and an optional NT-Xent contrastive term. The output of the pipeline is a trained autoencoder, embeddings of every profile in the val / test splits, plus PCA / UMAP visualisations and a Markdown run report.

The pipeline is implemented in [pipelines/autoencoder_pipeline/pipeline.py](../pipelines/autoencoder_pipeline/pipeline.py) and is configured by [AutoencoderConfig](../pipelines/autoencoder_pipeline/config.py).

## 2. Top-level flow

[AutoencoderPipeline.run()](../pipelines/autoencoder_pipeline/pipeline.py#L122):

1. Build the patch-level dataset via the **dataset creation pipeline** (same as training).
2. Wrap each patch dataset into a `ProfileDataset` that flattens patches into individual elevation profiles.
3. Build three `DataLoader`s with a `LoaderBuilder` that supports `persistent_workers` and `prefetch_factor=4`.
4. Save `docs/ae_config.json` and `meta/run_summary.json`.
5. Instantiate `Autoencoder(ae_config)` and `Trainer(model, …, plotter, writer)`.
6. `trainer.fit()` → returns the loss history dict.
7. Run [Inference.run()](../pipelines/autoencoder_pipeline/inference.py) on val and test loaders, producing per-split HDF5 dumps + galleries + projection plots.
8. Build a Markdown report through [Reporter.write()](../pipelines/autoencoder_pipeline/reporter.py).
9. Return `{history, inference_results, checkpoint_paths, report_path}`.

The run directory layout is:

```
<run>/
  tensorboard/  docs/  logs/  images/  meta/
  embeddings/      # PCA/UMAP NPZ + plots
  reconstructions/ # per-split galleries
  checkpoints/     # encoder/decoder/ae × best/final
  report.md
```

## 3. Configuration surface

[AutoencoderConfig](../pipelines/autoencoder_pipeline/config.py) is the root dataclass with `latent_dim`, `profile_length`, plus six sub-configs:

### 3.1 `EncoderConfig` / `DecoderConfig`

`backbone : BackboneType` selects between:

- `conv1d` — strided 1-D convolutions over the profile axis, with a final linear bottleneck to `latent_dim`. The `channels` list defines the conv channel widths; `kernel_size`, `stride`, `activation` (`gelu`/`relu`/`silu`), `normalization` (`batch`/`layer`/`group`/`none`), and `dropout` are uniform across blocks.
- `mlp` — a fully-connected stack with widths from `mlp_hidden`.

The encoder additionally optionally builds a **projection head** (`use_projection_head`) — a small MLP with hidden widths `proj_hidden` ending at `proj_dim`. The projection is what is used for the variance / covariance / contrastive losses; the raw `latent` is used elsewhere. The decoder mirrors the encoder and ends with an optional `output_activation`.

### 3.2 `LossConfig`

Weights and toggles for the four VICReg-style terms:

| Term            | Toggle              | Weight                     | Notes                                                 |
|-----------------|---------------------|----------------------------|--------------------------------------------------------|
| Reconstruction  | `use_reconstruction`| `reconstruction_weight=1.0`| One of `mse`, `l1`, `smooth_l1`, `charbonnier(eps)`.   |
| Variance        | `use_variance`      | `variance_weight=1.0`      | Hinge target std `variance_target_std=1.0`.            |
| Covariance      | `use_covariance`    | `covariance_weight=0.04`   | Off-diagonal squared sum / latent dim.                 |
| Contrastive     | `use_contrastive`   | `contrastive_weight=0.5`   | NT-Xent on `(z_a, z_b)` with `temperature=0.1`.        |

`contrastive_view : ContrastiveView` chooses how the second view is produced (`augmentation`, `neighbor`, `both`).

### 3.3 `AugmentationConfig`

Per-profile stochastic augmentation: `jitter_std` (additive Gaussian noise), `scale_range` (multiplicative), `shift_max` (integer shift), `mask_prob` × `mask_max_width` (random zero-mask), `seed`.

### 3.4 `DataConfig`

- `profile_length` — clipped/padded length of every profile.
- `normalize` — `none` / `per_profile_max` / `per_profile_zscore` / `global`.
- `log_compress` + `log_eps` — apply `log(|x|+ε)` before normalization.
- `drop_zero_profiles` — discard all-zero profiles.
- `contrastive_view`, `augmentation` — passed to `Augmenter`.
- `max_profiles`, `sampling_seed` — optional subsampling cap.

### 3.5 `TrainerConfig`

`epochs`, `learning_rate`, `weight_decay`, `optimizer` (`adamw`), `scheduler` (`cosine` etc.), `warmup_steps`, `grad_clip`, `use_amp`, `save_every`, `val_every`, `early_stop_patience`, `device`, `log_every_n_steps`.

### 3.6 `IOConfig`

All output paths (logdir, tb_dir, docs_dir, logs_dir, images_dir, embed_dir, recon_dir, checkpoint_dir, report_path). The pipeline overwrites them in `__init__` to point at the resolved run directory.

## 4. Data path

[ProfileDataset](../pipelines/autoencoder_pipeline/data.py) flattens the upstream patch dataset (one item = `(C_in, p_h, p_w)` plus `(N_h, p_h, p_w)`) into a stream of 1-D profiles:

1. Iterate over patches, take **the magnitude of the target tomogram** of length `N_h`.
2. Optionally drop all-zero profiles, optionally subsample to `max_profiles`.
3. Apply `log_compress`: $p \leftarrow \log(p + \epsilon)$.
4. Apply normalization:

   - `per_profile_max`: $\hat p = p / (\max(p) + 10^{-8})$.
   - `per_profile_zscore`: $\hat p = (p - \mu_p) / (\sigma_p + 10^{-8})$.
   - `global`: $\hat p = (p - \mu_\text{global}) / \sigma_\text{global}$ from training-time statistics.

5. For contrastive training, `__getitem__` returns `(profile, view_a, view_b)` with views drawn per `contrastive_view`:
   - `augmentation`: both views are independent `Augmenter(profile)`.
   - `neighbor`: view_b is a *spatial neighbor* of view_a within the same patch.
   - `both`: the two strategies are randomly mixed.

[Augmenter.__call__](../pipelines/autoencoder_pipeline/data.py): in order — random scale $s\in U(s_\text{min},s_\text{max})$, additive jitter $\eta \sim \mathcal{N}(0,\sigma_j)$, integer cyclic shift in $[-S, S]$, random length-$w$ zero mask with probability $p_m$.

## 5. Model

[Autoencoder.forward()](../pipelines/autoencoder_pipeline/model.py) returns an `AutoencoderOutput(reconstruction, latent, projection)`:

- `Encoder` maps `(B, 1, L)` → `(B, latent_dim)`.
- `ProjectionHead` (optional) maps `latent` → `(B, proj_dim)` through `Linear → BatchNorm1d → activation → Linear → BatchNorm1d → … → Linear`.
- `Decoder` maps `(B, latent_dim)` → `(B, 1, L)`. For `conv1d` it uses transposed convolutions plus a final `F.interpolate` head to land exactly on `profile_length`.

`Layers` is a static factory selecting activation (`gelu`, `relu`, `silu`, `tanh`), 1-D normalization (`batch`, `layer`, `group`, `none`), and the optional decoder output activation.

## 6. Composite loss

[CompositeLoss.forward()](../pipelines/autoencoder_pipeline/losses.py#L66) consumes one or two `AutoencoderOutput`s.

### 6.1 Reconstruction

$$ \mathcal{L}_\text{rec} = \begin{cases}
\text{MSE}(\hat x, x) \\
\text{L1}(\hat x, x) \\
\text{SmoothL1}(\hat x, x) \\
\sqrt{(\hat x - x)^2 + \epsilon^2}\ \text{(Charbonnier)} \end{cases}$$

### 6.2 Variance regularization (VICReg)

For latent representation $z \in \mathbb{R}^{B \times d}$ (the projection if present, otherwise the latent):

$$
\sigma_j = \sqrt{\operatorname{Var}_b(z_{b,j}) + 10^{-4}}, \qquad
\mathcal{L}_\text{var} = \frac{1}{d}\sum_{j=1}^{d} \max(0,\ \tau_\sigma - \sigma_j),
$$

with target std $\tau_\sigma = $ `variance_target_std`. This pushes every latent dimension to keep at least unit variance, preventing dimensional collapse.

### 6.3 Covariance regularization

$$
C = \frac{1}{B-1} (z - \bar z)^\top (z - \bar z) \in \mathbb{R}^{d\times d},\qquad
\mathcal{L}_\text{cov} = \frac{1}{d} \sum_{i\neq j} C_{ij}^2.
$$

This decorrelates latent dimensions.

### 6.4 NT-Xent contrastive

For two normalized embeddings $z^a, z^b \in \mathbb{R}^{B\times d}$ and temperature $\tau$:

$$
\ell_{a\to b} = \text{CrossEntropy}\!\left(\frac{z^a (z^b)^\top}{\tau},\ \text{diag-labels}\right),\qquad
\mathcal{L}_\text{con} = \tfrac{1}{2}(\ell_{a\to b} + \ell_{b\to a}).
$$

### 6.5 Aggregate

$$
\mathcal{L} = w_\text{rec}\mathcal{L}_\text{rec} + w_\text{var}\mathcal{L}_\text{var} + w_\text{cov}\mathcal{L}_\text{cov} + w_\text{con}\mathcal{L}_\text{con}.
$$

When `output_b is None` (single-view), the variance and covariance terms use only `output_a` and the contrastive term is replaced by zero.

## 7. Trainer

[Trainer.fit()](../pipelines/autoencoder_pipeline/trainer.py):

- Builds AdamW with `(lr, wd)` from `TrainerConfig`.
- Builds the requested LR scheduler (cosine by default), with `warmup_steps` linear warmup applied per optimizer step.
- AMP (`torch.amp.GradScaler`) when enabled.
- Per epoch:
  1. Train pass with `MetricMeter` tracking weighted sums of every loss component (returns averaged dict on `result()`).
  2. Validation pass every `val_every` epochs.
  3. `EarlyStopping(patience=early_stop_patience)` on the validation total loss.
  4. Periodic checkpoint every `save_every` epochs (full model + encoder + decoder state dicts) and an additional “best” checkpoint when the validation total loss improves.
  5. Rich progress bars + Rich live tables; per-step TensorBoard logging (`loss/total`, `loss/components/*`, `train/throughput`, etc.).
  6. CSV row appended to `<run>/logs/training_loss.csv` per epoch.
- After training, `Plotter.plot_loss_history(history)` saves `images/loss_overview.png` and `images/loss_components_grid.png`.

The trainer exposes `trainer.best_epoch` and `trainer.best_val_total` to the pipeline for the final report.

## 8. Inference (per-split)

[Inference.run()](../pipelines/autoencoder_pipeline/inference.py):

1. Iterates the loader in `eval()` mode, no grad. Per batch collects:
   - input profiles, reconstructions, latents (and projections), per-profile MSE.
2. Persists everything to `<run>/embeddings/<split>.h5` (compressed) — see `_save_h5`.
3. Computes **reconstruction stats**: MSE / MAE / RMSE / PSNR / R² (mean / median / std).
4. Computes **embedding stats**: PCA over the latent matrix; reports
   - eigenvalues `λ_i`,
   - explained-variance ratio `λ_i / Σλ`,
   - participation ratio `(Σλ)² / Σλ²`,
   - effective rank `exp(H(p))` with $p_i = \lambda_i/\Sigma\lambda$,
   - dimension to reach 80 % / 95 % cumulative explained variance.
5. Saves galleries of best / worst / median / random reconstructions as NPZ + PNG/PDF figures.
6. Saves a 2-D PCA scatter (`embeddings/pca_<split>.png`) and an optional UMAP scatter when the package is importable. Both can be coloured by per-profile reconstruction MSE.
7. Writes JSON summaries `embeddings/<split>_recon_stats.json`, `<split>_embed_stats.json`.

Returns a dict consumed by `Reporter`.

## 9. Plotter

[Plotter](../pipelines/autoencoder_pipeline/plotter.py) is a thin Matplotlib helper using a serif RC override:

- `plot_loss_history` — train + val total loss curve.
- `plot_loss_components_grid` — multi-panel grid of all components.
- `plot_reconstructions` — overlay grid of original vs reconstruction.
- `plot_2d_projection` — PCA / UMAP scatter, optionally MSE-coloured.
- `plot_explained_variance` — bar + cumulative line.
- `plot_mse_histogram` — log-scale histogram of per-profile MSE.

All saves emit both PNG and PDF.

## 10. Reporter

[Reporter.write()](../pipelines/autoencoder_pipeline/reporter.py) generates `<run>/report.md` with:

- Header (run dir, latent / encoder / decoder summary).
- Configuration tables (model, loss weights, trainer, data).
- Training summary (epochs, best epoch, initial / final loss tables, `loss_overview.png`).
- Per-split sections (recon quality table, embedding quality table, gallery image, spectrum image, PCA/UMAP).
- Checkpoint table.
- Artifact listing (HDF5 paths).

## 11. Outputs

```python
{
  "history"           : dict,                       # train/val loss histories per component
  "inference_results" : {"val": {...}, "test": {...}},
  "checkpoint_paths"  : {                           # 6 entries
      "encoder_best",     "decoder_best",     "autoencoder_best",
      "encoder_final",    "decoder_final",    "autoencoder_final",
  },
  "report_path"       : str,
}
```
