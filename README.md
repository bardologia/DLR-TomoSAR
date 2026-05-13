# Training Pipeline — Technical Reference

## Table of Contents

1. [Overview](#1-overview)
2. [Gaussian Curve Reconstruction](#2-gaussian-curve-reconstruction)
3. [Loss Function](#3-loss-function)
4. [Optimizer and Parameter Groups](#4-optimizer-and-parameter-groups)
5. [Learning Rate Warmup](#5-learning-rate-warmup)
6. [Learning Rate Scheduler](#6-learning-rate-scheduler)
7. [Exponential Moving Average (EMA)](#7-exponential-moving-average-ema)
8. [Early Stopping](#8-early-stopping)
9. [Mixed-Precision Training](#9-mixed-precision-training)
10. [Gradient Accumulation](#10-gradient-accumulation)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Checkpointing](#12-checkpointing)
13. [Training Loop](#13-training-loop)

---

## 1. Overview

The training pipeline optimises a segmentation or regression model to predict per-pixel Gaussian mixture parameters from input images. The model output is a tensor of shape $(B, C, H, W)$, where $C = 3K$ (or $3K + 1$ when a heteroscedastic noise head is enabled) and $K$ is the number of Gaussian components. The predicted parameters are used to reconstruct spectral curves, which are compared against experimental curves to compute the loss.

Supported model architectures:

| Key | Architecture |
|---|---|
| `unet` | UNet |
| `resunet` | ResUNet |
| `attention_unet` | Attention UNet |
| `unetplusplus` | UNet++ |
| `fcn` | Fully Convolutional Network |
| `linknet` | LinkNet |
| `swin_unet` | Swin-UNet |
| `transunet` | TransUNet |
| `unetr` | UNETR |

---

## 2. Gaussian Curve Reconstruction

Given $K$ Gaussian components, the model predicts $3K$ channels per pixel:

$$
\hat{\mathbf{p}} = [a_1, \mu_1, \sigma_1, \; a_2, \mu_2, \sigma_2, \; \dots, \; a_K, \mu_K, \sigma_K]
$$

where $a_k$ is the amplitude, $\mu_k$ is the mean, and $\sigma_k$ is the standard deviation of the $k$-th Gaussian.

For a discrete set of $N$ sample points $\{x_n\}_{n=1}^{N}$ along the spectral axis, the reconstructed curve at each pixel is the superposition:

$$
\hat{y}(x_n) = \sum_{k=1}^{K} a_k \exp\!\left( -\frac{(x_n - \mu_k)^2}{2\sigma_k^2 + \epsilon} \right)
$$

where $\epsilon = 10^{-8}$ prevents division by zero.

The output tensor has shape $(B, N, H, W)$.

---

## 3. Loss Function

### 3.1 Standard Mode (No Noise Head)

When the number of output channels equals $3K$, the loss is the mean squared error between the reconstructed and experimental curves:

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{B \cdot N \cdot H \cdot W} \sum_{b,n,h,w} \left( \hat{y}_{b,n,h,w} - y_{b,n,h,w} \right)^2
$$

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}
$$

### 3.2 Heteroscedastic Noise Mode

When the number of output channels exceeds $3K$, the model predicts an additional channel $\log \sigma_{\text{noise}}$, representing the log-variance of a per-pixel noise estimate. The noise standard deviation is obtained as:

$$
\sigma_{\text{noise}} = \text{clamp}\!\left( \exp(\log \sigma_{\text{noise}}), \; 10^{-4}, \; 10.0 \right)
$$

The loss is the mean of a Gaussian negative log-likelihood:

$$
\mathcal{L}_{\text{NLL}} = \frac{1}{B \cdot N \cdot H \cdot W} \sum_{b,n,h,w} \left[ \frac{(\hat{y}_{b,n,h,w} - y_{b,n,h,w})^2}{2(\sigma_{\text{noise}}^2 + \epsilon)} + \log \sigma_{\text{noise}} \right]
$$

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL}}
$$

where $\epsilon = 10^{-8}$.

All loss values are logged in log-scale via $\log(1 + \ell)$.

---

## 4. Optimizer and Parameter Groups

The optimizer is **AdamW** with configurable betas $(\beta_1, \beta_2)$ and $\epsilon$.

Model parameters are partitioned into two groups with independent learning rates and weight decay coefficients:

| Group | Learning Rate | Weight Decay | Parameters |
|---|---|---|---|
| `backbone` | `lr_backbone` | `weight_decay_backbone` | All parameters not belonging to the output head |
| `output_head` | `lr_output_head` | `weight_decay_output_head` | Parameters of `output_head`, `output_heads`, `score_final`, `score_pool4`, `score_pool3` |

The AdamW update rule for each parameter $\theta$ with learning rate $\eta$, weight decay $\lambda$, and moments $m_t$, $v_t$ is:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

---

## 5. Learning Rate Warmup

When enabled, the warmup phase linearly scales the learning rate from an initial factor to 1.0 over a configurable number of steps.

At warmup step $s$ (where $1 \le s \le S_{\text{warmup}}$):

$$
\alpha(s) = \alpha_{\text{start}} + \left(1 - \alpha_{\text{start}}\right) \cdot \frac{s}{S_{\text{warmup}}}
$$

$$
\eta(s) = \alpha(s) \cdot \eta_{\text{base}}
$$

where $\alpha_{\text{start}}$ is the warmup start factor, $S_{\text{warmup}}$ is the total number of warmup steps, and $\eta_{\text{base}}$ is the base learning rate for each parameter group.

After step $S_{\text{warmup}}$, the factor is set to $1.0$ and the warmup phase terminates.

**Configuration parameters:**

| Parameter | Description |
|---|---|
| `warmup_enabled` | Boolean flag to enable/disable warmup |
| `warmup_steps` | Number of warmup steps $S_{\text{warmup}}$ |
| `warmup_start_factor` | Initial scaling factor $\alpha_{\text{start}}$ |

---

## 6. Learning Rate Scheduler

The scheduler is **Cosine Annealing** (`CosineAnnealingLR`). It is applied per epoch after the warmup phase is completed.

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t \cdot \pi}{T_{\max}}\right)\right)
$$

where $\eta_{\max}$ is the initial learning rate (post-warmup), $\eta_{\min}$ is the minimum learning rate (`eta_min`), $t$ is the current epoch, and $T_{\max}$ is the total number of scheduled epochs.

The scheduler does not step during the warmup phase.

---

## 7. Exponential Moving Average (EMA)

When enabled, EMA maintains shadow copies of all trainable parameters. At each optimiser step $t$, the shadow parameters are updated:

$$
\tilde{\theta}_t = \gamma \, \tilde{\theta}_{t-1} + (1 - \gamma) \, \theta_t
$$

where $\gamma \in [0, 1)$ is the decay coefficient and $\theta_t$ are the current model parameters.

During evaluation, the shadow parameters $\tilde{\theta}$ replace the model parameters. After evaluation, the original parameters $\theta$ are restored.

**Tracked diagnostics (per step):**

| Metric | Definition |
|---|---|
| Parameter divergence | $\sum_i \|\tilde{\theta}_i - \theta_i\|_2$ |
| Shadow norm | $\sum_i \|\tilde{\theta}_i\|_2$ |
| Model norm | $\sum_i \|\theta_i\|_2$ |
| Norm ratio | $\frac{\text{shadow norm}}{\text{model norm} + 10^{-8}}$ |

---

## 8. Early Stopping

Training terminates when the validation loss does not improve by at least $\delta_{\min}$ for $P$ consecutive epochs.

The stopping criterion at epoch $t$ is:

$$
\text{stop}(t) =
\begin{cases}
\text{True}  & \text{if } \ell_{\text{val}}(t') \geq \ell^{*}_{\text{val}} - \delta_{\min} \;\; \forall \, t' \in \{t - P + 1, \dots, t\} \\
\text{False} & \text{otherwise}
\end{cases}
$$

where $\ell^{*}_{\text{val}}$ is the best recorded validation loss and $P$ is the patience.

When `restore_best` is enabled, the model weights are reverted to the state corresponding to $\ell^{*}_{\text{val}}$ upon early termination.

**Configuration parameters:**

| Parameter | Description |
|---|---|
| `patience` | Number of epochs $P$ without improvement before stopping |
| `min_delta` | Minimum improvement threshold $\delta_{\min}$ |
| `restore_best` | Whether to restore the best model state on stop |

---

## 9. Mixed-Precision Training

When `use_amp` is enabled and a CUDA device is available, forward passes execute under `torch.amp.autocast("cuda")`, which selects lower-precision floating-point types (e.g., float16) for eligible operations. Gradient scaling via `torch.amp.GradScaler` prevents underflow during backpropagation:

1. The loss is scaled by a dynamic factor before `.backward()`.
2. Before the optimiser step, gradients are unscaled.
3. The scale factor is updated based on the presence of `inf`/`NaN` gradients.

---

## 10. Gradient Accumulation

Gradient accumulation allows the effective batch size to exceed the physical batch size by deferring the optimiser step over $A$ mini-batches. Each mini-batch loss is divided by $A$:

$$
\mathcal{L}_{\text{accum}} = \frac{\mathcal{L}_{\text{total}}}{A}
$$

The optimiser steps when $(i + 1) \mod A = 0$ or when the current batch is the last in the epoch, where $i$ is the zero-indexed batch index within the epoch.

Gradients are clipped to a maximum norm before each optimiser step:

$$
\mathbf{g} \leftarrow \frac{\mathbf{g}}{\max\!\left(1, \; \frac{\|\mathbf{g}\|_2}{g_{\max}}\right)}
$$

where $g_{\max}$ is the configured `max_grad_norm`.

---

## 11. Evaluation Metrics

All metrics are computed over the full evaluation set by concatenating batch predictions and targets. Let $\hat{y} \in \mathbb{R}^{B \times N \times H \times W}$ denote reconstructed curves and $y$ the experimental curves.

### 11.1 Per-Pixel Curve MSE

$$
\text{MSE}_{b,h,w} = \frac{1}{N} \sum_{n=1}^{N} (\hat{y}_{b,n,h,w} - y_{b,n,h,w})^2
$$

### 11.2 Per-Pixel Curve MAE

$$
\text{MAE}_{b,h,w} = \frac{1}{N} \sum_{n=1}^{N} |\hat{y}_{b,n,h,w} - y_{b,n,h,w}|
$$

### 11.3 Curve RMSE

$$
\text{RMSE} = \sqrt{ \frac{1}{B \cdot H \cdot W} \sum_{b,h,w} \text{MSE}_{b,h,w} }
$$

### 11.4 Per-Pixel Coefficient of Determination ($R^2$)

For each pixel $(b, h, w)$:

$$
\text{SS}_{\text{res}} = \sum_{n=1}^{N} (\hat{y}_{b,n,h,w} - y_{b,n,h,w})^2
$$

$$
\text{SS}_{\text{tot}} = \sum_{n=1}^{N} (y_{b,n,h,w} - \bar{y}_{b,h,w})^2
$$

$$
R^2_{b,h,w} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}} + \epsilon}
$$

where $\bar{y}_{b,h,w} = \frac{1}{N}\sum_{n} y_{b,n,h,w}$ and $\epsilon = 10^{-8}$.

Reported statistics: mean, standard deviation, median, minimum, maximum over all pixels.

### 11.5 Overall $R^2$

$$
R^2_{\text{overall}} = 1 - \frac{\sum_{b,n,h,w}(\hat{y}_{b,n,h,w} - y_{b,n,h,w})^2}{\sum_{b,n,h,w}(y_{b,n,h,w} - \bar{y})^2}
$$

where $\bar{y}$ is the global mean of all experimental curve values.

### 11.6 Per-Pixel Cosine Similarity

$$
\text{CosSim}_{b,h,w} = \frac{\sum_{n} \hat{y}_{b,n,h,w} \cdot y_{b,n,h,w}}{\|\hat{y}_{b,\cdot,h,w}\|_2 \; \|y_{b,\cdot,h,w}\|_2}
$$

Reported statistics: mean, standard deviation, median over all pixels.

### 11.7 Parameter Distribution Statistics

For each predicted Gaussian parameter channel $p \in \{a_k, \mu_k, \sigma_k\}_{k=1}^{K}$, the following statistics are computed: mean, standard deviation, minimum, maximum.

When the noise head is active, statistics are computed for $\log \sigma_{\text{noise}}$ and $\sigma_{\text{noise}} = \text{clamp}(\exp(\log \sigma_{\text{noise}}), 10^{-4}, 10.0)$.

---

## 12. Checkpointing

A checkpoint is saved whenever the validation loss improves. The checkpoint contains:

| Field | Content |
|---|---|
| `epoch` | Epoch number at save time |
| `global_step` | Total optimiser steps executed |
| `best_val_loss` | Best recorded validation loss |
| `best_epoch` | Epoch corresponding to best validation loss |
| `best_metrics` | Metric dictionary at best epoch |
| `train_losses` | List of training losses per epoch |
| `val_losses` | List of validation losses per epoch |
| `model_state_dict` | Model parameters |
| `optimizer_state_dict` | Optimiser state |
| `lr_scheduler_state_dict` | Scheduler state |
| `ema_state_dict` | EMA shadow parameters, decay, and enabled flag |
| `early_stopping_state` | Best loss, counter, and best model state |
| `warmup_state` | Current step and completion flag |
| `scaler_state_dict` | Gradient scaler state (if AMP is enabled) |
| `config` | Training configuration |
| `x_axis` | Spectral axis sample points |

Checkpoints are stored at `<run_dir>/best_model.pt`.

Loading a checkpoint restores all fields and resumes training from the saved epoch.

---

## 13. Training Loop

The training procedure follows these steps for each epoch $t \in \{1, \dots, T\}$:

1. **Train epoch**: iterate over the training data loader, compute forward pass, compute loss, accumulate gradients, and step the optimiser (with warmup, gradient clipping, and EMA updates).
2. **Evaluate on validation set**: apply EMA parameters, compute loss and all metrics (Section 11), restore original parameters.
3. **Evaluate on training set**: same procedure as validation, on training data.
4. **Log loss comparison**: record training and validation losses.
5. **Checkpoint**: if $\ell_{\text{val}}(t) < \ell^{*}_{\text{val}}$, save checkpoint.
6. **Scheduler step**: advance cosine annealing (if warmup is complete).
7. **Early stopping check**: if the stopping criterion (Section 8) is met, terminate.

After training completes (either by exhausting all epochs or by early stopping):

1. Load the best checkpoint.
2. Evaluate on training, validation, and test sets (`final_train`, `final_validation`, `final_test`).
3. Return the three result dictionaries.

**Overfitting mode**: when enabled, a single batch is sampled from the training loader and repeated for the entire epoch. This mode is used for debugging and verifying model capacity.

**Activation and weight logging**: every 10 epochs (after epoch 0), forward-hook-based activation distributions and model weight distributions are logged.

**Gradient and optimiser diagnostics**: every 100 optimiser steps, gradient norms and optimiser state statistics are logged.
