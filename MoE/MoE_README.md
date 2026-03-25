# Dense Mixture of Experts for Pixel-Level Multi-Complexity Prediction

## 1. Introduction

In Tomographic Synthetic Aperture Radar (TomoSAR) reconstruction, the signal complexity varies spatially across the scene. A pixel containing a single scatterer can be adequately modelled by 3 parameters, whereas double- and triple-scatterer pixels require 6 and 9 parameters, respectively. A monolithic neural network forced to predict the maximum number of parameters everywhere will waste capacity on simple pixels and may still underfit complex ones.

The Mixture of Experts (MoE) paradigm [1] addresses this heterogeneity by combining *K* specialized sub-networks — each designed for a different output dimensionality — with a trainable gating mechanism that assigns experts to spatial locations. Crucially, in our formulation the gating decision is made *before* expert execution: the gate first inspects the input and determines which experts are needed, and only those experts process the data. This gate-first design avoids the computational waste of running all experts unconditionally.

This document formalizes the architecture, the three routing strategies, the composite loss function, and the supported training regimes.


## 2. Notation

The following table summarizes all symbols used throughout.

| Symbol | Description |
|:---|:---|
| *B* | Batch size |
| *H*, *W* | Spatial height and width of the input/output |
| *C_in* | Number of input channels |
| *K* | Number of experts |
| *C_k* | Number of output channels for expert *k* |
| *C_max* | Maximum output channels across all experts, equal to max(*C_1*, ..., *C_K*) |
| **x** | Input tensor of shape (*B*, *C_in*, *H*, *W*) |
| **y*** | Ground-truth target tensor of shape (*B*, *C_max*, *H*, *W*) |
| **ŷ** | Predicted output tensor of shape (*B*, *C_max*, *H*, *W*) |
| *E_k* | The *k*-th expert network |
| *g* | Gating network |
| **z** | Raw gating logits of shape (*B*, *K*, *H*, *W*) |
| tau | Temperature parameter for the softmax in the gating network |
| **G** | Gating probability tensor of shape (*B*, *K*, *H*, *W*), after softmax |
| **M** | Binary expert mask of shape (*B*, *K*, *H*, *W*) |
| A | Set of active expert indices determined by routing |
| S(b,h,w) | Set of selected expert indices for pixel *(b, h, w)* in top-*k* routing |
| **o_k** | Output of expert *k* before padding, shape (*B*, *C_k*, *H*, *W*) |
| **õ_k** | Output of expert *k* after zero-padding to *C_max* channels |
| **G̃** | Renormalized gating weights used in top-*k* aggregation |
| *k** | Winning expert index for a given pixel in hard routing |
| epsilon | Small constant for numerical stability |
| L_recon | Reconstruction loss |
| L_balance | Load-balance loss |
| L_entropy | Gating entropy loss |
| L_total | Weighted sum of all loss terms |
| lambda_recon | Weight for the reconstruction loss (default 1.0) |
| lambda_balance | Weight for the load-balance loss (default 0.01) |
| lambda_entropy | Weight for the entropy loss (default 0.01) |
| eta_E | Learning rate for expert parameters |
| eta_G | Learning rate for gating parameters |
| ell | Reconstruction loss function (e.g. MSE, L1, or Smooth L1) |


## 3. Problem Statement

The input is a batch of images **x** with shape *(B, C_in, H, W)*. The goal is to produce a dense prediction **ŷ** with shape *(B, C_max, H, W)*, where *C_max* is the largest output dimensionality among all experts. Each spatial location *(h, w)* may require a different number of active output channels, determined by the local signal complexity.


## 4. Architecture

```
                        ┌─────────────────────┐
                        │  Input (B,Cin,H,W)  │
                        └──────────┬──────────┘
                       ┌───────────┴──────────┐
                       │                      │
                       ▼                      │
              ┌─────────────────┐             │
              │   Gating g(x)   │             │
              │  softmax(z / τ) │             │
              └────────┬────────┘             │
                       │                      │
                       ▼                      │
              ┌─────────────────┐             │
              │     Routing     │             │
              │ soft/hard/top-k │             │
              └────────┬────────┘             │
                       │                      │
                       │  active set A        │
                       │                      │
            ┌──────────┼──────────┐           │
            │          │          │           │
            ▼          ▼          ▼           │
       ┌─────────┐┌─────────┐┌─────────┐     │
       │ Expert 1││ Expert 2││ Expert 3│◄────┘
       │ (C₁=3)  ││ (C₂=6)  ││ (C₃=9)  │  only active
       └────┬────┘└────┬────┘└────┬────┘  experts run
            │          │          │
            ▼          ▼          ▼
       ┌──────────────────────────────┐
       │    Pad to Cmax channels      │
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │    Aggregate (G-weighted)    │
       └──────────────┬───────────────┘
                      │
                      ▼
              ┌─────────────────────┐
              │ Output (B,Cmax,H,W) │
              └─────────────────────┘
```

The model comprises two components:

- **Expert pool**: *K* expert networks, where each *E_k* maps an input with *C_in* channels to an output with *C_k* channels. The experts have ordered output sizes *C_1 < C_2 < ... < C_K*.

- **Gating network** *g*: produces a per-pixel probability distribution over the *K* experts, outputting a tensor of shape *(B, K, H, W)*.

### 4.1 Gating Network

The gating network receives the same input as the experts. It first produces raw logits **z** of shape *(B, K, H, W)*, then converts them to assignment probabilities **G** via temperature-scaled softmax:

$$
G_{b,k,h,w} = \frac{\exp(z_{b,k,h,w} \;/\; \tau)}{\sum_{j=1}^{K} \exp(z_{b,j,h,w} \;/\; \tau)}
$$

A lower tau sharpens the distribution toward hard selection; a higher tau encourages softer mixing.

Three gating architectures are provided, offering a trade-off between capacity and cost:

| Architecture | Structure | Param. count |
|:---|:---|:---|
| **Lightweight CNN** | Sequential 3x3 conv blocks with batch normalization and max-pooling, followed by a 1x1 projection head and bilinear upsampling to input resolution. | Low |
| **Encoder–Only** | Multi-scale encoder with progressive downsampling, followed by symmetric upsampling stages (without skip connections), and a 1x1 head. | Medium |
| **Linear Probe** | A single 1x1 convolution applied directly to the input. | Minimal |

### 4.2 Routing

Given the gate probabilities **G**, a routing step determines the active expert set A and a binary mask **M** of shape *(B, K, H, W)*. Three modes are supported:

**Soft routing.** All experts are active. Every pixel is served by all *K* experts, each weighted by its gating probability:

$$
A = \lbrace 1, \dots, K \rbrace, \quad M_{b,k,h,w} = 1 \;\;\forall\; k
$$

**Hard routing.** Winner-take-all. Each pixel is assigned to the single highest-probability expert. The winning index *k** is:

$$
k^{\ast}(b,h,w) = \underset{k}{\text{argmax}} \; G_{b,k,h,w}
$$

The mask selects only the winner:

$$
M_{b,k,h,w} = \begin{cases} 1 & \text{if } k = k^{\ast}(b,h,w) \\\ 0 & \text{otherwise} \end{cases}
$$

Only experts that win at least one pixel across the entire batch are included in A.

**Top-*k* routing.** Each pixel is assigned to the *k* experts with the highest gating probabilities. Let S(b,h,w) denote that selected set. The selected weights are renormalized:

$$
\tilde{G}_{b,j,h,w} = \frac{G_{b,j,h,w}}{\sum_{j' \in S(b,h,w)} G_{b,j',h,w} + \epsilon} \quad \text{for } j \in S(b,h,w)
$$

Only experts that appear in at least one pixel's selection set are included in A.

### 4.3 Conditional Expert Execution

The key computational benefit of the gate-first design is that only experts in A are executed:

$$
o_k = E_k(x) \quad \text{for each } k \in A
$$

Each output *o_k* has shape *(B, C_k, H, W)*. Inactive experts are never evaluated, yielding proportional computational savings in hard and top-*k* modes.

### 4.4 Channel Padding

Because experts produce outputs of different dimensionalities, each expert output is zero-padded along the channel axis to *C_max* before aggregation:

$$
\tilde{o}_k = \text{Pad}(o_k, \; C_{\text{max}})
$$

The padded tensor has shape *(B, C_max, H, W)*. The extra channels carry a constant fill value (zero by default) and do not contribute meaningful gradients, ensuring that expert *k* is only supervised on its own *C_k* channels.

### 4.5 Aggregation

The padded expert outputs are combined into the final prediction **ŷ**.

**Soft aggregation** — weighted sum over all experts:

$$
\hat{y}_{b,:,h,w} = \sum_{k=1}^{K} G_{b,k,h,w} \cdot \tilde{o}_{k,b,:,h,w}
$$

**Hard aggregation** — output of the winning expert only:

$$
\hat{y}_{b,:,h,w} = \tilde{o}_{k^{\ast}(b,h,w),\;b,:,h,w}
$$

**Top-*k* aggregation** — renormalized weighted sum over selected experts:

$$
\hat{y}_{b,:,h,w} = \sum_{j \in S(b,h,w)} \tilde{G}_{b,j,h,w} \cdot \tilde{o}_{j,b,:,h,w}
$$


## 5. Loss Function

Training is guided by a composite objective consisting of three terms.

### 5.1 Reconstruction Loss

Two variants are supported.

**Aggregated reconstruction.** A standard pixel-wise loss applied to the final prediction:

$$
L_{\text{recon}} = \ell(\hat{y}, \; y^{\ast})
$$

where ell is a reconstruction criterion — MSE, L1, or Smooth L1 — and **y*** is the ground-truth target.

**Per-expert reconstruction.** Each expert is individually supervised on the first *C_k* channels of the target, weighted by its gating probability. This provides a stronger learning signal for specialization:

$$
L_{\text{per-expert}} = \sum_{k \in A} \frac{1}{BHW} \sum_{b,h,w} G_{b,k,h,w} \cdot \left\| o_{k,b,:,h,w} - y^{\ast}_{b,1:C_k,h,w} \right\|_2^2
$$

The weighting by **G** ensures that each expert's loss is concentrated on the pixels the gate has assigned to it.

### 5.2 Load-Balance Loss

Without regularization, the gating network can collapse to routing all pixels to a single expert, leaving the rest unused. The load-balance loss penalizes imbalanced utilization by measuring the squared coefficient of variation (CV²) of the mean expert load. First, compute the average gating probability per expert:

$$
\bar{g}_k = \frac{1}{BHW} \sum_{b,h,w} G_{b,k,h,w}
$$

Then:

$$
L_{\text{balance}} = \frac{\text{Var}(\bar{g}_1, \ldots, \bar{g}_K)}{\left[\text{Mean}(\bar{g}_1, \ldots, \bar{g}_K)\right]^2 + \epsilon}
$$

This term equals zero when all experts receive identical average load, and grows as the distribution becomes more skewed.

### 5.3 Gating Entropy Loss

The entropy of the per-pixel gating distribution measures how uncertain the gate is at each location. Minimizing entropy encourages the gate to make sharper, more decisive assignments:

$$
L_{\text{entropy}} = -\frac{1}{BHW} \sum_{b,h,w} \sum_{k=1}^{K} G_{b,k,h,w} \log G_{b,k,h,w}
$$

When the gate assigns all probability to one expert at every pixel, the entropy is zero. When the gate assigns uniform probability 1/*K* everywhere, the entropy reaches its maximum of log *K*.

**Interplay between balance and entropy.** These two losses exert competing pressures. The balance loss pushes toward uniform global utilization, while the entropy loss pushes toward deterministic per-pixel decisions. Together, they encourage the gate to specialize experts for different regions while keeping all experts active.

### 5.4 Total Objective

$$
L_{\text{total}} = \lambda_{\text{recon}} \cdot L_{\text{recon}} + \lambda_{\text{balance}} \cdot L_{\text{balance}} + \lambda_{\text{entropy}} \cdot L_{\text{entropy}}
$$

The default loss weights are:

| Weight | Default value | Role |
|:---|:---:|:---|
| lambda_recon | 1.0 | Reconstruction fidelity |
| lambda_balance | 0.01 | Expert utilization balance |
| lambda_entropy | 0.01 | Gating decision sharpness |


## 6. Training Regimes

Three training modes decouple the expert and gating learning phases.

| Mode | Expert params | Gating params | Use case |
|:---|:---:|:---:|:---|
| **End-to-end** | Trainable | Trainable | Joint optimization from scratch. |
| **Gate-only** | Frozen | Trainable | Pretrained experts; learn routing on top. |
| **Experts-only** | Trainable | Frozen | Fixed routing policy; refine expert weights. |

### 6.1 Optimization

The optimizer uses separate parameter groups with different learning rates: eta_E for expert parameters and eta_G for the gating network. In gate-only mode, eta_E is effectively zero (experts are frozen). Only parameters with active gradients are registered.

Available optimizers: Adam, AdamW, and SGD (with momentum 0.9). Learning rate scheduling follows cosine annealing or step decay.


## 7. Summary

The gate-first dense MoE architecture provides an efficient and principled approach to pixel-level tasks with spatially varying complexity. The gating network identifies the local complexity before routing, the conditional execution avoids redundant expert computation, and the composite loss balances reconstruction accuracy, expert utilization, and gating confidence.


## References

[1] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive Mixtures of Local Experts," *Neural Computation*, vol. 3, no. 1, pp. 79–87, 1991.

[2] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," in *Proc. ICLR*, 2017.

[3] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. S. Pinto, D. Keysers, and N. Houlsby, "Scaling Vision with Sparse Mixture of Experts," in *Proc. NeurIPS*, 2021.
