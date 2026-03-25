# Dense Mixture of Experts for Pixel-Level Multi-Complexity Prediction

## 1. Introduction

In Tomographic Synthetic Aperture Radar (TomoSAR) reconstruction, the signal complexity varies spatially across the scene. A pixel containing a single scatterer can be adequately modelled by 3 parameters, whereas double- and triple-scatterer pixels require 6 and 9 parameters, respectively. A monolithic neural network forced to predict $C_{\max}$ parameters everywhere will waste capacity on simple pixels and may still underfit complex ones.

The Mixture of Experts (MoE) paradigm [1] addresses this heterogeneity by combining $K$ specialized sub-networks — each designed for a different output dimensionality — with a trainable gating mechanism that assigns experts to spatial locations. Crucially, in our formulation the gating decision is made *before* expert execution: the gate first inspects the input and determines which experts are needed, and only those experts process the data. This gate-first design avoids the computational waste of running all experts unconditionally.

This document formalizes the architecture, the three routing strategies, the composite loss function, and the supported training regimes.


## 2. Problem Statement

Let $\mathbf{x} \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}$ be a batch of input images. The task is to produce a dense prediction $\hat{\mathbf{y}} \in \mathbb{R}^{B \times C_{\max} \times H \times W}$, where $C_{\max} = \max_k C_k$ and $C_k$ is the output dimensionality of expert $k$. Each spatial location $(h, w)$ may require a different number of active output channels, determined by the local signal complexity.


## 3. Architecture

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
       ┌─────────┐┌─────────┐┌─────────┐      │
       │ Expert 1││ Expert 2││ Expert 3│◄─────┘
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
       │  Σ G_k · Pad(E_k(x), Cmax)   │
       └──────────────┬───────────────┘
                      │
                      ▼
              ┌─────────────────────┐
              │ Output (B,Cmax,H,W) |
              └─────────────────────┘
```

The model comprises two components:

- **Expert pool** $\{E_k\}_{k=1}^{K}$: each expert $E_k : \mathbb{R}^{C_{\text{in}} \times H \times W} \to \mathbb{R}^{C_k \times H \times W}$ is a dense prediction network (e.g., UNet). Different experts have different output channel counts ($C_1 < C_2 < \cdots < C_K$).

- **Gating network** $g : \mathbb{R}^{C_{\text{in}} \times H \times W} \to \mathbb{R}^{K \times H \times W}$: produces a per-pixel probability distribution over experts.

### 3.1 Gating Network

The gating network receives the same input as the experts and outputs a probability map. Let $\mathbf{z} = g_{\text{logits}}(\mathbf{x}) \in \mathbb{R}^{B \times K \times H \times W}$ denote the raw logits. The assignment probabilities are obtained via temperature-scaled softmax:

$$
\mathbf{G}_{b,k,h,w} = \frac{\exp\!\bigl(\mathbf{z}_{b,k,h,w} \,/\, \tau\bigr)}{\displaystyle\sum_{j=1}^{K} \exp\!\bigl(\mathbf{z}_{b,j,h,w} \,/\, \tau\bigr)}
$$

where $\tau > 0$ is the temperature. Lower temperatures sharpen the distribution toward hard selection; higher temperatures encourage softer mixing.

Three gating architectures are provided, offering a trade-off between capacity and cost:

| Architecture | Structure | Parameters |
|:---|:---|:---|
| **Lightweight CNN** | Sequential $3\!\times\!3$ convolution blocks with batch normalization and max-pooling, followed by a $1\!\times\!1$ projection head and bilinear upsampling to input resolution. | Low |
| **Encoder–Only** | Multi-scale encoder with progressive downsampling, followed by symmetric upsampling stages (without skip connections), and a $1\!\times\!1$ head. | Medium |
| **Linear Probe** | A single $1\!\times\!1$ convolution applied directly to the input. | Minimal |

### 3.2 Routing

Given the gate probabilities $\mathbf{G}$, a routing step determines the set of active experts $\mathcal{A}$ and an expert mask $\mathbf{M} \in \{0, 1\}^{B \times K \times H \times W}$. Three modes are supported:

**Soft routing.** All experts are active. Every pixel is served by all $K$ experts, weighted by $\mathbf{G}$:

$$
\mathcal{A} = \{1, \dots, K\}, \quad \mathbf{M}_{b,k,h,w} = 1 \;\;\forall\, k
$$

**Hard routing.** Winner-take-all. Each pixel is assigned to the single highest-probability expert:

$$
k^{*}(b,h,w) = \arg\max_{k} \; \mathbf{G}_{b,k,h,w}
$$

$$
\mathbf{M}_{b,k,h,w} = \mathbb{1}\bigl[k = k^{*}(b,h,w)\bigr]
$$

Only experts that win at least one pixel across the entire batch are included in $\mathcal{A}$.

**Top-$\boldsymbol{k}$ routing.** Each pixel is assigned to the $k$ experts with the highest gating probabilities. The selected weights are renormalized:

$$
\mathcal{S}(b,h,w) = \operatorname{top\text{-}k}\bigl(\mathbf{G}_{b,:,h,w}\bigr)
$$

$$
\tilde{\mathbf{G}}_{b,j,h,w} = \frac{\mathbf{G}_{b,j,h,w}}{\displaystyle\sum_{j' \in \mathcal{S}(b,h,w)} \mathbf{G}_{b,j',h,w} + \epsilon}
\quad \text{for } j \in \mathcal{S}(b,h,w)
$$

Only experts that appear in at least one pixel's selection set are included in $\mathcal{A}$.

### 3.3 Conditional Expert Execution

The key computational benefit of the gate-first design is that only experts in $\mathcal{A}$ are executed. For each active expert $k \in \mathcal{A}$:

$$
\mathbf{o}_k = E_k(\mathbf{x}) \in \mathbb{R}^{B \times C_k \times H \times W}
$$

Inactive experts are never evaluated, yielding proportional savings in hard and top-$k$ modes.

### 3.4 Channel Padding

Because experts produce outputs of different dimensionalities ($C_k \neq C_j$ in general), each expert output is zero-padded along the channel axis to $C_{\max}$ before aggregation:

$$
\tilde{\mathbf{o}}_k = \operatorname{Pad}(\mathbf{o}_k,\; C_{\max}) \in \mathbb{R}^{B \times C_{\max} \times H \times W}
$$

The padded channels carry a constant fill value (zero by default) and do not contribute meaningful gradients, ensuring that expert $k$ is only supervised on its own $C_k$ channels.

### 3.5 Aggregation

The padded expert outputs are combined into the final prediction $\hat{\mathbf{y}}$.

**Soft aggregation:**

$$
\hat{\mathbf{y}}_{b,:,h,w} = \sum_{k=1}^{K} \mathbf{G}_{b,k,h,w} \cdot \tilde{\mathbf{o}}_{k,\,b,:,h,w}
$$

**Hard aggregation:**

$$
\hat{\mathbf{y}}_{b,:,h,w} = \tilde{\mathbf{o}}_{k^{*}(b,h,w),\; b,:,h,w}
$$

**Top-$k$ aggregation:**

$$
\hat{\mathbf{y}}_{b,:,h,w} = \sum_{j \in \mathcal{S}(b,h,w)} \tilde{\mathbf{G}}_{b,j,h,w} \cdot \tilde{\mathbf{o}}_{j,\,b,:,h,w}
$$


## 4. Loss Function

Training is guided by a composite objective consisting of three terms.

### 4.1 Reconstruction Loss

Two variants are supported depending on the training configuration.

**Aggregated reconstruction.** A standard pixel-wise loss applied to the final (aggregated) prediction:

$$
\mathcal{L}_{\text{recon}} = \ell\!\bigl(\hat{\mathbf{y}},\; \mathbf{y}^{*}\bigr)
$$

where $\ell$ is MSE, $L_1$, or Smooth-$L_1$, and $\mathbf{y}^{*}$ is the ground-truth target.

**Per-expert reconstruction.** Each expert is individually supervised on the first $C_k$ channels of the target, weighted by its gating probability. This provides a stronger learning signal for specialization:

$$
\mathcal{L}_{\text{per-expert}} = \sum_{k \in \mathcal{A}} \frac{1}{BHW} \sum_{b,h,w} \mathbf{G}_{b,k,h,w} \cdot \bigl\| \mathbf{o}_{k,\,b,:,h,w} - \mathbf{y}^{*}_{b,\,1:C_k,\,h,w} \bigr\|_2^2
$$

The weighting by $\mathbf{G}$ ensures that each expert's reconstruction loss is concentrated on the pixels that the gate has assigned to it.

### 4.2 Load-Balance Loss

Without regularization, the gating network can collapse to routing all pixels to a single expert, leaving the remaining experts unused. The load-balance loss penalizes imbalanced utilization by measuring the squared coefficient of variation ($\mathrm{CV}^2$) of the mean expert assignment:

$$
\bar{g}_k = \frac{1}{BHW} \sum_{b,h,w} \mathbf{G}_{b,k,h,w}
$$

$$
\mathcal{L}_{\text{balance}} = \frac{\operatorname{Var}(\bar{g}_1, \ldots, \bar{g}_K)}{\bigl[\operatorname{Mean}(\bar{g}_1, \ldots, \bar{g}_K)\bigr]^2 + \epsilon}
$$

This term equals zero when all experts receive identical average load ($\bar{g}_1 = \cdots = \bar{g}_K$), and grows as the load distribution becomes more skewed.

### 4.3 Gating Entropy Loss

The entropy of the per-pixel gating distribution measures how uncertain the gate is at each location. Minimizing entropy encourages the gate to make sharper, more decisive assignments:

$$
\mathcal{L}_{\text{entropy}} = -\frac{1}{BHW} \sum_{b,h,w} \sum_{k=1}^{K} \mathbf{G}_{b,k,h,w} \log \mathbf{G}_{b,k,h,w}
$$

The minimum ($\mathcal{L}_{\text{entropy}} = 0$) is achieved when the gate assigns probability 1 to a single expert at every pixel. The maximum ($\mathcal{L}_{\text{entropy}} = \log K$) occurs when the gate assigns uniform probability $1/K$ everywhere.

**Interplay between balance and entropy.** The load-balance loss and the entropy loss exert competing pressures. The balance loss pushes toward uniform global utilization, while the entropy loss pushes toward deterministic per-pixel decisions. Together, they encourage the gate to specialize experts for different regions while keeping all experts active.

### 4.4 Total Objective

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{recon}} \, \mathcal{L}_{\text{recon}} + \lambda_{\text{balance}} \, \mathcal{L}_{\text{balance}} + \lambda_{\text{entropy}} \, \mathcal{L}_{\text{entropy}}
$$

Default weights: $\lambda_{\text{recon}} = 1.0$, $\lambda_{\text{balance}} = 0.01$, $\lambda_{\text{entropy}} = 0.01$.


## 5. Training Regimes

Three training modes decouple the expert and gating learning phases.

| Mode | Expert parameters | Gating parameters | Use case |
|:---|:---:|:---:|:---|
| **End-to-end** | Trainable | Trainable | Joint optimization from scratch. |
| **Gate-only** | Frozen | Trainable | Pretrained experts; learn routing on top. |
| **Experts-only** | Trainable | Frozen | Fixed routing policy; refine expert weights. |

In gate-only mode, separate learning rates are used for experts (frozen, $\eta_E = 0$) and the gating network ($\eta_G$), enabling stable training when only one component is active.

### 5.1 Optimization

The optimizer supports separate parameter groups with different learning rates for experts ($\eta_E$) and gating ($\eta_G$). Only parameters with active gradients are registered. Available optimizers include Adam, AdamW, and SGD (with momentum 0.9). Learning rate scheduling follows cosine annealing or step decay.


## 6. Summary

The gate-first dense MoE architecture provides an efficient and principled approach to pixel-level tasks with spatially varying complexity. The gating network identifies the local complexity before routing, the conditional execution avoids redundant expert computation, and the composite loss balances reconstruction accuracy, expert utilization, and gating confidence.


## References

[1] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive Mixtures of Local Experts," *Neural Computation*, vol. 3, no. 1, pp. 79–87, 1991.

[2] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," in *Proc. ICLR*, 2017.

[3] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. S. Pinto, D. Keysers, and N. Houlsby, "Scaling Vision with Sparse Mixture of Experts," in *Proc. NeurIPS*, 2021.
