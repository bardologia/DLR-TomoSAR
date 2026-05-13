# Dense Mixture of Experts for Pixel-Level Multi-Complexity Prediction

## 1. Introduction

In Tomographic Synthetic Aperture Radar (TomoSAR) reconstruction, the signal complexity varies spatially across the scene. A pixel containing a single scatterer can be adequately modelled by 3 parameters, whereas double- and triple-scatterer pixels require 6 and 9 parameters, respectively. A monolithic neural network forced to predict the maximum number of parameters everywhere will waste capacity on simple pixels and may still underfit complex ones.

The Mixture of Experts (MoE) paradigm [1] addresses this heterogeneity by combining *K* specialized sub-networks — each designed for a different output dimensionality — with a trainable gating mechanism that assigns experts to spatial locations. Crucially, in our formulation the gating decision is made *before* expert execution: the gate first inspects the input and determines which experts are needed, and only those experts process the data. This gate-first design avoids the computational waste of running all experts unconditionally.

This document formalizes the architecture, the three routing strategies, the composite loss function, and the supported training regimes.


## 2. Notation

All operations are applied independently at every pixel. To keep equations clean, we write them **for a single pixel** and omit batch/spatial indices unless needed for clarity. Tensors have shape *(batch, channels, height, width)*.

| Symbol | Meaning |
|:---|:---|
| *K* | Number of experts |
| *x* | Input image |
| *y* | Ground-truth target |
| *E_k* | Expert network *k*, producing *C_k* output channels |
| *C_k* | Output channels of expert *k* (e.g. 3, 6, 9) |
| *C_max* | Largest *C_k* across all experts |
| *g* | Gating network |
| *z_k* | Raw logit for expert *k* (before softmax) |
| *p_k* | Gating probability for expert *k* (after softmax) |
| *T* | Temperature of the softmax |
| *A* | Set of active expert indices after routing |
| *w* | Winner expert index in hard routing |


## 3. Problem Statement

Given an input image *x*, the goal is a dense per-pixel prediction with *C_max* output channels. Each pixel may only need a subset of those channels (e.g. 3 out of 9), and the MoE framework lets the gating network select the appropriate expert for each location.


## 4. Architecture

```
                        ┌─────────────────────┐
                        │       Input x       │
                        └──────────┬──────────┘
                       ┌───────────┴──────────┐
                       │                      │
                       ▼                      │
              ┌─────────────────┐             │
              │  Gating g(x)    │             │
              │  softmax(z / T) │             │
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
       │  C=3    ││  C=6    ││  C=9    │  only active
       └────┬────┘└────┬────┘└────┬────┘  experts run
            │          │          │
            ▼          ▼          ▼
       ┌──────────────────────────────┐
       │  Pad all outputs to C_max    │
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │  Weighted aggregation        │
       └──────────────┬───────────────┘
                      │
                      ▼
              ┌─────────────────────┐
              │    Prediction ŷ     │
              └─────────────────────┘
```

The model has two components:

- **Expert pool**: *K* networks. Expert *k* takes the input *x* and produces an output with *C_k* channels. The experts are ordered by complexity: *C_1 < C_2 < ... < C_K* (e.g. 3, 6, 9).

- **Gating network** *g*: takes the same input *x* and produces one probability per expert per pixel.


### 4.1 Gating Network

The gating network outputs one logit *z_k* per expert, then converts to probabilities via temperature-scaled softmax:

$$
p_k = \frac{\exp(z_k \;/\; T)}{\sum_{j=1}^{K} \exp(z_j \;/\; T)}
$$

A low *T* makes the distribution sharp (close to one-hot); a high *T* makes it uniform. Three gating architectures are provided:

| Architecture | Structure | Cost |
|:---|:---|:---|
| **Lightweight CNN** | 3x3 conv blocks with batch norm and pooling, then a 1x1 head with bilinear upsampling. | Low |
| **Encoder–Only** | Multi-scale down/up path (no skip connections), then a 1x1 head. | Medium |
| **Linear Probe** | A single 1x1 convolution. | Minimal |


### 4.2 Routing

Using the probabilities *p_1, ..., p_K*, the routing step decides which experts are active. Three modes:

**Soft:** all experts are active, weighted by their probabilities.

**Hard (winner-take-all):** each pixel picks the single best expert:

$$
w = \underset{k}{\text{argmax}} \; p_k
$$

Only experts that win at least one pixel in the batch actually run.

**Top-*k*:** each pixel picks the *k* best experts; their probabilities are renormalized to sum to 1:

$$
p'_j = \frac{p_j}{\sum_{j' \in \text{top-}k} p_{j'}}
$$


### 4.3 Conditional Expert Execution

Only experts in the active set *A* are executed. For each active expert:

$$
o_k = E_k(x)
$$

Inactive experts are skipped entirely, saving computation in hard and top-*k* modes.


### 4.4 Channel Padding

Each expert output *o_k* has *C_k* channels, but aggregation requires a common size. All outputs are zero-padded to *C_max* channels:

$$
\tilde{o}_k = \text{Pad}(o_k, \; C_{\text{max}})
$$

The padded channels are filled with zeros and do not receive meaningful gradients, so expert *k* is only trained on its own *C_k* channels.


### 4.5 Aggregation

The padded outputs are combined into the final prediction.

**Soft** — weighted sum over all experts:

$$
\hat{y} = \sum_{k=1}^{K} p_k \cdot \tilde{o}_k
$$

**Hard** — output of the winning expert:

$$
\hat{y} = \tilde{o}_w
$$

**Top-*k*** — renormalized weighted sum over the selected experts:

$$
\hat{y} = \sum_{j \in \text{top-}k} p'_j \cdot \tilde{o}_j
$$


## 5. Loss Function

The total loss combines three terms.

### 5.1 Reconstruction Loss

**Aggregated variant.** A standard loss (MSE, L1, or Smooth L1) on the final prediction:

$$
L_{\text{recon}} = \text{loss}(\hat{y}, \; y)
$$

**Per-expert variant.** Each expert is supervised individually on the first *C_k* channels of the target, weighted by the gating probability. This gives each expert a direct learning signal on the pixels the gate assigns to it:

$$
L_{\text{expert}} = \sum_{k \in A} \; p_k \cdot \| \; o_k - y_{1:C_k} \; \|^2
$$

averaged over all pixels in the batch.


### 5.2 Load-Balance Loss

Prevents the gate from collapsing onto a single expert. It measures how evenly the experts are used across all pixels, via the squared coefficient of variation (CV²). Let *f_k* be the average gating probability for expert *k* across the batch:

$$
L_{\text{balance}} = \frac{\text{Var}(f_1, \ldots, f_K)}{\text{Mean}(f_1, \ldots, f_K)^2}
$$

This is zero when all experts get equal load, and grows when load is imbalanced.


### 5.3 Gating Entropy Loss

Encourages the gate to make confident (low-entropy) decisions at each pixel:

$$
L_{\text{entropy}} = - \sum_{k=1}^{K} p_k \log p_k
$$

averaged over all pixels. Entropy is zero when the gate is fully decisive (one expert gets probability 1) and maximal (log *K*) when the gate is uniform.

**Interplay.** These two losses pull in opposite directions: balance wants all experts used equally across the image, while entropy wants each individual pixel to commit to one expert. Together they encourage specialization — each expert owns a region of the image, but no expert is left idle.


### 5.4 Total Objective

$$
L_{\text{total}} = \lambda_r \cdot L_{\text{recon}} + \lambda_b \cdot L_{\text{balance}} + \lambda_e \cdot L_{\text{entropy}}
$$

| Weight | Default | Role |
|:---|:---:|:---|
| *λ_r* | 1.0 | Reconstruction fidelity |
| *λ_b* | 0.01 | Expert utilization balance |
| *λ_e* | 0.01 | Gating decision sharpness |


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
