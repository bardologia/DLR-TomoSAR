"""
sweep_sigma_order_selection.py
==============================
Large-scale GPU sweep: σ-only Adam fit + model-order selection via complexity penalty.

Strategy
--------
- CPU  : prominence init for all pixels × all K  (scipy, not batchable)
- GPU  : JAX-batched σ-only Adam (all N pixels in one jit+scan kernel per K)
- CPU  : argmin over penalised loss → best K per pixel; plots + stats

Usage
-----
    python sweep_sigma_order_selection.py [--tomogram PATH] [--n_sweep 5000] [--k_max 5]
                                         [--steps 800] [--lr 0.01] [--lambda_k 0.005]
                                         [--seed 123] [--outdir ./sweep_results]
"""

import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed on remote
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="GPU σ-sweep over a tomogram")
    p.add_argument("--tomogram",  type=str,
                   default="/ste/rnd/User/vice_vi/Dataset/new_toy/data/"
                           "tomogram_reduced_1000a1050a500a550_dtmf_Xtomo_id2X.npy")
    p.add_argument("--h_min",    type=float, default=-50.0)
    p.add_argument("--h_max",    type=float, default= 50.0)
    p.add_argument("--n_sweep",  type=int,   default=5000)
    p.add_argument("--k_max",    type=int,   default=5)
    p.add_argument("--steps",    type=int,   default=800)
    p.add_argument("--lr",       type=float, default=1e-2)
    p.add_argument("--lambda_k", type=float, default=5e-3)
    p.add_argument("--prom_frac",type=float, default=0.05)
    p.add_argument("--seed",     type=int,   default=123)
    p.add_argument("--outdir",   type=str,   default="./sweep_results")
    p.add_argument("--b1",       type=float, default=0.9)
    p.add_argument("--b2",       type=float, default=0.999)
    p.add_argument("--eps",      type=float, default=1e-8)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Prominence init  (CPU / scipy)
# ─────────────────────────────────────────────────────────────────────────────
def prominence_init(profile_raw: np.ndarray, height_axis: np.ndarray,
                    n_gaussians: int, prominence_frac: float = 0.05):
    H           = len(height_axis)
    sigma_guess = float((height_axis[-1] - height_axis[0]) / (4.0 * n_gaussians))
    min_dist    = max(1, int(sigma_guess / (height_axis[1] - height_axis[0])))
    smoothed    = uniform_filter1d(profile_raw.astype(np.float32), size=5, mode="nearest")
    pmax        = smoothed.max()
    params      = np.zeros(3 * n_gaussians, dtype=np.float32)

    if pmax < 1e-10:
        idxs = np.linspace(0, H - 1, n_gaussians, dtype=int)
    else:
        peaks, props = find_peaks(smoothed,
                                  prominence=pmax * prominence_frac,
                                  distance=min_dist)
        if len(peaks) >= n_gaussians:
            order = np.argsort(props["prominences"])[::-1][:n_gaussians]
            idxs  = peaks[order]
        elif len(peaks) > 0:
            residual        = smoothed.copy()
            residual[peaks] = 0.0
            extra_idxs      = []
            for _ in range(n_gaussians - len(peaks)):
                ei = int(np.argmax(residual))
                extra_idxs.append(ei)
                lo = max(0, ei - min_dist)
                hi = min(H, ei + min_dist + 1)
                residual[lo:hi] = 0.0
            idxs = np.concatenate([peaks, np.array(extra_idxs, dtype=int)])
        else:
            idxs = np.linspace(0, H - 1, n_gaussians, dtype=int)

    for g, idx in enumerate(idxs[:n_gaussians]):
        params[g*3 + 0] = max(float(smoothed[idx]), 1e-10)
        params[g*3 + 1] = float(height_axis[idx])
        params[g*3 + 2] = sigma_guess

    return params


# ─────────────────────────────────────────────────────────────────────────────
# JAX batched σ-only Adam
# ─────────────────────────────────────────────────────────────────────────────
@partial(jax.jit, static_argnums=(3, 4))
def adam_sigma_only_jax_batch(init_params_batch, profiles_batch, h_jax,
                               n_steps: int, K: int,
                               lr: float, lambda_k: float,
                               b1: float, b2: float, eps: float,
                               sig_clip: float = 50.0):
    """
    σ-only Adam for a full batch of pixels.

    Parameters
    ----------
    init_params_batch : (N, 3*K)  normalised init params [amp, mu, sigma] × K
    profiles_batch    : (N, H)    normalised profiles
    h_jax             : (H,)      height axis
    sig_clip          : float     upper bound for σ (= height span / 2)
    Returns: sig_final (N,K), final_mse (N,), penalised_loss (N,)
    """
    amp_init = init_params_batch[:, 0::3]   # (N, K)
    mu_init  = init_params_batch[:, 1::3]   # (N, K)
    sig_init = init_params_batch[:, 2::3]   # (N, K)

    def step_fn(carry, _):
        sig, m, v, t = carry
        d     = h_jax[None, None, :] - mu_init[:, :, None]          # (N, K, H)
        e     = jnp.exp(-(d**2) / (2.0 * jnp.maximum(sig, 1e-6)[:, :, None]**2))
        pred  = jnp.sum(amp_init[:, :, None] * e, axis=1)           # (N, H)
        resid = pred - profiles_batch                                # (N, H)
        mse   = jnp.mean(resid**2, axis=1)                          # (N,)

        scale  = 2.0 * resid[:, None, :] / h_jax.shape[0]          # (N,1,H)
        g_sig  = jnp.sum(scale * amp_init[:, :, None] * e *
                         (d**2 / jnp.maximum(sig, 1e-6)[:, :, None]**3), axis=2)

        m_new  = b1 * m + (1 - b1) * g_sig
        v_new  = b2 * v + (1 - b2) * g_sig**2
        m_hat  = m_new / (1 - b1**t)
        v_hat  = v_new / (1 - b2**t)
        sig_new = jnp.clip(sig - lr * m_hat / (jnp.sqrt(v_hat) + eps), 1e-4, sig_clip)
        return (sig_new, m_new, v_new, t + 1), mse

    sig0 = jnp.clip(sig_init, 1e-4, sig_clip)
    (sig_final, _, _, _), mse_hist = jax.lax.scan(
        step_fn, (sig0, jnp.zeros_like(sig0), jnp.zeros_like(sig0), jnp.array(1.0)),
        None, length=n_steps
    )
    final_mse = mse_hist[-1]
    pen_loss  = final_mse + lambda_k * K * jnp.mean(amp_init, axis=1)
    return sig_final, final_mse, pen_loss


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
COLORS = plt.cm.tab10.colors

def draw_pixel_grid(pixel_idxs, profiles_n, h_axis, sweep_best_K,
                    sweep_mse, init_stack, all_sig_final, title, mix_color, outpath):
    COLS_W = 6
    ROWS_W = int(np.ceil(len(pixel_idxs) / COLS_W))
    fig, axes = plt.subplots(ROWS_W, COLS_W,
                             figsize=(COLS_W * 3, ROWS_W * 2.5),
                             constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for plot_i, sw_i in enumerate(pixel_idxs):
        ax      = axes[plot_i]
        K_b     = int(sweep_best_K[sw_i])
        k_idx_b = K_b - 1
        ip      = init_stack[k_idx_b, sw_i, :K_b*3].copy()
        sig_b   = all_sig_final[k_idx_b][sw_i, :K_b]

        ax.plot(h_axis, profiles_n[sw_i], color="black", lw=1.0, alpha=0.5)
        mix = np.zeros_like(h_axis)
        for g in range(K_b):
            a_g  = ip[g*3]
            mu_g = ip[g*3+1]
            sg   = max(float(sig_b[g]), 1e-6)
            comp = a_g * np.exp(-((h_axis - mu_g)**2) / (2*sg**2))
            mix += comp
            ax.plot(h_axis, comp, lw=0.8, ls="--",
                    color=COLORS[g % len(COLORS)], alpha=0.7)
        ax.plot(h_axis, mix, color=mix_color, lw=1.4)
        ax.set_title(f"#{plot_i+1} K={K_b} MSE={sweep_mse[sw_i]:.4f}", fontsize=7)
        ax.set_ylim(-0.05, 1.2)
        ax.tick_params(labelsize=5)
    for j in range(len(pixel_idxs), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=9)
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"JAX devices : {jax.devices()}")
    print(f"Tomogram    : {args.tomogram}")

    # ── Load data ──────────────────────────────────────────────────────────
    tomo        = np.load(args.tomogram, mmap_mode="r", allow_pickle=False)
    H, Az, R    = tomo.shape
    height_axis = np.linspace(args.h_min, args.h_max, H, dtype=np.float32)
    print(f"Shape: H={H} Az={Az} R={R}  |  height {args.h_min}→{args.h_max} m")

    rng    = np.random.default_rng(args.seed)
    az_sw  = rng.integers(0, Az, args.n_sweep)
    r_sw   = rng.integers(0, R,  args.n_sweep)

    print(f"Loading {args.n_sweep} profiles...", end=" ", flush=True)
    t0 = time.time()
    profiles_raw = np.abs(tomo[:, az_sw, r_sw]).T.astype(np.float64)  # (N, H)
    norms        = profiles_raw.max(axis=1, keepdims=True)
    norms        = np.where(norms < 1e-10, 1.0, norms)
    profiles_n   = profiles_raw / norms
    print(f"{time.time()-t0:.1f}s")

    # ── CPU: build init params for every pixel × every K ──────────────────
    PAD        = args.k_max * 3
    init_stack = np.zeros((args.k_max, args.n_sweep, PAD), dtype=np.float64)
    print("Building prominence inits (CPU):")
    t0 = time.time()
    for k_idx, K in enumerate(range(1, args.k_max + 1)):
        for n in range(args.n_sweep):
            ip = prominence_init(profiles_raw[n], height_axis, K, args.prom_frac).astype(np.float64)
            ip[0::3] /= norms[n, 0]
            init_stack[k_idx, n, :K*3] = ip
        print(f"  K={K}  {time.time()-t0:.1f}s cumulative", flush=True)
    print(f"Init done in {time.time()-t0:.1f}s")

    # ── GPU: σ-only Adam sweep ─────────────────────────────────────────────
    h_jax         = jnp.array(height_axis.astype(np.float64))
    all_pen_loss  = np.zeros((args.k_max, args.n_sweep), dtype=np.float64)
    all_final_mse = np.zeros((args.k_max, args.n_sweep), dtype=np.float64)
    all_sig_final = [None] * args.k_max

    print("\nRunning JAX GPU sweep:")
    total_t0 = time.time()
    for k_idx, K in enumerate(range(1, args.k_max + 1)):
        t0        = time.time()
        ip_batch  = jnp.array(init_stack[k_idx, :, :K*3])
        pf_batch  = jnp.array(profiles_n)

        sig_clip_val = float(height_axis[-1] - height_axis[0]) / 2.0
        sig_f, mse_f, pen_f = adam_sigma_only_jax_batch(
            ip_batch, pf_batch, h_jax,
            args.steps, K, args.lr, args.lambda_k,
            args.b1, args.b2, args.eps, sig_clip_val
        )
        sig_f.block_until_ready()

        all_sig_final[k_idx] = np.array(sig_f)
        all_final_mse[k_idx] = np.array(mse_f)
        all_pen_loss[k_idx]  = np.array(pen_f)
        print(f"  K={K}  {time.time()-t0:.2f}s  "
              f"median_pen={np.median(all_pen_loss[k_idx]):.6f}", flush=True)

    gpu_time = time.time() - total_t0
    total_steps = args.n_sweep * args.k_max * args.steps
    print(f"\nGPU total: {gpu_time:.1f}s  ({total_steps/1e6:.1f}M Adam steps,  "
          f"{total_steps/gpu_time/1e6:.0f}M steps/s)")

    # ── Model-order selection ──────────────────────────────────────────────
    best_k_idx   = np.argmin(all_pen_loss, axis=0)
    sweep_best_K = best_k_idx + 1
    sweep_mse    = all_final_mse[best_k_idx, np.arange(args.n_sweep)]
    sweep_sigmas = [all_sig_final[best_k_idx[n]][n, :sweep_best_K[n]]
                    for n in range(args.n_sweep)]

    print(f"\nBest-K distribution: "
          f"{ {k: int((sweep_best_K==k).sum()) for k in range(1, args.k_max+1)} }")
    print(f"Median MSE (best K): {np.nanmedian(sweep_mse):.6f}")

    # ── Save raw results ───────────────────────────────────────────────────
    np.save(outdir / "sweep_best_K.npy",   sweep_best_K)
    np.save(outdir / "sweep_mse.npy",      sweep_mse)
    np.save(outdir / "all_pen_loss.npy",   all_pen_loss)
    np.save(outdir / "all_final_mse.npy",  all_final_mse)
    np.save(outdir / "init_stack.npy",     init_stack)
    for k_idx in range(args.k_max):
        np.save(outdir / f"sig_final_K{k_idx+1}.npy", all_sig_final[k_idx])
    print(f"Results saved to {outdir}/")

    # ── Figure 1: distribution + MSE violin + σ histogram ─────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    k_vals, k_counts = np.unique(sweep_best_K, return_counts=True)
    bars = axes[0].bar(k_vals, k_counts / args.n_sweep * 100,
                       color="steelblue", edgecolor="white")
    axes[0].bar_label(bars, fmt="%.1f%%", fontsize=9)
    axes[0].set_xlabel("Selected K"); axes[0].set_ylabel("% of pixels")
    axes[0].set_title(f"Model-order distribution\n(λ={args.lambda_k}, n={args.n_sweep})")

    for k in range(1, args.k_max + 1):
        mask = sweep_best_K == k
        if mask.sum() > 1:
            axes[1].violinplot(sweep_mse[mask], positions=[k],
                               showmedians=True, widths=0.6)
    axes[1].set_xlabel("Selected K"); axes[1].set_ylabel("MSE")
    axes[1].set_title("MSE per model order"); axes[1].set_yscale("log")

    all_sig_flat = np.concatenate(sweep_sigmas)
    axes[2].hist(all_sig_flat, bins=80, color="steelblue", edgecolor="white")
    axes[2].axvline(np.median(all_sig_flat), color="crimson", lw=1.5, ls="--",
                    label=f"median={np.median(all_sig_flat):.2f}m")
    axes[2].set_xlabel("σ (m)"); axes[2].set_ylabel("count")
    axes[2].set_title("Pooled σ distribution"); axes[2].legend()
    fig.suptitle(f"σ-sweep  |  {args.n_sweep} pixels  λ={args.lambda_k}  "
                 f"steps={args.steps}  lr={args.lr}", fontsize=9)
    fig.savefig(outdir / "fig_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: penalised loss violin per K ─────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for k_idx, K in enumerate(range(1, args.k_max + 1)):
        ax2.violinplot(all_pen_loss[k_idx], positions=[K],
                       showmedians=True, widths=0.6)
    ax2.set_xlabel("K"); ax2.set_ylabel("Penalised loss"); ax2.set_yscale("log")
    ax2.set_title(f"Penalised loss per K  (λ={args.lambda_k})")
    fig2.savefig(outdir / "fig_pen_loss_per_K.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Figure 3 & 4: worst / best 30 pixels ──────────────────────────────
    h64 = height_axis.astype(np.float64)
    worst_idxs = np.argsort(sweep_mse)[::-1][:30]
    best_idxs  = np.argsort(sweep_mse)[:30]

    draw_pixel_grid(worst_idxs, profiles_n, h64, sweep_best_K, sweep_mse,
                    init_stack, all_sig_final,
                    f"30 worst-fit pixels  λ={args.lambda_k}  steps={args.steps}",
                    "crimson", outdir / "fig_worst30.png")
    draw_pixel_grid(best_idxs,  profiles_n, h64, sweep_best_K, sweep_mse,
                    init_stack, all_sig_final,
                    f"30 best-fit pixels   λ={args.lambda_k}  steps={args.steps}",
                    "steelblue", outdir / "fig_best30.png")

    print(f"\nAll figures saved to {outdir}/")
    print("Done.")


if __name__ == "__main__":
    main()
