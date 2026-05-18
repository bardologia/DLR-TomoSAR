"""
test_gpu_vs_scipy.py
====================
Validates that the JAX GPU Adam optimiser produces results comparable to
scipy curve_fit on 200 synthetic dual-Gaussian profiles.

Run:
    cd /ste/rnd/User/vice_vi
    conda run -n stetools python DLR-TomoSAR/scripts/test_gpu_vs_scipy.py
"""
import sys, time, warnings
sys.path.insert(0, "DLR-TomoSAR")

import numpy as np
import jax.numpy as jnp
from scipy.optimize import curve_fit

from pipelines.param_extraction_pipeline.gaussian_model import GaussianModel
from pipelines.param_extraction_pipeline.gpu_fitting    import AdamKernel, JaxGaussianModel

# ── Reproducible synthetic data ──────────────────────────────────────────────
np.random.seed(42)
H      = 150
N_PIX  = 200
height = np.linspace(-50.0, 50.0, H, dtype=np.float64)

true_amp1 = np.random.uniform(0.5, 1.5, N_PIX)
true_mu1  = np.random.uniform(-30.0, -5.0, N_PIX)
true_sig1 = np.random.uniform(3.0, 8.0, N_PIX)
true_amp2 = np.random.uniform(0.3, 1.0, N_PIX)
true_mu2  = np.random.uniform(5.0, 30.0, N_PIX)
true_sig2 = np.random.uniform(3.0, 8.0, N_PIX)

def make_profile(a1, m1, s1, a2, m2, s2):
    return (a1 * np.exp(-(height - m1)**2 / (2 * s1**2))
          + a2 * np.exp(-(height - m2)**2 / (2 * s2**2)))

profiles = np.stack([
    make_profile(true_amp1[i], true_mu1[i], true_sig1[i],
                 true_amp2[i], true_mu2[i], true_sig2[i])
    for i in range(N_PIX)
]).astype(np.float32)   # (N_PIX, H)

print(f"Test: {N_PIX} synthetic dual-Gaussian profiles  (H={H})")

# ── SCIPY reference ───────────────────────────────────────────────────────────
model = GaussianModel()
lo    = [0.0, -50.0, 1e-6,  0.0, -50.0, 1e-6]
hi    = [np.inf, 50.0, 100.0,  np.inf, 50.0, 100.0]

scipy_params = np.zeros((N_PIX, 6), dtype=np.float32)
failed       = 0
t0           = time.time()

for i, prof in enumerate(profiles):
    p0 = model.estimate_initial_parameters(height, prof.astype(np.float64), 2)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            fp, _ = curve_fit(
                model.multi_gaussian, height, prof.astype(np.float64),
                p0=p0, bounds=(lo, hi), maxfev=5000,
            )
            scipy_params[i] = fp
        except Exception:
            failed += 1
            scipy_params[i] = p0

scipy_time = time.time() - t0
print(f"\nScipy  : {scipy_time:.2f}s  failed={failed}/{N_PIX}")

# ── JAX GPU ───────────────────────────────────────────────────────────────────
# Use data-adaptive amplitude bounds (mirrors GPUParameterExtractor.run())
# Amplitude upper = 2 × max observed amplitude in the batch
amp_cap = float(profiles.max()) * 2.0
lo_j = jnp.array([0.0, -50.0, 1e-6,  0.0, -50.0, 1e-6], dtype=jnp.float32)
hi_j = jnp.array([amp_cap, 50.0, 100.0, amp_cap, 50.0, 100.0], dtype=jnp.float32)
print(f"  amp_cap = {amp_cap:.3f}  (2x max profile amplitude)")
h_j  = jnp.array(height, dtype=jnp.float32)

kernel   = AdamKernel(JaxGaussianModel.mse_loss)
init_np  = np.zeros((N_PIX, 6), dtype=np.float32)
for i, prof in enumerate(profiles):
    init_np[i] = model.estimate_initial_parameters(height, prof.astype(np.float64), 2)

prof_j = jnp.array(profiles)
init_j = jnp.array(init_np)

kernel(init_j[:4], h_j, prof_j[:4], lo_j, hi_j, n_steps=2)

t0      = time.time()
jax_out = kernel(init_j, h_j, prof_j, lo_j, hi_j, n_steps=1500)
jax_out.block_until_ready()
jax_time = time.time() - t0

jax_params = np.asarray(jax_out, dtype=np.float32)
print(f"JAX GPU: {jax_time:.3f}s  (kernel already compiled)")

# ── Metrics ───────────────────────────────────────────────────────────────────
def r2_batch(params, profiles, h):
    n_g  = params.shape[1] // 3
    pred = np.zeros_like(profiles)
    for g in range(n_g):
        a = params[:, g*3 + 0][:, None]
        m = params[:, g*3 + 1][:, None]
        s = np.maximum(params[:, g*3 + 2], 1e-6)[:, None]
        pred += a * np.exp(np.clip(-(h[None, :] - m)**2 / (2 * s**2), -100.0, 0.0))
    ss_res = np.sum((profiles - pred)**2, axis=1)
    ss_tot = np.sum((profiles - profiles.mean(axis=1, keepdims=True))**2, axis=1)
    return np.where(ss_tot > 1e-20, 1.0 - ss_res / ss_tot, np.nan)

h32      = height.astype(np.float32)
r2_scipy = r2_batch(scipy_params, profiles, h32)
r2_jax   = r2_batch(jax_params,   profiles, h32)

sep = "-" * 55
print(f"\n{sep}")
print(f"{'Metric':<30} {'Scipy':>10} {'JAX GPU':>10}")
print(sep)
print(f"{'Mean R2':<30} {np.nanmean(r2_scipy):>10.4f} {np.nanmean(r2_jax):>10.4f}")
print(f"{'Median R2':<30} {np.nanmedian(r2_scipy):>10.4f} {np.nanmedian(r2_jax):>10.4f}")
print(f"{'R2 > 0.95 (%)':<30} {100*np.mean(r2_scipy>0.95):>10.1f} {100*np.mean(r2_jax>0.95):>10.1f}")
print(f"{'R2 > 0.99 (%)':<30} {100*np.mean(r2_scipy>0.99):>10.1f} {100*np.mean(r2_jax>0.99):>10.1f}")
print(sep)

# Per-parameter MAE (sort Gaussians by mean to align before comparing)
def sort_by_mean(p):
    n_g = p.shape[1] // 3
    out = np.zeros_like(p)
    for i in range(len(p)):
        order = np.argsort([p[i, g*3 + 1] for g in range(n_g)])
        for j, k in enumerate(order):
            out[i, j*3:j*3+3] = p[i, k*3:k*3+3]
    return out

ss   = sort_by_mean(scipy_params)
jj   = sort_by_mean(jax_params)
mae  = np.mean(np.abs(ss - jj), axis=0)
lbls = [item for g in range(2) for item in (f"amp{g+1}", f"mu{g+1}", f"sig{g+1}")]

print("\nMean |scipy - JAX| per parameter:")
for lbl, v in zip(lbls, mae):
    print(f"  {lbl:<8} {v:.4f}")

# ── Recovery from ground truth ────────────────────────────────────────────────
gt = np.column_stack([
    true_amp1, true_mu1, true_sig1,
    true_amp2, true_mu2, true_sig2,
]).astype(np.float32)
gt_s = sort_by_mean(gt)

mae_scipy_gt = np.mean(np.abs(ss - gt_s), axis=0)
mae_jax_gt   = np.mean(np.abs(jj - gt_s), axis=0)

print("\nMean |estimate - ground_truth| per parameter:")
print(f"  {'param':<8} {'scipy':>8} {'jax':>8}")
for lbl, vs, vj in zip(lbls, mae_scipy_gt, mae_jax_gt):
    print(f"  {lbl:<8} {vs:>8.4f} {vj:>8.4f}")

print("\nDone.")
