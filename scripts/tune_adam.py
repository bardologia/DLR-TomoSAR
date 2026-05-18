from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configuration.param_extraction_config import ExtractionConfig, FitMode, FitSettings
from pipelines.param_extraction_pipeline.gpu_fitting import (
    AdamKernel,
    GPUParameterExtractor,
    JaxGaussianModel,
)
from scipy.ndimage import uniform_filter1d
from tools.logger import Logger

import jax
import jax.numpy as jnp

TOY_DATASET = Path("/ste/rnd/User/vice_vi/Dataset/toy")

N_SAMPLE_PIXELS = 8192
SEED            = 42

SEARCH_SPACE = {
    "lr"      : [1e-3, 3e-3, 5e-3, 8e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1],
    "b1"      : [0.80, 0.85, 0.90, 0.92, 0.95],
    "b2"      : [0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999],
    "n_steps" : [500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000],
}

HEIGHT_RANGE     = (-20.0, 80.0)
N_GAUSSIANS      = 2
THRESHOLD_FACTOR = 0.25
TRUNCATION_INDEX = 170
GPU_DEVICE_IDS   = [0]


def _vectorised_initial_guess(profiles: np.ndarray, height_axis: np.ndarray, n_gaussians: int) -> np.ndarray:
    N, H        = profiles.shape
    sigma_guess = float((height_axis[-1] - height_axis[0]) / (4.0 * n_gaussians))
    working     = uniform_filter1d(profiles.astype(np.float32), size=5, mode="nearest", axis=1).copy()
    params      = np.zeros((N, 3 * n_gaussians), dtype=np.float32)
    for g in range(n_gaussians):
        peak_idx            = np.argmax(working, axis=1)
        peak_amp            = working[np.arange(N), peak_idx]
        peak_mu             = height_axis[peak_idx].astype(np.float32)
        params[:, g*3 + 0]  = np.maximum(peak_amp, 1e-10)
        params[:, g*3 + 1]  = peak_mu
        params[:, g*3 + 2]  = sigma_guess
        dist                = np.abs(height_axis[None, :] - peak_mu[:, None])
        working[dist < 2.0 * sigma_guess] = 0.0
    return params


def _load_toy(toy_dir: Path, n_sample: int, rng: np.random.Generator):
    data_dir  = toy_dir / "data"
    tomo_path = next(data_dir.glob("tomofull_*_1_*.npy"), None)
    if tomo_path is None:
        tomo_path = next(data_dir.glob("*.npy"), None)
    if tomo_path is None:
        raise FileNotFoundError(f"No tomogram found in {data_dir}")

    if tomo_path.suffix == ".npy":
        tomo = np.load(tomo_path)
    else:
        import h5py
        with h5py.File(tomo_path, "r") as f:
            key  = list(f.keys())[0]
            tomo = f[key][:]

    if np.iscomplexobj(tomo):
        tomo = np.abs(tomo).astype(np.float32)
    else:
        tomo = tomo.astype(np.float32)

    H, R, Az = tomo.shape
    flat      = tomo.reshape(H, R * Az).T
    idx       = rng.choice(R * Az, size=min(n_sample, R * Az), replace=False)
    return flat[idx], H


def _build_height_axis(H: int) -> np.ndarray:
    return np.linspace(HEIGHT_RANGE[0], HEIGHT_RANGE[1], H, dtype=np.float32)


def _prep_profiles(profiles_raw: np.ndarray, height_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    threshold  = THRESHOLD_FACTOR * profiles_raw.max(axis=1, keepdims=True)
    active     = profiles_raw.max(axis=1) > 1e-9
    profiles   = profiles_raw.copy()
    profiles[profiles < threshold] = 0.0
    scale      = profiles.max(axis=1, keepdims=True) + 1e-12
    norm       = profiles / scale
    return norm[active], scale[active], active


def _make_bounds(height_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h_min, h_max = float(height_axis[0]), float(height_axis[-1])
    h_range      = h_max - h_min
    amp_lo, amp_hi   = 0.0,     1.0
    mu_lo,  mu_hi    = h_min,   h_max
    sig_lo, sig_hi   = 0.005,   h_range * 0.5
    per_g = [amp_lo, mu_lo, sig_lo]
    up_g  = [amp_hi, mu_hi, sig_hi]
    lower = jnp.array(per_g * N_GAUSSIANS, dtype=jnp.float32)
    upper = jnp.array(up_g  * N_GAUSSIANS, dtype=jnp.float32)
    return lower, upper


def _run_trial(kernel, profiles_j, init_j, height_ax_j, lower, upper, n_steps, lr, b1, b2) -> tuple[float, float]:
    t0     = time.perf_counter()
    fitted = kernel(init_j, height_ax_j, profiles_j, lower, upper, n_steps=n_steps, lr=lr, b1=b1, b2=b2)
    fitted.block_until_ready()
    elapsed = time.perf_counter() - t0

    fitted_np   = np.array(fitted)
    height_np   = np.array(height_ax_j)
    profiles_np = np.array(profiles_j)

    N = fitted_np.shape[0]
    r2_vals = []
    for i in range(N):
        pred    = np.zeros(len(height_np), dtype=np.float64)
        params  = fitted_np[i].astype(np.float64)
        for g in range(N_GAUSSIANS):
            a, mu, sig = params[3*g], params[3*g+1], params[3*g+2]
            sig        = max(sig, 1e-6)
            pred      += a * np.exp(-((height_np - mu) ** 2) / (2 * sig ** 2))
        obs      = profiles_np[i].astype(np.float64)
        ss_res   = np.sum((obs - pred) ** 2)
        ss_tot   = np.sum((obs - obs.mean()) ** 2)
        if ss_tot > 1e-20:
            r2_vals.append(1.0 - ss_res / (ss_tot + 1e-10))

    mean_r2 = float(np.mean(r2_vals)) if r2_vals else float("nan")
    return mean_r2, elapsed


def main():
    rng     = np.random.default_rng(SEED)
    logger  = Logger(log_dir="/tmp", name="tune_adam")

    print(f"Loading toy dataset from {TOY_DATASET} …")
    profiles_raw, H = _load_toy(TOY_DATASET, N_SAMPLE_PIXELS, rng)
    height_axis     = _build_height_axis(H)

    norm_profiles, scale, active = _prep_profiles(profiles_raw, height_axis)
    print(f"Active pixels: {active.sum()} / {len(profiles_raw)}")

    init_params = _vectorised_initial_guess(
        norm_profiles[:, :TRUNCATION_INDEX],
        height_axis[:TRUNCATION_INDEX],
        N_GAUSSIANS,
    )

    lower, upper = _make_bounds(height_axis)

    device    = jax.devices("gpu")[GPU_DEVICE_IDS[0]]
    height_j  = jax.device_put(jnp.array(height_axis), device)
    prof_j    = jax.device_put(jnp.array(norm_profiles, dtype=jnp.float32), device)
    init_j    = jax.device_put(jnp.array(init_params,  dtype=jnp.float32), device)

    loss_fn = JaxGaussianModel.mse_loss
    kernel  = AdamKernel(loss_fn)

    print("Warming up JIT …")
    _run_trial(kernel, prof_j[:4], init_j[:4], height_j, lower, upper, n_steps=2, lr=1e-2, b1=0.9, b2=0.999)
    print("Warm-up done.\n")

    keys   = list(SEARCH_SPACE.keys())
    combos = list(itertools.product(*[SEARCH_SPACE[k] for k in keys]))
    print(f"Running {len(combos)} trials …\n")

    results = []
    for combo in combos:
        params       = dict(zip(keys, combo))
        mean_r2, sec = _run_trial(
            kernel, prof_j, init_j, height_j, lower, upper,
            n_steps = params["n_steps"],
            lr      = params["lr"],
            b1      = params["b1"],
            b2      = params["b2"],
        )
        results.append({**params, "r2": mean_r2, "time_s": sec})
        print(f"  lr={params['lr']:.0e}  b1={params['b1']}  b2={params['b2']}  steps={params['n_steps']:4d}"
              f"  →  R²={mean_r2:.4f}  t={sec:.2f}s")

    results.sort(key=lambda x: -x["r2"])
    print("\n=== Top 10 by R² ===")
    for r in results[:10]:
        print(f"  lr={r['lr']:.0e}  b1={r['b1']}  b2={r['b2']}  steps={r['n_steps']:4d}"
              f"  →  R²={r['r2']:.4f}  t={r['time_s']:.2f}s")

    results_by_speed = sorted(results, key=lambda x: x["time_s"])
    print("\n=== Pareto front (R² vs speed) ===")
    best_r2  = -1.0
    pareto   = []
    for r in results_by_speed:
        if r["r2"] > best_r2:
            best_r2 = r["r2"]
            pareto.append(r)
    for r in pareto:
        print(f"  lr={r['lr']:.0e}  b1={r['b1']}  b2={r['b2']}  steps={r['n_steps']:4d}"
              f"  →  R²={r['r2']:.4f}  t={r['time_s']:.2f}s")

    best = results[0]
    print(f"\n=== Best overall ===")
    print(f"  adam_lr    = {best['lr']}")
    print(f"  adam_b1    = {best['b1']}")
    print(f"  adam_b2    = {best['b2']}")
    print(f"  adam_steps = {best['n_steps']}")
    print(f"  R²         = {best['r2']:.4f}")
    print(f"  time       = {best['time_s']:.2f}s")


if __name__ == "__main__":
    main()
