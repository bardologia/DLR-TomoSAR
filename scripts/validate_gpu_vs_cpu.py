"""
validate_gpu_vs_cpu.py
======================
Extensive validation of the JAX GPU fitting backend against the scipy CPU
baseline on the toy dataset.

Sections:
  1. Run both backends and record timing
  2. Per-parameter absolute and relative error statistics
  3. Per-pixel R² for both backends (computed identically via GaussianModel)
  4. Outlier analysis — pixels where |Δparam| exceeds a threshold
  5. Scipy failure audit — CPU pixels with unreasonable parameter values
  6. Per-pixel parameter correlation (slope, R² of linear fit CPU vs GPU)
  7. Summary pass/fail verdict

Run:
    cd /ste/rnd/User/vice_vi
    conda run -n stetools python DLR-TomoSAR/scripts/validate_gpu_vs_cpu.py
"""
from __future__ import annotations

import sys, time
sys.path.insert(0, "DLR-TomoSAR")

import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

from configuration.param_extraction_config   import FitSettings, FitMode
from pipelines.param_extraction_pipeline.fitting       import ParameterExtractor, FittingMethods
from pipelines.param_extraction_pipeline.gaussian_model import GaussianModel
from tools.logger import Logger

TOY_PATH     = Path("Dataset/toy")
TOMO_NAME    = "tomofull_1000a1050a500a550_dtmf_Xtomo_id2X.npy"
HEIGHT_RANGE = (-20.0, 80.0)
N_GAUSSIANS  = 3
THRESHOLD    = 0.25
TRUNCATION   = 170
WORKERS      = 8
GPU_STEPS    = 2000
GPU_BATCH    = 256

SEP    = "─" * 72
SUBSEP = "  " + "·" * 68


def _r2_per_pixel(
    params_array : np.ndarray,
    tomo_raw     : np.ndarray,
    height_axis  : np.ndarray,
    threshold    : float,
    truncation   : int,
) -> np.ndarray:
    model    = GaussianModel()
    H, Az, R = tomo_raw.shape
    r2_map   = np.full((Az, R), np.nan, dtype=np.float64)

    for ri in range(R):
        for ai in range(Az):
            raw  = np.abs(tomo_raw[:, ai, ri]).astype(np.float64)
            mx   = raw.max()
            if mx < 1e-7:
                continue
            prof = np.where(raw > mx * threshold, raw, 0.0)
            prof[truncation:] = 0.0

            p    = params_array[:, ai, ri].astype(np.float64)
            pred = model.multi_gaussian(height_axis, *p)
            r2_val = FittingMethods._compute_r2(prof, pred)
            if np.isfinite(r2_val):
                r2_map[ai, ri] = r2_val

    return r2_map


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    xm, ym = x[mask] - x[mask].mean(), y[mask] - y[mask].mean()
    denom  = np.sqrt((xm**2).sum() * (ym**2).sum())
    return float(np.dot(xm, ym) / denom) if denom > 0 else float("nan")


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    xm = x[mask]
    ym = y[mask]
    denom = ((xm - xm.mean())**2).sum()
    if denom < 1e-30:
        return float("nan")
    return float(((xm - xm.mean()) * (ym - ym.mean())).sum() / denom)


def _print_r2_histogram(r2_flat: np.ndarray, label: str) -> None:
    finite = r2_flat[np.isfinite(r2_flat)]
    total  = len(finite)
    if total == 0:
        print(f"  {label}: no finite R² values")
        return
    bins   = [(-np.inf, 0.0), (0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.01)]
    labels = ["<0", "0–0.5", "0.5–0.8", "0.8–0.9", "0.9–0.95", "0.95–0.99", "≥0.99"]
    counts = [int(((finite >= lo) & (finite < hi)).sum()) for lo, hi in bins]
    parts  = "  ".join(f"{lbl}:{c/total*100:5.1f}%" for lbl, c in zip(labels, counts))
    print(f"  {label}:  mean={finite.mean():.4f}  median={np.median(finite):.4f}  |  {parts}")


def main() -> None:
    fit_settings = FitSettings(
        number_of_gaussians = N_GAUSSIANS,
        max_fit_iterations  = 5000,
        fit_config          = FitMode.Adaptive(
            threshold_factor = THRESHOLD,
            truncation_index = TRUNCATION,
        ),
    )
    logger    = Logger(name="validate")
    tomo_path = TOY_PATH / "data" / TOMO_NAME
    tomo      = np.load(str(tomo_path), mmap_mode="r", allow_pickle=False)
    H, Az, R  = tomo.shape
    height    = np.linspace(HEIGHT_RANGE[0], HEIGHT_RANGE[1], H, dtype=np.float64)
    n_params  = 3 * N_GAUSSIANS
    pnames    = [f"{t}{g}" for g in range(1, N_GAUSSIANS + 1) for t in ("amp", "mu", "sig")]
    pnames    = [f"amp{g}" for g in range(1, N_GAUSSIANS+1)]
    pnames    = []
    for g in range(1, N_GAUSSIANS + 1):
        pnames += [f"amp{g}", f"mu{g}", f"sig{g}"]

    print(SEP)
    print(f"  Tomogram  : {tomo.shape}  dtype={tomo.dtype}")
    print(f"  Height    : {HEIGHT_RANGE}  span={HEIGHT_RANGE[1]-HEIGHT_RANGE[0]:.0f} m")
    print(f"  Gaussians : {N_GAUSSIANS}  →  {n_params} params/pixel")
    print(f"  Pixels    : {Az * R:,}   (Az={Az} × R={R})")
    print(SEP)

    # ── 1. Run both backends ──────────────────────────────────────────────────
    print("\n[1] Running scipy CPU …")
    t0  = time.perf_counter()
    cpu = ParameterExtractor(fit_settings, WORKERS, logger, use_gpu=False)
    cp  = cpu.run(tomo_path, HEIGHT_RANGE)
    cpu_time = time.perf_counter() - t0
    print(f"    done in {cpu_time:.1f}s\n")

    print("[1] Running JAX GPU …")
    t0  = time.perf_counter()
    gpu = ParameterExtractor(fit_settings, 1, logger, use_gpu=True,
                             gpu_batch_size=GPU_BATCH, adam_steps=GPU_STEPS)
    gp  = gpu.run(tomo_path, HEIGHT_RANGE)
    gpu_time = time.perf_counter() - t0
    print(f"    done in {gpu_time:.1f}s\n")

    assert cp.shape == gp.shape, f"Shape mismatch: {cp.shape} vs {gp.shape}"
    speedup = cpu_time / gpu_time

    # per-pixel profile max — used to filter noise-only pixels in section 6
    tomo_abs       = np.abs(tomo)                        # (H, Az, R)
    profile_max_2d = tomo_abs.max(axis=0)               # (Az, R)
    global_max     = float(profile_max_2d.max())
    SIGNAL_THRESH  = 0.05 * global_max                  # 5% of global max
    signal_mask    = profile_max_2d > SIGNAL_THRESH      # (Az, R)

    # ── 2. Per-parameter error statistics ────────────────────────────────────
    print(SEP)
    print("  [2] Per-parameter absolute and relative error  (CPU vs GPU)")
    print(SUBSEP)
    height_span = HEIGHT_RANGE[1] - HEIGHT_RANGE[0]
    ref_scales  = {}
    for g in range(N_GAUSSIANS):
        ref_scales[f"amp{g+1}"] = float(np.abs(tomo).max())
        ref_scales[f"mu{g+1}"]  = height_span
        ref_scales[f"sig{g+1}"] = height_span

    print(f"  {'param':<8}  {'mean|Δ|':>10}  {'p50|Δ|':>10}  {'p95|Δ|':>10}  {'p99|Δ|':>10}  {'max|Δ|':>10}  {'rel(p50)%':>10}")
    print("  " + "-" * 68)

    issues = []
    for i, name in enumerate(pnames):
        d    = np.abs(cp[i].astype(np.float64) - gp[i].astype(np.float64)).ravel()
        ref  = ref_scales[name]
        p50  = float(np.percentile(d, 50))
        p95  = float(np.percentile(d, 95))
        p99  = float(np.percentile(d, 99))
        mx   = float(d.max())
        rel  = 100.0 * p50 / ref if ref > 0 else float("nan")
        print(f"  {name:<8}  {d.mean():>10.4f}  {p50:>10.4f}  {p95:>10.4f}  {p99:>10.4f}  {mx:>10.4f}  {rel:>9.3f}%")
        if rel > 5.0:
            issues.append(f"    ⚠  {name}: p50 relative error {rel:.1f}% > 5% of height span")

    # ── 3. Per-pixel R² ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [3] Per-pixel R² — computed identically for both backends")
    print(SUBSEP)
    print("  Computing CPU R² map …", end=" ", flush=True)
    r2_cpu = _r2_per_pixel(cp, tomo, height, THRESHOLD, TRUNCATION)
    print("done")
    print("  Computing GPU R² map …", end=" ", flush=True)
    r2_gpu = _r2_per_pixel(gp, tomo, height, THRESHOLD, TRUNCATION)
    print("done\n")

    _print_r2_histogram(r2_cpu.ravel(), "CPU")
    _print_r2_histogram(r2_gpu.ravel(), "GPU")

    fin_cpu = r2_cpu[np.isfinite(r2_cpu)]
    fin_gpu = r2_gpu[np.isfinite(r2_gpu)]
    r2_gap  = float(np.nanmean(r2_cpu) - np.nanmean(r2_gpu))
    print(f"\n  Mean R² gap (CPU − GPU) : {r2_gap:+.4f}")

    pct_close_r2 = float(np.nanmean(np.abs(r2_cpu - r2_gpu) < 0.05)) * 100
    print(f"  Pixels with |R²_cpu − R²_gpu| < 0.05 : {pct_close_r2:.1f}%")

    # ── 4. Outlier analysis ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [4] Outlier analysis — pixels where backends diverge most")
    print(SUBSEP)

    r2_diff    = np.abs(r2_cpu - r2_gpu)
    worst_mask = np.isfinite(r2_diff)
    if worst_mask.any():
        worst_flat  = np.where(worst_mask, r2_diff, -1.0)
        worst_idx   = np.unravel_index(np.argsort(worst_flat.ravel())[-5:][::-1], (Az, R))
        model       = GaussianModel()

        print(f"  {'rank':<5}  {'(az,r)':<12}  {'R²_cpu':>8}  {'R²_gpu':>8}  {'|ΔR²|':>8}  {'profile_max':>12}")
        print("  " + "-" * 60)
        for k in range(len(worst_idx[0])):
            ai, ri  = worst_idx[0][k], worst_idx[1][k]
            raw     = np.abs(tomo[:, ai, ri]).astype(np.float64)
            mx      = raw.max()
            print(f"  {k+1:<5}  ({ai:>3},{ri:>3})       {r2_cpu[ai,ri]:>8.4f}  {r2_gpu[ai,ri]:>8.4f}  {r2_diff[ai,ri]:>8.4f}  {mx:>12.6f}")

    # ── 5. Scipy failure audit ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [5] Scipy failure audit — CPU pixels with unreasonable amplitudes")
    print(SUBSEP)

    amp_indices = list(range(0, n_params, 3))
    amp_cpu_all = cp[amp_indices].ravel()
    amp_gpu_all = gp[amp_indices].ravel()
    amp_max_raw = float(np.abs(tomo).max())
    blow_thresh = amp_max_raw * 100.0

    cpu_blown = int((amp_cpu_all > blow_thresh).sum())
    gpu_blown = int((amp_gpu_all > blow_thresh).sum())
    print(f"  Profile abs max in tomogram : {amp_max_raw:.6f}")
    print(f"  'blown' threshold (100×max) : {blow_thresh:.4f}")
    print(f"  CPU pixels with amp > thresh: {cpu_blown} / {amp_cpu_all.size}")
    print(f"  GPU pixels with amp > thresh: {gpu_blown} / {amp_gpu_all.size}")

    sig_indices = list(range(2, n_params, 3))
    sig_cpu_all = cp[sig_indices].ravel()
    sig_gpu_all = gp[sig_indices].ravel()
    cpu_neg_sig = int((sig_cpu_all < 0).sum())
    gpu_neg_sig = int((sig_gpu_all < 0).sum())
    print(f"  CPU pixels with sigma < 0   : {cpu_neg_sig}")
    print(f"  GPU pixels with sigma < 0   : {gpu_neg_sig}")

    mu_lo, mu_hi = HEIGHT_RANGE[0] - 5.0, HEIGHT_RANGE[1] + 5.0
    mu_indices   = list(range(1, n_params, 3))
    cpu_oob_mu   = int(((cp[mu_indices] < mu_lo) | (cp[mu_indices] > mu_hi)).sum())
    gpu_oob_mu   = int(((gp[mu_indices] < mu_lo) | (gp[mu_indices] > mu_hi)).sum())
    print(f"  CPU pixels with mu outside [{mu_lo:.0f},{mu_hi:.0f}]: {cpu_oob_mu}")
    print(f"  GPU pixels with mu outside [{mu_lo:.0f},{mu_hi:.0f}]: {gpu_oob_mu}")

    # ── 6. Per-pixel parameter correlation ───────────────────────────────────
    print(f"\n{SEP}")
    print("  [6] CPU vs GPU correlation per parameter")
    print("  Note: Spearman ρ is outlier-robust; Pearson r is distorted by the")
    print("        ~5% of pixels with genuine multi-solution ambiguity.")
    print(f"  Signal threshold : {SIGNAL_THRESH:.5f} ({100*SIGNAL_THRESH/global_max:.0f}% of global max)")
    n_signal = int(signal_mask.sum())
    print(f"  Signal pixels    : {n_signal} / {Az*R} ({100*n_signal/(Az*R):.1f}%)")
    print(SUBSEP)
    height_span_corr = HEIGHT_RANGE[1] - HEIGHT_RANGE[0]
    sig_flat = signal_mask.ravel()
    print(f"  {'param':<8}  {'Spearman ρ':>11}  {'Pearson r':>10}  {'slope':>8}  {'agree@1%span':>13}  note")
    print("  " + "-" * 72)

    for i, name in enumerate(pnames):
        x    = cp[i].ravel().astype(np.float64)
        y    = gp[i].ravel().astype(np.float64)
        mask = sig_flat & np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1e6) & (np.abs(y) < 1e6)
        rho  = float(spearmanr(x[mask], y[mask]).statistic) if mask.sum() > 2 else float('nan')
        r    = _pearson(x[mask], y[mask])
        s    = _linear_slope(x[mask], y[mask])
        band = height_span_corr * 0.01  # 1% of height span
        pct_agree = float(100.0 * (np.abs(x[mask] - y[mask]) < band).sum() / mask.sum()) if mask.sum() > 0 else float('nan')
        note = ""
        if np.isfinite(rho) and rho < 0.90:
            note = "⚠  low ρ"
            issues.append(f"    ⚠  {name}: CPU–GPU Spearman ρ={rho:.3f} < 0.90 (signal pixels only)")
        print(f"  {name:<8}  {rho:>11.4f}  {r:>10.4f}  {s:>8.4f}  {pct_agree:>12.1f}%  {note}")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [7] Summary")
    print(SUBSEP)
    print(f"  CPU time          : {cpu_time:.1f}s")
    print(f"  GPU time          : {gpu_time:.1f}s")
    print(f"  Speedup           : {speedup:.1f}×")
    print(f"  CPU mean R²       : {float(np.nanmean(r2_cpu)):.4f}")
    print(f"  GPU mean R²       : {float(np.nanmean(r2_gpu)):.4f}")
    print(f"  Mean R² gap       : {r2_gap:+.4f}")
    print(f"  |R²| within 0.05  : {pct_close_r2:.1f}% of pixels")
    print(f"  CPU blown amps    : {cpu_blown}")
    print(f"  GPU blown amps    : {gpu_blown}")

    if issues:
        print(f"\n  Warnings ({len(issues)}):")
        for w in issues:
            print(w)
        print(f"\n  Verdict: ⚠  REVIEW WARNINGS before production use")
    else:
        print(f"\n  Verdict: GPU backend results are consistent with CPU baseline")

    print(SEP)


if __name__ == "__main__":
    main()
