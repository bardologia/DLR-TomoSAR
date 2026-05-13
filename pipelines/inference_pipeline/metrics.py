from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from skimage.metrics import structural_similarity as ssim

from pipelines.inference_pipeline.predictor import PredictionResult


def _percentiles(x: np.ndarray, qs: Tuple[int, ...] = (1, 5, 25, 50, 75, 95, 99)) -> Dict[str, float]:
    flat = x.reshape(-1)
    out  = {}
    p    = np.percentile(flat, qs)
    for q, v in zip(qs, p):
        out[f"p{q}"] = float(v)
    return out


def _basic_stats(x: np.ndarray) -> Dict[str, float]:
    flat = x.reshape(-1).astype(np.float64, copy=False)
    return {
        "mean"   : float(flat.mean()),
        "std"    : float(flat.std()),
        "median" : float(np.median(flat)),
        "min"    : float(flat.min()),
        "max"    : float(flat.max()),
    }


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    diff = (pred - gt).astype(np.float64)
    mse  = float((diff * diff).mean())
    if mse <= 0.0:
        return float("inf")
    data_range = float(gt.max() - gt.min())
    if data_range <= 0.0:
        return float("nan")
    return 10.0 * np.log10(data_range * data_range / mse)


def _fwhm(x_axis: np.ndarray, profile: np.ndarray) -> float:
    if profile.size < 3 or not np.isfinite(profile).all():
        return float("nan")
    peak_idx = int(np.argmax(profile))
    peak_val = float(profile[peak_idx])
    half     = peak_val / 2.0
    if peak_val <= 0.0:
        return float("nan")

    left = peak_idx
    while left > 0 and profile[left] > half:
        left -= 1
    if left == peak_idx:
        return float("nan")
    x_left = np.interp(half, [profile[left], profile[left + 1]], [x_axis[left], x_axis[left + 1]])

    right = peak_idx
    while right < profile.size - 1 and profile[right] > half:
        right += 1
    if right == peak_idx:
        return float("nan")
    x_right = np.interp(half, [profile[right], profile[right - 1]], [x_axis[right], x_axis[right - 1]])

    return float(x_right - x_left)


def _ssim_2d(pred_slice: np.ndarray, gt_slice: np.ndarray) -> float:
    """SSIM between two 2-D float arrays. Returns NaN if the slice is degenerate."""
    if pred_slice.ndim != 2 or gt_slice.ndim != 2:
        return float("nan")
    data_range = float(gt_slice.max() - gt_slice.min())
    if data_range <= 0.0 or not np.isfinite(pred_slice).all() or not np.isfinite(gt_slice).all():
        return float("nan")
    min_side = min(pred_slice.shape)
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    if win_size < 3:
        return float("nan")
    return float(ssim(gt_slice, pred_slice, data_range=data_range, win_size=win_size))


def compute_slice_ssim(
    pred  : np.ndarray,
    gt    : np.ndarray,
    *,
    elev_indices : Optional[np.ndarray] = None,
    range_indices: Optional[np.ndarray] = None,
    az_indices   : Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute per-slice SSIM for elevation, range, and azimuth cuts.

    Parameters
    ----------
    pred / gt:
        Cubes with shape ``(n_elev, H, W)``.
    elev_indices:
        Elevation-bin indices to slice along the first axis → 2-D plane (H × W).
    range_indices:
        Range-pixel indices to slice along axis-2 (W) → 2-D plane (n_elev × H).
    az_indices:
        Azimuth-pixel indices to slice along axis-1 (H) → 2-D plane (n_elev × W).

    Returns
    -------
    Dict with keys ``ssim_elev_<i>``, ``ssim_range_<i>``, ``ssim_azimuth_<i>``
    and their per-axis mean summaries.
    """
    out: Dict[str, float] = {}

    def _axis_ssim(indices: np.ndarray, axis: int, label: str) -> None:
        vals = []
        for i, idx in enumerate(indices):
            if axis == 0:
                p_sl = pred[idx, :, :]
                g_sl = gt  [idx, :, :]
            elif axis == 1:
                p_sl = pred[:, idx, :]
                g_sl = gt  [:, idx, :]
            else:
                p_sl = pred[:, :, idx]
                g_sl = gt  [:, :, idx]
            v = _ssim_2d(p_sl.astype(np.float64), g_sl.astype(np.float64))
            out[f"ssim_{label}_{i}"] = v
            if np.isfinite(v):
                vals.append(v)
        out[f"ssim_{label}_mean"] = float(np.mean(vals)) if vals else float("nan")

    if elev_indices is not None and len(elev_indices):
        _axis_ssim(elev_indices,  axis=0, label="elev")
    if range_indices is not None and len(range_indices):
        _axis_ssim(range_indices, axis=2, label="range")
    if az_indices is not None and len(az_indices):
        _axis_ssim(az_indices,    axis=1, label="azimuth")

    return out


def compute_global_metrics(
    result        : PredictionResult,
    x_axis        : np.ndarray,
    n_gaussians   : int,
    *,
    elev_indices  : Optional[np.ndarray] = None,
    range_indices : Optional[np.ndarray] = None,
    az_indices    : Optional[np.ndarray] = None,
) -> Dict[str, float]:
    pred = result.pred_curves
    gt   = result.gt_curves
    raw  = result.raw_curves

    # pred vs gt (Gaussian)
    diff_gt      = pred - gt
    mse_gt       = float((diff_gt * diff_gt).mean())
    mae_gt       = float(np.abs(diff_gt).mean())
    gt_mean      = float(gt.mean())
    overall_r2   = 1.0 - float((diff_gt * diff_gt).sum()) / (float(((gt - gt_mean) ** 2).sum()) + 1e-8)

    # pred vs raw
    diff_raw         = pred - raw
    mse_raw          = float((diff_raw * diff_raw).mean())
    mae_raw          = float(np.abs(diff_raw).mean())
    raw_mean         = float(raw.mean())
    overall_r2_raw   = 1.0 - float((diff_raw * diff_raw).sum()) / (float(((raw - raw_mean) ** 2).sum()) + 1e-8)

    x_step = float(x_axis[1] - x_axis[0])

    metrics = {
        "n_pixels"    : int(result.pixel_mse.size),
        "n_elevation" : int(pred.shape[0]),
        "x_axis_min"  : float(x_axis.min()),
        "x_axis_max"  : float(x_axis.max()),
        "x_axis_step" : x_step,

        # pred vs gt
        "curve_mse"   : mse_gt,
        "curve_mae"   : mae_gt,
        "curve_rmse"  : float(np.sqrt(mse_gt)),
        "overall_r2"  : float(overall_r2),
        "psnr_db"     : _psnr(pred, gt),

        # pred vs raw
        "curve_mse_raw"  : mse_raw,
        "curve_mae_raw"  : mae_raw,
        "curve_rmse_raw" : float(np.sqrt(mse_raw)),
        "overall_r2_raw" : float(overall_r2_raw),
        "psnr_db_raw"    : _psnr(pred, raw),

        "gt_mean"    : gt_mean,
        "gt_std"     : float(gt.std()),
        "gt_max"     : float(gt.max()),
        "raw_mean"   : raw_mean,
        "raw_std"    : float(raw.std()),
        "raw_max"    : float(raw.max()),
        "pred_mean"  : float(pred.mean()),
        "pred_std"   : float(pred.std()),
        "pred_max"   : float(pred.max()),
    }

    for tag, arr in (
        ("pixel_mse",            result.pixel_mse),
        ("pixel_mae",            result.pixel_mae),
        ("pixel_r2",             result.pixel_r2),
        ("pixel_cosine",         result.pixel_cosine),
        ("pixel_peak_idx_d",     result.pixel_peak_err_idx.astype(np.float32)),
        ("pixel_mse_raw",        result.pixel_mse_raw),
        ("pixel_mae_raw",        result.pixel_mae_raw),
        ("pixel_r2_raw",         result.pixel_r2_raw),
        ("pixel_cosine_raw",     result.pixel_cosine_raw),
        ("pixel_peak_idx_d_raw", result.pixel_peak_err_idx_raw.astype(np.float32)),
    ):
        s = _basic_stats(arr)
        for k, v in s.items():
            metrics[f"{tag}_{k}"] = v
        for k, v in _percentiles(arr).items():
            metrics[f"{tag}_{k}"] = v

    metrics["pixel_peak_err_units_mean"]       = float(result.pixel_peak_err_idx.mean())                 * x_step
    metrics["pixel_peak_err_units_median"]     = float(np.median(result.pixel_peak_err_idx))             * x_step
    metrics["pixel_peak_err_units_p95"]        = float(np.percentile(result.pixel_peak_err_idx, 95))     * x_step
    metrics["pixel_peak_err_units_mean_raw"]   = float(result.pixel_peak_err_idx_raw.mean())             * x_step
    metrics["pixel_peak_err_units_median_raw"] = float(np.median(result.pixel_peak_err_idx_raw))         * x_step
    metrics["pixel_peak_err_units_p95_raw"]    = float(np.percentile(result.pixel_peak_err_idx_raw, 95)) * x_step

    # SSIM: pred vs gt
    ssim_gt_metrics = compute_slice_ssim(
        pred, gt,
        elev_indices  = elev_indices,
        range_indices = range_indices,
        az_indices    = az_indices,
    )
    metrics.update(ssim_gt_metrics)

    # SSIM: pred vs raw (prefixed with "raw_")
    ssim_raw_metrics = compute_slice_ssim(
        pred, raw,
        elev_indices  = elev_indices,
        range_indices = range_indices,
        az_indices    = az_indices,
    )
    for k, v in ssim_raw_metrics.items():
        metrics[f"raw_{k}"] = v

    return metrics


def select_pixels_by_metric(
    metric_map : np.ndarray,
    *,
    n_best     : int,
    n_worst    : int,
    n_random   : int,
    seed       : int = 0,
) -> Dict[str, np.ndarray]:
    flat = metric_map.reshape(-1)
    H, W = metric_map.shape
    rng  = np.random.default_rng(seed)

    valid_idx  = np.where(np.isfinite(flat))[0]

    order      = valid_idx[np.argsort(flat[valid_idx])]
    best_flat  = order[:n_best]
    worst_flat = order[-n_worst:][::-1]

    pool      = np.setdiff1d(valid_idx, np.concatenate([best_flat, worst_flat]), assume_unique=False)
    n_random  = min(n_random, pool.size)
    rand_flat = rng.choice(pool, size=n_random, replace=False) if n_random > 0 else np.array([], dtype=np.int64)

    def _to_yx(flat_idx: np.ndarray) -> np.ndarray:
        return np.stack([(flat_idx // W).astype(np.int32), (flat_idx % W).astype(np.int32)], axis=1)

    return {
        "best"   : _to_yx(best_flat),
        "worst"  : _to_yx(worst_flat),
        "random" : _to_yx(rand_flat),
    }
