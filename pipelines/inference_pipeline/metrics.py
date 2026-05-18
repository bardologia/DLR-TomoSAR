from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing             import Dict, Optional, Tuple

import numpy                as np
from skimage.metrics import structural_similarity as ssim
from tqdm                   import tqdm
from pipelines.inference_pipeline.predictor import Result


class Metrics:

    _EPS : float = 1e-12

    def __init__(
        self,
        result      : Result,
        x_axis      : np.ndarray,
        n_gaussians : int,
    ) -> None:

        self.result      = result
        self.x_axis      = x_axis
        self.n_gaussians = n_gaussians
        self.x_step      = float(x_axis[1] - x_axis[0])
        self.num_workers = 40

    @staticmethod
    def _percentiles(x: np.ndarray, qs: Tuple[int, ...] = (1, 5, 25, 50, 75, 95, 99)) -> Dict[str, float]:

        flat = x.reshape(-1)
        out  = {}
        p    = np.percentile(flat, qs)
        for q, v in zip(qs, p):
            out[f"p{q}"] = float(v)
        return out

    @staticmethod
    def _basic_stats(x: np.ndarray) -> Dict[str, float]:

        flat = x.reshape(-1).astype(np.float64, copy=False)
        return {
            "mean"   : float(flat.mean()),
            "std"    : float(flat.std()),
            "median" : float(np.median(flat)),
            "min"    : float(flat.min()),
            "max"    : float(flat.max()),
        }

    @staticmethod
    def _psnr(pred: np.ndarray, ref: np.ndarray) -> float:

        diff       = (pred - ref).astype(np.float64)
        mse        = float((diff * diff).mean())
        if mse <= 0.0:
            return float("inf")
        data_range = float(ref.max() - ref.min())
        if data_range <= 0.0:
            return float("nan")
        return 10.0 * np.log10(data_range * data_range / mse)

    @staticmethod
    def _ssim_2d(pred_slice: np.ndarray, ref_slice: np.ndarray) -> float:

        if pred_slice.ndim != 2 or ref_slice.ndim != 2:
            return float("nan")
        data_range = float(ref_slice.max() - ref_slice.min())
        if data_range <= 0.0 or not np.isfinite(pred_slice).all() or not np.isfinite(ref_slice).all():
            return float("nan")
        min_side = min(pred_slice.shape)
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        if win_size < 3:
            return float("nan")
        return float(ssim(ref_slice, pred_slice, data_range=data_range, win_size=win_size))

    def _slice_ssim(
        self,
        pred          : np.ndarray,
        ref           : np.ndarray,
        elev_indices  : Optional[np.ndarray],
        range_indices : Optional[np.ndarray],
        az_indices    : Optional[np.ndarray],
        prefix        : str,
    ) -> Dict[str, float]:

        out: Dict[str, float] = {}

        tasks: list[tuple[str, np.ndarray, np.ndarray]] = []

        def _enqueue(indices: np.ndarray, axis: int, label: str) -> None:
            for i, idx in enumerate(indices):
                if axis == 0:
                    p_sl, r_sl = pred[idx, :, :], ref[idx, :, :]
                elif axis == 1:
                    p_sl, r_sl = pred[:, idx, :], ref[:, idx, :]
                else:
                    p_sl, r_sl = pred[:, :, idx], ref[:, :, idx]
                tasks.append((f"ssim_{prefix}_{label}_{i}", label, p_sl.astype(np.float64), r_sl.astype(np.float64)))

        if elev_indices  is not None and len(elev_indices):
            _enqueue(elev_indices,  axis=0, label="elev")
        if range_indices is not None and len(range_indices):
            _enqueue(range_indices, axis=2, label="range")
        if az_indices    is not None and len(az_indices):
            _enqueue(az_indices,    axis=1, label="azimuth")

        if not tasks:
            return out

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {pool.submit(self._ssim_2d, p, r): key for key, _label, p, r in tasks}
            with tqdm(total=len(futures), desc=f"SSIM ({prefix})", unit="slice", leave=False) as pbar:
                for fut in as_completed(futures):
                    key = futures[fut]
                    out[key] = fut.result()
                    pbar.update(1)

        labels_seen = {t[1] for t in tasks}
        for label in labels_seen:
            vals = [v for (key, lbl, _, _) in tasks if lbl == label and np.isfinite(v := out.get(key, float("nan")))]
            out[f"ssim_{prefix}_{label}_mean"] = float(np.mean(vals)) if vals else float("nan")

        return out

    def _elev_metrics(self, pred: np.ndarray, gt: np.ndarray, raw: np.ndarray) -> Dict[str, np.ndarray]:
        eps = self._EPS

        P = pred.reshape(pred.shape[0], -1).astype(np.float64)
        G = gt  .reshape(gt  .shape[0], -1).astype(np.float64)
        R = raw .reshape(raw .shape[0], -1).astype(np.float64)

        diff_g = P - G
        diff_r = P - R

        mae_gt  = np.abs(diff_g).mean(axis=1)
        rmse_gt = np.sqrt((diff_g ** 2).mean(axis=1))
        g_var   = ((G - G.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) + eps
        r2_gt   = 1.0 - (diff_g ** 2).sum(axis=1) / g_var

        mae_raw  = np.abs(diff_r).mean(axis=1)
        rmse_raw = np.sqrt((diff_r ** 2).mean(axis=1))
        r_var    = ((R - R.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) + eps
        r2_raw   = 1.0 - (diff_r ** 2).sum(axis=1) / r_var

        gt_prob   = G / G.sum(axis=0, keepdims=True).clip(eps, None)
        pred_prob = P / P.sum(axis=0, keepdims=True).clip(eps, None)
        raw_prob  = R / R.sum(axis=0, keepdims=True).clip(eps, None)

        log_pp = np.log(pred_prob.clip(eps, None))
        ce_gt  = -(gt_prob  * log_pp).mean(axis=1)
        ce_raw = -(raw_prob * log_pp).mean(axis=1)

        return {
            "elev_mae_gt"   : mae_gt,
            "elev_rmse_gt"  : rmse_gt,
            "elev_r2_gt"    : r2_gt,
            "elev_ce_gt"    : ce_gt,
            "elev_mae_raw"  : mae_raw,
            "elev_rmse_raw" : rmse_raw,
            "elev_r2_raw"   : r2_raw,
            "elev_ce_raw"   : ce_raw,
        }

    def compute(
        self,
        *,
        elev_indices  : Optional[np.ndarray] = None,
        range_indices : Optional[np.ndarray] = None,
        az_indices    : Optional[np.ndarray] = None,
    ) -> Dict[str, float]:

        pred = self.result.pred_curves
        gt   = self.result.gt_curves
        raw  = self.result.raw_curves

        diff_gt        = pred - gt
        mse_gt         = float((diff_gt * diff_gt).mean())
        mae_gt         = float(np.abs(diff_gt).mean())
        gt_mean        = float(gt.mean())
        overall_r2_gt  = 1.0 - float((diff_gt * diff_gt).sum()) / (float(((gt  - gt_mean)  ** 2).sum()) + self._EPS)

        diff_raw       = pred - raw
        mse_raw        = float((diff_raw * diff_raw).mean())
        mae_raw        = float(np.abs(diff_raw).mean())
        raw_mean       = float(raw.mean())
        overall_r2_raw = 1.0 - float((diff_raw * diff_raw).sum()) / (float(((raw - raw_mean) ** 2).sum()) + self._EPS)

        metrics: Dict[str, float] = {
            "n_pixels"       : int(self.result.pixel_mse.size),
            "n_elevation"    : int(pred.shape[0]),
            "x_axis_min"     : float(self.x_axis.min()),
            "x_axis_max"     : float(self.x_axis.max()),
            "x_axis_step"    : self.x_step,

            "curve_mse_gt"   : mse_gt,
            "curve_mae_gt"   : mae_gt,
            "curve_rmse_gt"  : float(np.sqrt(mse_gt)),
            "overall_r2_gt"  : float(overall_r2_gt),
            "psnr_db_gt"     : self._psnr(pred, gt),

            "curve_mse_raw"  : mse_raw,
            "curve_mae_raw"  : mae_raw,
            "curve_rmse_raw" : float(np.sqrt(mse_raw)),
            "overall_r2_raw" : float(overall_r2_raw),
            "psnr_db_raw"    : self._psnr(pred, raw),

            "gt_mean"        : gt_mean,
            "gt_std"         : float(gt.std()),
            "gt_max"         : float(gt.max()),
            "raw_mean"       : raw_mean,
            "raw_std"        : float(raw.std()),
            "raw_max"        : float(raw.max()),
            "pred_mean"      : float(pred.mean()),
            "pred_std"       : float(pred.std()),
            "pred_max"       : float(pred.max()),
        }

        for tag, arr in (
            ("pixel_mse_gt",         self.result.pixel_mse),
            ("pixel_mae_gt",         self.result.pixel_mae),
            ("pixel_r2_gt",          self.result.pixel_r2),
            ("pixel_cosine_gt",      self.result.pixel_cosine),
            ("pixel_peak_idx_d_gt",  self.result.pixel_peak_err_idx.astype(np.float32)),
            ("pixel_mse_raw",        self.result.pixel_mse_raw),
            ("pixel_mae_raw",        self.result.pixel_mae_raw),
            ("pixel_r2_raw",         self.result.pixel_r2_raw),
            ("pixel_cosine_raw",     self.result.pixel_cosine_raw),
            ("pixel_peak_idx_d_raw", self.result.pixel_peak_err_idx_raw.astype(np.float32)),
        ):
            for k, v in self._basic_stats(arr).items():
                metrics[f"{tag}_{k}"] = v
            for k, v in self._percentiles(arr).items():
                metrics[f"{tag}_{k}"] = v

        metrics["pixel_peak_err_units_mean_gt"]    = float(self.result.pixel_peak_err_idx.mean())                  * self.x_step
        metrics["pixel_peak_err_units_median_gt"]  = float(np.median(self.result.pixel_peak_err_idx))              * self.x_step
        metrics["pixel_peak_err_units_p95_gt"]     = float(np.percentile(self.result.pixel_peak_err_idx, 95))      * self.x_step
        metrics["pixel_peak_err_units_mean_raw"]   = float(self.result.pixel_peak_err_idx_raw.mean())              * self.x_step
        metrics["pixel_peak_err_units_median_raw"] = float(np.median(self.result.pixel_peak_err_idx_raw))          * self.x_step
        metrics["pixel_peak_err_units_p95_raw"]    = float(np.percentile(self.result.pixel_peak_err_idx_raw, 95))  * self.x_step

        for k, v in self._slice_ssim(pred, gt,  elev_indices, range_indices, az_indices, prefix="gt" ).items():
            metrics[k] = v
        for k, v in self._slice_ssim(pred, raw, elev_indices, range_indices, az_indices, prefix="raw").items():
            metrics[k] = v

        for metric_name, arr in self._elev_metrics(pred, gt, raw).items():
            for i, v in enumerate(arr):
                metrics[f"{metric_name}_{i}"] = float(v)
            metrics[f"{metric_name}_mean"] = float(np.nanmean(arr))

        return metrics

    @staticmethod
    def select_pixels(
        metric_map : np.ndarray,
        *,
        n_best     : int,
        n_worst    : int,
        n_random   : int,
        seed       : int = 0,
    ) -> Dict[str, np.ndarray]:

        flat      = metric_map.reshape(-1)
        H, W      = metric_map.shape
        rng       = np.random.default_rng(seed)
        valid_idx = np.where(np.isfinite(flat))[0]
        order     = valid_idx[np.argsort(flat[valid_idx])]

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


