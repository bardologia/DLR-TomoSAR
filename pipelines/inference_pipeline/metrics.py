from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib            import Path
from typing             import Dict, Optional, Tuple

import numpy                            as np
from skimage.metrics                    import structural_similarity as ssim
from tqdm                               import tqdm
from pipelines.inference_pipeline.types import Result


class Metrics:
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
        self.num_workers = min(os.cpu_count() or 1, 16)

    @staticmethod
    def _ssim_worker(args: tuple) -> tuple[str, float]:
        key, pred_slice, ref_slice = args
        data_range = float(ref_slice.max() - ref_slice.min())
        min_side   = min(pred_slice.shape)
        win_size   = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        value      = float(ssim(ref_slice, pred_slice, data_range=data_range, win_size=win_size))

        return key, value

    @staticmethod
    def write_json(metrics: Dict[str, object], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, default=str)

        return path

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
        flat = x.reshape(-1)

        return {
            "mean"   : float(flat.mean(dtype=np.float64)),
            "std"    : float(flat.std(dtype=np.float64)),
            "median" : float(np.median(flat)),
            "min"    : float(flat.min()),
            "max"    : float(flat.max()),
        }

    @staticmethod
    def _psnr(pred: np.ndarray, ref: np.ndarray) -> float:
        diff       = pred - ref
        mse        = float((diff * diff).mean(dtype=np.float64))
        
        if mse <= 0.0:
            return float("inf")
        
        data_range = float(ref.max() - ref.min())
        if data_range <= 0.0:
            return float("nan")
       
        return 10.0 * np.log10(data_range * data_range / mse)

    @staticmethod
    def select_pixels(
        metric_map : np.ndarray,
        *,
        n_best     : int,
        n_worst    : int,
        n_random   : int,
        seed       : int = 0,
    ) -> Dict[str, np.ndarray]:

        flat       = metric_map.reshape(-1)
        H, W       = metric_map.shape
        rng        = np.random.default_rng(seed)
        valid_idx  = np.where(np.isfinite(flat))[0]
        order      = valid_idx[np.argsort(flat[valid_idx])]

        best_flat  = order[:n_best]
        worst_flat = order[-n_worst:][::-1]

        pool       = np.setdiff1d(valid_idx, np.concatenate([best_flat, worst_flat]), assume_unique=False)
        n_random   = min(n_random, pool.size)
        rand_flat  = rng.choice(pool, size=n_random, replace=False) if n_random > 0 else np.array([], dtype=np.int64)

        def _to_yx(flat_idx: np.ndarray) -> np.ndarray:
            return np.stack([(flat_idx // W).astype(np.int32), (flat_idx % W).astype(np.int32)], axis=1)

        return {
            "best"   : _to_yx(best_flat),
            "worst"  : _to_yx(worst_flat),
            "random" : _to_yx(rand_flat),
        }

    @staticmethod
    def _enqueue_tasks(
        tasks  : list,
        pred   : np.ndarray,
        ref    : np.ndarray,
        prefix : str,
        indices: np.ndarray,
        axis   : int,
        label  : str,
    ) -> None:
       
        for i, idx in enumerate(indices):
            if axis == 0:
                p_sl, r_sl = pred[idx, :, :], ref[idx, :, :]
            elif axis == 1:
                p_sl, r_sl = pred[:, idx, :], ref[:, idx, :]
            else:
                p_sl, r_sl = pred[:, :, idx], ref[:, :, idx]
       
            tasks.append((f"ssim_{prefix}_{label}_{i}", label, p_sl.astype(np.float64), r_sl.astype(np.float64)))

    def _slice_ssim(self, pred : np.ndarray, ref : np.ndarray, elev_indices : Optional[np.ndarray], range_indices : Optional[np.ndarray], az_indices : Optional[np.ndarray], prefix : str) -> Dict[str, float]:
        out   : Dict[str, float]                         = {}
        tasks : list[tuple[str, np.ndarray, np.ndarray]] = []

        self._enqueue_tasks(tasks, pred, ref, prefix, elev_indices,  axis=0, label="elev")
        self._enqueue_tasks(tasks, pred, ref, prefix, range_indices, axis=2, label="range")
        self._enqueue_tasks(tasks, pred, ref, prefix, az_indices,    axis=1, label="azimuth")

        if not tasks:
            return out

        worker_args = [(key, p, r) for key, _label, p, r in tasks]
        n_workers   = min(self.num_workers, len(worker_args))

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(Metrics._ssim_worker, arg): arg[0] for arg in worker_args}
            with tqdm(total=len(futures), desc=f"SSIM ({prefix})", unit="slice", leave=False) as pbar:
                for fut in as_completed(futures):
                    key, value = fut.result()
                    out[key]   = value
                    pbar.update(1)

        labels_seen = {t[1] for t in tasks}
        for label in labels_seen:
            vals = [v for (key, lbl, _, _) in tasks if lbl == label and np.isfinite(v := out.get(key, float("nan")))]
            out[f"ssim_{prefix}_{label}_mean"] = float(np.mean(vals)) if vals else float("nan")

        return out

    def _elev_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, np.ndarray]:
        P = pred.reshape(pred.shape[0], -1)
        G = gt  .reshape(gt  .shape[0], -1)

        diff_g    = P - G
        mae_gt    = np.abs(diff_g).mean(axis=1, dtype=np.float64)
        rmse_gt   = np.sqrt((diff_g ** 2).mean(axis=1, dtype=np.float64))
        g_var     = ((G - G.mean(axis=1, keepdims=True, dtype=np.float64)) ** 2).sum(axis=1, dtype=np.float64) + 1e-12
        r2_gt     = 1.0 - (diff_g ** 2).sum(axis=1, dtype=np.float64) / g_var

        gt_prob   = G / G.sum(axis=0, keepdims=True, dtype=np.float64).clip(1e-12, None)
        pred_prob = P / P.sum(axis=0, keepdims=True, dtype=np.float64).clip(1e-12, None)

        log_pp    = np.log(pred_prob.clip(1e-12, None))
        ce_gt     = -(gt_prob * log_pp).mean(axis=1, dtype=np.float64)

        return {
            "elev_mae_gt"  : mae_gt,
            "elev_rmse_gt" : rmse_gt,
            "elev_r2_gt"   : r2_gt,
            "elev_ce_gt"   : ce_gt,
        }

    def _mu_ordering_rate(self) -> float:
        pp  = self.result.params_pred
        n_K = self.n_gaussians
        
        if n_K < 2:
            return float("nan")

        mus    = np.stack([pp[3 * k + 1] for k in range(n_K)], axis=0)  # (G, H, W)
        amps   = np.stack([pp[3 * k]     for k in range(n_K)], axis=0)  # (G, H, W)
        active = amps >= 1e-3                                             # (G, H, W)

        ordered       = mus[:-1] < mus[1:]                               # (G-1, H, W)
        both_active   = active[:-1] & active[1:]
        has_violation = ((~ordered) & both_active).any(axis=0)           # (H, W)

        n_active     = active.sum(axis=0)                                 # (H, W)
        multi_active = n_active >= 2
        denom        = int(multi_active.sum())
        
        if denom == 0:
            return float("nan")
      
        return float((~has_violation & multi_active).sum() / denom)

    def _slot_mu_stats(self) -> Dict[str, float]:
        """Mean and std of µ (pred and GT) per Gaussian slot, restricted to active pixels."""
        pp  = self.result.params_pred
        pg  = self.result.params_gt
        n_K = self.n_gaussians
        out: Dict[str, float] = {}

        for k in range(n_K):
            gt_amp  = pg[3 * k].reshape(-1)
            active  = (gt_amp >= 1e-3) & np.isfinite(gt_amp)

            mu_pred = pp[3 * k + 1].reshape(-1)[active]
            mu_gt   = pg[3 * k + 1].reshape(-1)[active]
            mu_pred = mu_pred[np.isfinite(mu_pred)]
            mu_gt   = mu_gt  [np.isfinite(mu_gt)]

            out[f"slot_{k}_mu_pred_mean"] = float(np.mean(mu_pred)) if mu_pred.size > 0 else float("nan")
            out[f"slot_{k}_mu_pred_std"]  = float(np.std( mu_pred)) if mu_pred.size > 0 else float("nan")
            out[f"slot_{k}_mu_gt_mean"]   = float(np.mean(mu_gt))   if mu_gt.size   > 0 else float("nan")
            out[f"slot_{k}_mu_gt_std"]    = float(np.std( mu_gt))   if mu_gt.size   > 0 else float("nan")

        return out

    def _placeholder_detection(self) -> Dict[str, float]:
        """Precision / recall / F1 for inactive-slot detection, per slot and overall."""
        pp  = self.result.params_pred
        pg  = self.result.params_gt
        n_K = self.n_gaussians
        out: Dict[str, float] = {}

        all_gt_ph: list[np.ndarray]   = []
        all_pred_ph: list[np.ndarray] = []

        for k in range(n_K):
            gt_amp   = pg[3 * k].reshape(-1)
            pred_amp = pp[3 * k].reshape(-1)
            valid    = np.isfinite(gt_amp) & np.isfinite(pred_amp)

            gt_ph   = (gt_amp  [valid] < 1e-3).astype(np.float32)
            pred_ph = (pred_amp[valid] < 1e-3).astype(np.float32)

            tp = (pred_ph * gt_ph).sum()
            fp = (pred_ph * (1.0 - gt_ph)).sum()
            fn = ((1.0 - pred_ph) * gt_ph).sum()

            precision = float(tp / (tp + fp + 1e-8))
            recall    = float(tp / (tp + fn + 1e-8))
            f1        = 2.0 * precision * recall / (precision + recall + 1e-8)

            out[f"slot_{k}_placeholder_precision"] = precision
            out[f"slot_{k}_placeholder_recall"]    = recall
            out[f"slot_{k}_placeholder_f1"]        = f1
            out[f"slot_{k}_placeholder_gt_rate"]   = float(gt_ph.mean())   if gt_ph.size   > 0 else float("nan")
            out[f"slot_{k}_placeholder_pred_rate"] = float(pred_ph.mean()) if pred_ph.size > 0 else float("nan")

            all_gt_ph.append(gt_ph)
            all_pred_ph.append(pred_ph)

        gt_all   = np.concatenate(all_gt_ph)
        pred_all = np.concatenate(all_pred_ph)
        tp  = (pred_all * gt_all).sum()
        fp  = (pred_all * (1.0 - gt_all)).sum()
        fn  = ((1.0 - pred_all) * gt_all).sum()
        p   = float(tp / (tp + fp + 1e-8))
        r   = float(tp / (tp + fn + 1e-8))
        out["placeholder_precision"] = p
        out["placeholder_recall"]    = r
        out["placeholder_f1"]        = 2.0 * p * r / (p + r + 1e-8)

        return out

    def _permutation_consensus(self) -> Dict[str, float]:
        """Fraction of pixels that prefer the same pred→GT permutation (by µ distance).

        A high dominant-fraction means the model has learnt stable slot roles.
        identity_frac specifically measures how often the identity permutation
        (slot k matched to GT slot k) is optimal, which is expected when both
        pred and GT are µ-sorted.
        """
        pp  = self.result.params_pred
        pg  = self.result.params_gt
        n_K = self.n_gaussians

        if n_K == 1:
            return {"permutation_consensus_dominant_frac": 1.0,
                    "permutation_consensus_identity_frac":  1.0}

        from itertools import permutations as _perms

        pred_mu = np.stack([pp[3 * k + 1] for k in range(n_K)], axis=0).reshape(n_K, -1)
        gt_mu   = np.stack([pg[3 * k + 1] for k in range(n_K)], axis=0).reshape(n_K, -1)

        pred_mu = np.nan_to_num(pred_mu, nan=1e9)
        gt_mu   = np.nan_to_num(gt_mu,   nan=1e9)

        cost_mat = np.abs(pred_mu.T[:, :, None] - gt_mu.T[:, None, :])

        all_perms   = list(_perms(range(n_K)))
        identity_idx = all_perms.index(tuple(range(n_K)))

        if n_K <= 4:
            best_idx = self._best_perm_bruteforce(cost_mat, all_perms, n_K)
        else:
            best_idx = self._best_perm_assignment(cost_mat, all_perms)

        counts = np.bincount(best_idx, minlength=len(all_perms)).astype(np.float64)
        total  = counts.sum()

        dominant_frac = float(counts.max()           / total)
        identity_frac = float(counts[identity_idx]   / total)

        return {
            "permutation_consensus_dominant_frac": dominant_frac,
            "permutation_consensus_identity_frac": identity_frac,
        }

    @staticmethod
    def _best_perm_bruteforce(cost_mat: np.ndarray, all_perms: list, n_K: int) -> np.ndarray:
        perm_costs = np.stack(
            [cost_mat[:, np.arange(n_K), list(p)].sum(axis=1) for p in all_perms],
            axis=1,
        )

        return perm_costs.argmin(axis=1)

    @staticmethod
    def _best_perm_assignment(cost_mat: np.ndarray, all_perms: list) -> np.ndarray:
        from scipy.optimize import linear_sum_assignment

        perm_to_idx = {p: i for i, p in enumerate(all_perms)}
        best_idx    = np.empty(cost_mat.shape[0], dtype=np.int64)

        for hw in range(cost_mat.shape[0]):
            _, col       = linear_sum_assignment(cost_mat[hw])
            best_idx[hw] = perm_to_idx[tuple(col)]

        return best_idx

    def _gaussian_param_metrics(self) -> Dict[str, float]:
        out    : Dict[str, float] = {}
        pp     = self.result.params_pred
        pg     = self.result.params_gt
        n_K    = self.n_gaussians

        all_mu_ae  : list[np.ndarray] = []
        all_sig_ae : list[np.ndarray] = []

        for k in range(n_K):
            gt_amp  = pg[3 * k]                      # (H, W)
            valid   = (gt_amp >= 1e-3) & np.isfinite(pg[3 * k + 1])

            mu_pred = np.where(valid, pp[3 * k + 1], np.nan)
            mu_gt   = np.where(valid, pg[3 * k + 1], np.nan)
            sg_pred = np.where(valid, pp[3 * k + 2], np.nan)
            sg_gt   = np.where(valid, pg[3 * k + 2], np.nan)

            mu_ae  = np.abs(mu_pred  - mu_gt)
            sig_ae = np.abs(sg_pred  - sg_gt)
            mu_se  = (mu_pred  - mu_gt)  ** 2
            sig_se = (sg_pred  - sg_gt)  ** 2

            n_valid = int(np.sum(valid))

            out[f"gauss_{k}_mu_mae"]   = float(np.nanmean(mu_ae,  dtype=np.float64))  if n_valid > 0 else float("nan")
            out[f"gauss_{k}_mu_rmse"]  = float(np.sqrt(np.nanmean(mu_se,  dtype=np.float64)))  if n_valid > 0 else float("nan")
            out[f"gauss_{k}_sig_mae"]  = float(np.nanmean(sig_ae, dtype=np.float64)) if n_valid > 0 else float("nan")
            out[f"gauss_{k}_sig_rmse"] = float(np.sqrt(np.nanmean(sig_se, dtype=np.float64))) if n_valid > 0 else float("nan")
            out[f"gauss_{k}_n_valid"]  = n_valid

            if n_valid > 0:
                all_mu_ae .append(mu_ae [valid])
                all_sig_ae.append(sig_ae[valid])

        if all_mu_ae:
            cat_mu  = np.concatenate(all_mu_ae)
            cat_sig = np.concatenate(all_sig_ae)
            out["gauss_all_mu_mae"]   = float(cat_mu .mean(dtype=np.float64))
            out["gauss_all_mu_rmse"]  = float(np.sqrt((cat_mu  ** 2).mean(dtype=np.float64)))
            out["gauss_all_sig_mae"]  = float(cat_sig.mean(dtype=np.float64))
            out["gauss_all_sig_rmse"] = float(np.sqrt((cat_sig ** 2).mean(dtype=np.float64)))
        else:
            out["gauss_all_mu_mae"]   = float("nan")
            out["gauss_all_mu_rmse"]  = float("nan")
            out["gauss_all_sig_mae"]  = float("nan")
            out["gauss_all_sig_rmse"] = float("nan")

        return out

    def compute(self, *, elev_indices  : Optional[np.ndarray] = None, range_indices : Optional[np.ndarray] = None, az_indices : Optional[np.ndarray] = None) -> Dict[str, float]:
        pred = self.result.pred_curves
        gt   = self.result.gt_curves
    
        diff_gt        = pred - gt
        mse_gt         = float((diff_gt * diff_gt).mean(dtype=np.float64))
        mae_gt         = float(np.abs(diff_gt).mean(dtype=np.float64))
        gt_mean        = float(gt.mean(dtype=np.float64))
        overall_r2_gt  = 1.0 - float((diff_gt * diff_gt).sum(dtype=np.float64)) / (float(((gt  - gt_mean)  ** 2).sum(dtype=np.float64)) + 1e-12)

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

            "gt_mean"        : gt_mean,
            "gt_std"         : float(gt.std(dtype=np.float64)),
            "gt_max"         : float(gt.max()),
            "pred_mean"      : float(pred.mean(dtype=np.float64)),
            "pred_std"       : float(pred.std(dtype=np.float64)),
            "pred_max"       : float(pred.max()),
        }

        for tag, arr in (
            ("pixel_mse_gt",        self.result.pixel_mse),
            ("pixel_mae_gt",        self.result.pixel_mae),
            ("pixel_r2_gt",         self.result.pixel_r2),
            ("pixel_cosine_gt",     self.result.pixel_cosine),
            ("pixel_peak_idx_d_gt", self.result.pixel_peak_err_idx.astype(np.float32)),
        ):
            for k, v in self._basic_stats(arr).items():
                metrics[f"{tag}_{k}"] = v
            for k, v in self._percentiles(arr).items():
                metrics[f"{tag}_{k}"] = v

        metrics["pixel_peak_err_units_mean_gt"]   = float(self.result.pixel_peak_err_idx.mean())              * self.x_step
        metrics["pixel_peak_err_units_median_gt"] = float(np.median(self.result.pixel_peak_err_idx))          * self.x_step
        metrics["pixel_peak_err_units_p95_gt"]    = float(np.percentile(self.result.pixel_peak_err_idx, 95))  * self.x_step

        for k, v in self._slice_ssim(pred, gt, elev_indices, range_indices, az_indices, prefix="gt").items():
            metrics[k] = v

        for metric_name, arr in self._elev_metrics(pred, gt).items():
            for i, v in enumerate(arr):
                metrics[f"{metric_name}_{i}"] = float(v)
            
            metrics[f"{metric_name}_mean"] = float(np.nanmean(arr))

        # ── Per-Gaussian µ / σ errors (placeholder-masked, µ-sorted order) ──────────
        for k, v in self._gaussian_param_metrics().items():
            metrics[k] = v

        # ── Slot µ statistics (mean & std per slot, active pixels only) ─────────────
        for k, v in self._slot_mu_stats().items():
            metrics[k] = v

        # ── Inactive-Gaussian detection (precision / recall / F1) ───────────────────
        for k, v in self._placeholder_detection().items():
            metrics[k] = v

        # ── µ ordering rate & permutation consensus ──────────────────────────────────
        metrics["mu_ordering_rate"] = self._mu_ordering_rate()
        for k, v in self._permutation_consensus().items():
            metrics[k] = v

        return metrics

 

