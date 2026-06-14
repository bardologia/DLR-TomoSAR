from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses        import dataclass
from itertools          import permutations
from pathlib            import Path
from typing             import Dict, Optional, Tuple

import numpy         as np
from scipy.optimize  import linear_sum_assignment
from skimage.metrics import structural_similarity as ssim
from tqdm            import tqdm

from tools.data.io            import FileIO
from tools.data.preprocessing import ProfileNormalizer
from tools.metrics.scoring    import R2, RelativeImprovement


@dataclass
class ReducedComparison:
    reduced_curves : np.ndarray
    gt_norm        : np.ndarray
    pred_norm      : np.ndarray
    reduced_norm   : np.ndarray
    err_pred       : np.ndarray
    err_reduced    : np.ndarray
    improvement    : np.ndarray
    metrics        : Dict[str, float]


@dataclass
class Result:
    pred_curves        : np.ndarray
    gt_curves          : np.ndarray

    pixel_mse          : np.ndarray
    pixel_mae          : np.ndarray
    pixel_r2           : np.ndarray
    pixel_cosine       : np.ndarray
    pixel_peak_err_idx : np.ndarray

    cube_directory     : Path
    azimuth_offset     : int
    range_offset       : int

    params_pred : Optional[np.ndarray]        = None
    params_gt   : Optional[np.ndarray]        = None
    reduced     : Optional[ReducedComparison] = None


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
    def curve_pixel_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, np.ndarray]:
        diff   = pred - ref
        mse    = (diff * diff).mean(axis=0).astype(np.float32)
        mae    = np.abs(diff).mean(axis=0).astype(np.float32)
        r2     = R2.pixel_map(pred, ref, axis=0)
        dot    = (pred * ref).sum(axis=0)
        norm_p = np.sqrt((pred * pred).sum(axis=0)) + 1e-8
        norm_r = np.sqrt((ref  * ref ).sum(axis=0)) + 1e-8
        cos    = (dot / (norm_p * norm_r)).astype(np.float32)
        peak   = np.abs(pred.argmax(axis=0) - ref.argmax(axis=0)).astype(np.int32)

        return {"mse": mse, "mae": mae, "r2": r2, "cos": cos, "peak": peak}

    @staticmethod
    def write_json(metrics: Dict[str, object], path: Path) -> Path:
        return FileIO.save_json(metrics, path, indent=4)

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
        rng       = np.random.default_rng(seed)
        valid_idx = np.where(np.isfinite(flat))[0]
        order     = valid_idx[np.argsort(flat[valid_idx])]

        best_flat  = order[:n_best]
        worst_flat = order[-n_worst:][::-1]

        pool      = np.setdiff1d(valid_idx, np.concatenate([best_flat, worst_flat]), assume_unique=False)
        n_random  = min(n_random, pool.size)
        rand_flat = rng.choice(pool, size=n_random, replace=False) if n_random > 0 else np.array([], dtype=np.int64)

        return {
            "best"   : Metrics._flat_to_yx(best_flat,  W),
            "worst"  : Metrics._flat_to_yx(worst_flat, W),
            "random" : Metrics._flat_to_yx(rand_flat,  W),
        }

    @staticmethod
    def _flat_to_yx(flat_idx: np.ndarray, width: int) -> np.ndarray:
        return np.stack([(flat_idx // width).astype(np.int32), (flat_idx % width).astype(np.int32)], axis=1)

    def _curve_scalar_metrics(self, pred: np.ndarray, ref: np.ndarray, suffix: str) -> Dict[str, float]:
        diff       = pred - ref
        mse        = float((diff * diff).mean(dtype=np.float64))
        mae        = float(np.abs(diff).mean(dtype=np.float64))
        ref_mean   = float(ref.mean(dtype=np.float64))
        overall_r2 = 1.0 - float((diff * diff).sum(dtype=np.float64)) / (float(((ref - ref_mean) ** 2).sum(dtype=np.float64)) + 1e-12)

        return {
            f"curve_mse_{suffix}"  : mse,
            f"curve_mae_{suffix}"  : mae,
            f"curve_rmse_{suffix}" : float(np.sqrt(mse)),
            f"overall_r2_{suffix}" : float(overall_r2),
            f"psnr_db_{suffix}"    : self._psnr(pred, ref),
        }

    @staticmethod
    def _psnr(pred: np.ndarray, ref: np.ndarray) -> float:
        diff = pred - ref
        mse  = float((diff * diff).mean(dtype=np.float64))

        if mse <= 0.0:
            return float("inf")

        data_range = float(ref.max() - ref.min())
        if data_range <= 0.0:
            return float("nan")

        return 10.0 * np.log10(data_range * data_range / mse)

    def _expand_pixel_stats(self, tagged_arrays: Tuple[Tuple[str, np.ndarray], ...]) -> Dict[str, float]:
        out: Dict[str, float] = {}

        for tag, arr in tagged_arrays:
            out.update({f"{tag}_{k}": v for k, v in self._basic_stats(arr).items()})
            out.update({f"{tag}_{k}": v for k, v in self._percentiles(arr).items()})

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
    def _percentiles(x: np.ndarray, qs: Tuple[int, ...] = (1, 5, 25, 50, 75, 95, 99)) -> Dict[str, float]:
        flat = x.reshape(-1)
        out  = {}
        p    = np.percentile(flat, qs)

        for q, v in zip(qs, p):
            out[f"p{q}"] = float(v)

        return out

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

        if indices is None:
            return

        for i, idx in enumerate(indices):
            if axis == 0:
                p_sl, r_sl = pred[idx, :, :], ref[idx, :, :]
            elif axis == 1:
                p_sl, r_sl = pred[:, idx, :], ref[:, idx, :]
            else:
                p_sl, r_sl = pred[:, :, idx], ref[:, :, idx]

            tasks.append((f"ssim_{prefix}_{label}_{i}", label, p_sl.astype(np.float64), r_sl.astype(np.float64)))

    @staticmethod
    def _ssim_worker(args: tuple) -> tuple[str, float]:
        key, pred_slice, ref_slice = args
        data_range = float(ref_slice.max() - ref_slice.min())
        min_side   = min(pred_slice.shape)
        win_size   = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        value      = float(ssim(ref_slice, pred_slice, data_range=data_range, win_size=win_size))

        return key, value

    def _expand_elev(self, pred: np.ndarray, gt: np.ndarray, suffix: str) -> Dict[str, float]:
        out: Dict[str, float] = {}

        for metric_name, arr in self._elev_metrics(pred, gt, suffix=suffix).items():
            out.update({f"{metric_name}_{i}": float(v) for i, v in enumerate(arr)})
            out[f"{metric_name}_mean"] = float(np.nanmean(arr))

        return out

    def _elev_metrics(self, pred: np.ndarray, gt: np.ndarray, suffix: str = "gt") -> Dict[str, np.ndarray]:
        P = pred.reshape(pred.shape[0], -1)
        G = gt  .reshape(gt  .shape[0], -1)

        diff_g  = P - G
        mae_gt  = np.abs(diff_g).mean(axis=1, dtype=np.float64)
        rmse_gt = np.sqrt((diff_g ** 2).mean(axis=1, dtype=np.float64))
        g_var   = ((G - G.mean(axis=1, keepdims=True, dtype=np.float64)) ** 2).sum(axis=1, dtype=np.float64) + 1e-12
        r2_gt   = 1.0 - (diff_g ** 2).sum(axis=1, dtype=np.float64) / g_var

        gt_prob   = G / G.sum(axis=0, keepdims=True, dtype=np.float64).clip(1e-12, None)
        pred_prob = P / P.sum(axis=0, keepdims=True, dtype=np.float64).clip(1e-12, None)

        log_pp = np.log(pred_prob.clip(1e-12, None))
        ce_gt  = -(gt_prob * log_pp).mean(axis=1, dtype=np.float64)

        return {
            f"elev_mae_{suffix}"  : mae_gt,
            f"elev_rmse_{suffix}" : rmse_gt,
            f"elev_r2_{suffix}"   : r2_gt,
            f"elev_ce_{suffix}"   : ce_gt,
        }

    def _gaussian_param_metrics(self) -> Dict[str, float]:
        out    : Dict[str, float] = {}
        pp  = self.result.params_pred
        pg  = self.result.params_gt
        n_K = self.n_gaussians

        all_mu_ae  : list[np.ndarray] = []
        all_sig_ae : list[np.ndarray] = []

        for k in range(n_K):
            gt_amp = pg[3 * k]
            valid  = (gt_amp >= 1e-3) & np.isfinite(pg[3 * k + 1])

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

    def _slot_mu_stats(self) -> Dict[str, float]:
        pp  = self.result.params_pred
        pg  = self.result.params_gt
        n_K = self.n_gaussians
        out: Dict[str, float] = {}

        for k in range(n_K):
            gt_amp = pg[3 * k].reshape(-1)
            active = (gt_amp >= 1e-3) & np.isfinite(gt_amp)

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
        pp  = self.result.params_pred
        pg  = self.result.params_gt
        n_K = self.n_gaussians
        out: Dict[str, float] = {}

        all_gt_ph   : list[np.ndarray] = []
        all_pred_ph : list[np.ndarray] = []

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
        tp       = (pred_all * gt_all).sum()
        fp       = (pred_all * (1.0 - gt_all)).sum()
        fn       = ((1.0 - pred_all) * gt_all).sum()
        p        = float(tp / (tp + fp + 1e-8))
        r        = float(tp / (tp + fn + 1e-8))
        out["placeholder_precision"] = p
        out["placeholder_recall"]    = r
        out["placeholder_f1"]        = 2.0 * p * r / (p + r + 1e-8)

        return out

    def _mu_ordering_rate(self) -> float:
        pp  = self.result.params_pred
        n_K = self.n_gaussians

        if n_K < 2:
            return float("nan")

        mus    = np.stack([pp[3 * k + 1] for k in range(n_K)], axis=0)
        amps   = np.stack([pp[3 * k]     for k in range(n_K)], axis=0)
        active = amps >= 1e-3

        ordered       = mus[:-1] < mus[1:]
        both_active   = active[:-1] & active[1:]
        has_violation = ((~ordered) & both_active).any(axis=0)

        n_active     = active.sum(axis=0)
        multi_active = n_active >= 2
        denom        = int(multi_active.sum())

        if denom == 0:
            return float("nan")

        return float((~has_violation & multi_active).sum() / denom)

    def _permutation_consensus(self) -> Dict[str, float]:
        pp  = self.result.params_pred
        pg  = self.result.params_gt
        n_K = self.n_gaussians

        if n_K == 1:
            return {"permutation_consensus_dominant_frac": 1.0,
                    "permutation_consensus_identity_frac":  1.0}

        pred_mu = np.stack([pp[3 * k + 1] for k in range(n_K)], axis=0).reshape(n_K, -1)
        gt_mu   = np.stack([pg[3 * k + 1] for k in range(n_K)], axis=0).reshape(n_K, -1)

        pred_mu = np.nan_to_num(pred_mu, nan=1e9)
        gt_mu   = np.nan_to_num(gt_mu,   nan=1e9)

        cost_mat = np.abs(pred_mu.T[:, :, None] - gt_mu.T[:, None, :])

        all_perms    = list(permutations(range(n_K)))
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
        perm_to_idx = {p: i for i, p in enumerate(all_perms)}
        best_idx    = np.empty(cost_mat.shape[0], dtype=np.int64)

        for hw in range(cost_mat.shape[0]):
            _, col       = linear_sum_assignment(cost_mat[hw])
            best_idx[hw] = perm_to_idx[tuple(col)]

        return best_idx

    def compute(self, *, elev_indices  : Optional[np.ndarray] = None, range_indices : Optional[np.ndarray] = None, az_indices : Optional[np.ndarray] = None, param_space : bool = True) -> Dict[str, float]:
        pred = self.result.pred_curves
        gt   = self.result.gt_curves

        metrics: Dict[str, float] = {
            "n_pixels"    : int(self.result.pixel_mse.size),
            "n_elevation" : int(pred.shape[0]),
            "x_axis_min"  : float(self.x_axis.min()),
            "x_axis_max"  : float(self.x_axis.max()),
            "x_axis_step" : self.x_step,

            "gt_mean"     : float(gt.mean(dtype=np.float64)),
            "gt_std"      : float(gt.std(dtype=np.float64)),
            "gt_max"      : float(gt.max()),
            "pred_mean"   : float(pred.mean(dtype=np.float64)),
            "pred_std"    : float(pred.std(dtype=np.float64)),
            "pred_max"    : float(pred.max()),
        }

        metrics.update(self._curve_scalar_metrics(pred, gt, suffix="gt"))

        metrics.update(self._expand_pixel_stats((
            ("pixel_mse_gt",        self.result.pixel_mse),
            ("pixel_mae_gt",        self.result.pixel_mae),
            ("pixel_r2_gt",         self.result.pixel_r2),
            ("pixel_cosine_gt",     self.result.pixel_cosine),
            ("pixel_peak_idx_d_gt", self.result.pixel_peak_err_idx.astype(np.float32)),
        )))

        metrics["pixel_peak_err_units_mean_gt"]   = float(self.result.pixel_peak_err_idx.mean())              * self.x_step
        metrics["pixel_peak_err_units_median_gt"] = float(np.median(self.result.pixel_peak_err_idx))          * self.x_step
        metrics["pixel_peak_err_units_p95_gt"]    = float(np.percentile(self.result.pixel_peak_err_idx, 95))  * self.x_step

        metrics.update(self._slice_ssim(pred, gt, elev_indices, range_indices, az_indices, prefix="gt"))
        metrics.update(self._expand_elev(pred, gt, suffix="gt"))

        if param_space:
            metrics.update(self._gaussian_param_metrics())
            metrics.update(self._slot_mu_stats())
            metrics.update(self._placeholder_detection())

            metrics["mu_ordering_rate"] = self._mu_ordering_rate()
            metrics.update(self._permutation_consensus())

        return metrics

    def reduced_comparison(
        self,
        reduced_curves : np.ndarray,
        *,
        elev_indices   : Optional[np.ndarray] = None,
        range_indices  : Optional[np.ndarray] = None,
        az_indices     : Optional[np.ndarray] = None,
    ) -> ReducedComparison:

        gt_n   = ProfileNormalizer.unit_area(self.result.gt_curves)
        pred_n = ProfileNormalizer.unit_area(self.result.pred_curves)
        red_n  = ProfileNormalizer.unit_area(reduced_curves)

        out: Dict[str, float] = {}
        out.update(self._curve_scalar_metrics(red_n, gt_n, suffix="red"))
        out.update(self._slice_ssim(red_n, gt_n, elev_indices, range_indices, az_indices, prefix="red"))
        out.update(self._expand_elev(red_n, gt_n, suffix="red"))

        err_pred    = self.curve_pixel_metrics(pred_n, gt_n)["mse"]
        err_reduced = self.curve_pixel_metrics(red_n,  gt_n)["mse"]
        improvement = (err_reduced - err_pred).astype(np.float32)

        out.update(self._expand_pixel_stats((
            ("pixel_mse_red",   err_reduced),
            ("pixel_mse_predn", err_pred),
        )))

        finite           = np.isfinite(improvement)
        pred_mse_mean    = float(np.nanmean(err_pred))
        reduced_mse_mean = float(np.nanmean(err_reduced))

        out["pred_pixel_mse_norm_mean"]     = pred_mse_mean
        out["reduced_pixel_mse_norm_mean"]  = reduced_mse_mean
        out["improvement_pixel_mse_mean"]   = float(np.nanmean(improvement))
        out["improvement_pixel_mse_median"] = float(np.nanmedian(improvement))
        out["fraction_pred_beats_reduced"]  = float(np.mean(err_pred[finite] < err_reduced[finite])) if finite.any() else float("nan")
        out["relative_mse_reduction"]       = RelativeImprovement.fraction(reduced_mse_mean, pred_mse_mean, higher_is_better=False)

        return ReducedComparison(
            reduced_curves = reduced_curves,
            gt_norm        = gt_n,
            pred_norm      = pred_n,
            reduced_norm   = red_n,
            err_pred       = err_pred,
            err_reduced    = err_reduced,
            improvement    = improvement,
            metrics        = out,
        )
