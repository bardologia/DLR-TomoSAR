from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses        import dataclass
from pathlib            import Path
from typing             import Dict, Optional, Tuple

import numpy         as np
from skimage.metrics import structural_similarity as ssim
from tqdm            import tqdm

from tools.data.io               import FileIO
from tools.data.preprocessing    import ProfileNormalizer
from tools.metrics.gaussian_matching import GaussianMatcher
from tools.metrics.slot_organization import SlotOrganization
from tools.metrics.scoring       import R2, RelativeImprovement


@dataclass
class ReducedComparison:
    reduced_curves : np.ndarray
    gt_norm        : np.ndarray
    reduced_norm   : np.ndarray
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

    def _active_count_stats(self) -> Dict[str, float]:
        pp  = self.result.params_pred
        pg  = self.result.params_gt
        n_K = self.n_gaussians
        out: Dict[str, float] = {}

        H, W       = pg.shape[-2:]
        gt_count   = np.zeros((H, W), dtype=np.int32)
        pred_count = np.zeros((H, W), dtype=np.int32)

        for k in range(n_K):
            gt_active   = pg[3 * k] >= 1e-3
            pred_active = pp[3 * k] >= 1e-3

            out[f"slot_{k}_active_gt_frac"]   = float(gt_active.mean())
            out[f"slot_{k}_active_pred_frac"] = float(pred_active.mean())

            gt_count   += gt_active.astype(np.int32)
            pred_count += pred_active.astype(np.int32)

        out["active_frac_gt"]   = float(gt_count.sum(dtype=np.float64)   / (n_K * H * W))
        out["active_frac_pred"] = float(pred_count.sum(dtype=np.float64) / (n_K * H * W))

        out["active_count_gt_mean"]   = float(gt_count.mean(dtype=np.float64))
        out["active_count_pred_mean"] = float(pred_count.mean(dtype=np.float64))

        exact = pred_count == gt_count
        out["count_exact_frac"] = float(exact.mean())
        out["count_under_frac"] = float((pred_count < gt_count).mean())
        out["count_over_frac"]  = float((pred_count > gt_count).mean())

        for k in range(1, n_K + 1):
            gt_mask_k = gt_count == k
            gt_denom  = int(gt_mask_k.sum())
            if gt_denom > 0:
                out[f"count_acc_gt{k}"] = float((exact & gt_mask_k).sum() / gt_denom)

            pred_mask_k = pred_count == k
            pred_denom  = int(pred_mask_k.sum())
            if pred_denom > 0:
                out[f"count_acc_pred{k}"] = float((exact & pred_mask_k).sum() / pred_denom)

        return out

    def _matched_gaussian_metrics(self, match_tol: float = 5.0) -> Dict[str, float]:
        pp  = self.result.params_pred
        pg  = self.result.params_gt
        n_K = self.n_gaussians

        sel = GaussianMatcher().assignment(pp, pg, n_K)

        amp_pred = np.stack([pp[3 * k]     for k in range(n_K)], axis=0).reshape(n_K, -1)
        mu_pred  = np.stack([pp[3 * k + 1] for k in range(n_K)], axis=0).reshape(n_K, -1)
        sig_pred = np.stack([pp[3 * k + 2] for k in range(n_K)], axis=0).reshape(n_K, -1)
        amp_gt   = np.stack([pg[3 * k]     for k in range(n_K)], axis=0).reshape(n_K, -1)
        mu_gt    = np.stack([pg[3 * k + 1] for k in range(n_K)], axis=0).reshape(n_K, -1)
        sig_gt   = np.stack([pg[3 * k + 2] for k in range(n_K)], axis=0).reshape(n_K, -1)

        act_pred = amp_pred >= 1e-3
        act_gt   = amp_gt   >= 1e-3
        gt_count = act_gt.sum(axis=0).astype(np.int64)

        n_buckets        = n_K + 1
        rows             = np.arange(gt_count.size)
        sum_mu_ae        = np.zeros(n_buckets, dtype=np.float64)
        sum_mu_se        = np.zeros(n_buckets, dtype=np.float64)
        sum_sig_ae       = np.zeros(n_buckets, dtype=np.float64)
        sum_sig_se       = np.zeros(n_buckets, dtype=np.float64)
        n_matched_bucket = np.zeros(n_buckets, dtype=np.float64)
        tp_bucket        = np.zeros(n_buckets, dtype=np.float64)

        for i in range(n_K):
            j       = sel[:, i]
            matched = act_pred[i] & act_gt[j, rows]
            if not matched.any():
                continue

            jm   = j[matched]
            rm   = rows[matched]
            dmu  = np.abs(mu_pred [i, matched] - mu_gt [jm, rm])
            dsig = np.abs(sig_pred[i, matched] - sig_gt[jm, rm])
            ck   = gt_count[matched]
            tp   = ck[dmu <= match_tol]

            sum_mu_ae        += np.bincount(ck, weights=dmu,         minlength=n_buckets)
            sum_mu_se        += np.bincount(ck, weights=dmu * dmu,   minlength=n_buckets)
            sum_sig_ae       += np.bincount(ck, weights=dsig,        minlength=n_buckets)
            sum_sig_se       += np.bincount(ck, weights=dsig * dsig, minlength=n_buckets)
            n_matched_bucket += np.bincount(ck,                      minlength=n_buckets)
            tp_bucket        += np.bincount(tp,                      minlength=n_buckets)

        out: Dict[str, float] = {}

        total_matched = float(n_matched_bucket.sum())
        total_gt      = float(act_gt.sum())
        total_pred    = float(act_pred.sum())
        total_tp      = float(tp_bucket.sum())

        out["matched_mu_mae"]   = float(sum_mu_ae.sum()  / total_matched)          if total_matched > 0 else float("nan")
        out["matched_mu_rmse"]  = float(np.sqrt(sum_mu_se.sum()  / total_matched)) if total_matched > 0 else float("nan")
        out["matched_sig_mae"]  = float(sum_sig_ae.sum() / total_matched)          if total_matched > 0 else float("nan")
        out["matched_sig_rmse"] = float(np.sqrt(sum_sig_se.sum() / total_matched)) if total_matched > 0 else float("nan")

        recall    = total_tp / total_gt   if total_gt   > 0 else float("nan")
        precision = total_tp / total_pred if total_pred > 0 else float("nan")
        denom_f1  = recall + precision

        out["matched_recall"]    = float(recall)
        out["matched_precision"] = float(precision)
        out["matched_f1"]        = float(2.0 * recall * precision / denom_f1) if denom_f1 > 0 else float("nan")
        out["matched_n_pairs"]   = total_matched
        out["matched_tol"]       = float(match_tol)

        pixel_count_hist  = np.bincount(gt_count, minlength=n_buckets).astype(np.float64)
        pred_active       = act_pred.sum(axis=0).astype(np.float64)
        pred_active_bucket = np.bincount(gt_count, weights=pred_active, minlength=n_buckets)

        for k in range(1, n_buckets):
            n_k = n_matched_bucket[k]
            if n_k > 0:
                out[f"matched_mu_mae_gt{k}"]  = float(sum_mu_ae[k]  / n_k)
                out[f"matched_sig_mae_gt{k}"] = float(sum_sig_ae[k] / n_k)

            gt_active_k = k * pixel_count_hist[k]
            if gt_active_k > 0:
                out[f"matched_recall_gt{k}"] = float(tp_bucket[k] / gt_active_k)

            if pred_active_bucket[k] > 0:
                out[f"matched_precision_gt{k}"] = float(tp_bucket[k] / pred_active_bucket[k])

        return out

    def _slot_organization_stats(self) -> Dict[str, float]:
        pp  = self.result.params_pred
        pg  = self.result.params_gt
        n_K = self.n_gaussians
        out: Dict[str, float] = {}

        usage = SlotOrganization.usage_fractions(pp, n_K)
        out["slot_usage_entropy"] = SlotOrganization.usage_entropy(usage)

        rank_counts = SlotOrganization.mu_rank_matrix(pp, n_K)
        out["slot_mu_rank_diag"] = SlotOrganization.diagonality(rank_counts)

        if pg is not None:
            assign_counts = SlotOrganization.assignment_matrix(pp, pg, n_K)
            out["slot_gt_alignment"] = SlotOrganization.diagonality(assign_counts)

        return out

    def compute(self, *, elev_indices  : Optional[np.ndarray] = None, range_indices : Optional[np.ndarray] = None, az_indices : Optional[np.ndarray] = None, param_space : bool = True, match_tol : float = 5.0) -> Dict[str, float]:
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

        pred_norm = ProfileNormalizer.unit_area(pred)
        gt_norm   = ProfileNormalizer.unit_area(gt)
        metrics.update(self._slice_ssim(pred_norm, gt_norm, elev_indices, range_indices, az_indices, prefix="norm"))

        metrics.update(self._expand_elev(pred, gt, suffix="gt"))

        if param_space:
            metrics.update(self._active_count_stats())
            metrics.update(self._matched_gaussian_metrics(match_tol=match_tol))
            metrics.update(self._slot_organization_stats())

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
            reduced_norm   = red_n,
            err_reduced    = err_reduced,
            improvement    = improvement,
            metrics        = out,
        )
