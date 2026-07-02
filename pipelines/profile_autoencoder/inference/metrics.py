from __future__ import annotations

import numpy as np

from pipelines.autoencoder_common.inference.metrics import AeMetricsBase


class ProfileAeMetrics(AeMetricsBase):
    def __init__(self, result, x_axis: np.ndarray, normalizer, amp_zero_thr: float) -> None:
        super().__init__(result, normalizer)

        self.x_axis       = np.asarray(x_axis, dtype=np.float64)
        self.amp_zero_thr = amp_zero_thr
        self.active       = self.gt.max(axis=1) > self.amp_zero_thr

    def per_curve_mse(self) -> np.ndarray:
        return np.mean((self.pred - self.gt) ** 2, axis=1)

    def _physical_errors(self) -> dict:
        diff = self.pred - self.gt

        sse = float(np.sum(diff ** 2))
        sst = float(np.sum((self.gt - self.gt.mean()) ** 2))

        return {
            "mse_mean"           : float(np.mean(diff ** 2)),
            "mse_median"         : float(np.median(self.per_curve_mse())),
            "mae_mean"           : float(np.mean(np.abs(diff))),
            "rmse"               : float(np.sqrt(np.mean(diff ** 2))),
            "max_abs_error_mean" : float(np.mean(np.max(np.abs(diff), axis=1))),
            "r2"                 : float(1.0 - sse / (sst + self.EPS)),
        }

    def _normalized_errors(self) -> dict:
        gt_n   = self.normalizer.normalize(self.gt.astype(np.float32)).astype(np.float64)
        pred_n = self.normalizer.normalize(self.pred.astype(np.float32)).astype(np.float64)

        return {
            "mse_mean_normalized" : float(np.mean((pred_n - gt_n) ** 2)),
            "mae_mean_normalized" : float(np.mean(np.abs(pred_n - gt_n))),
        }

    def _shape_errors(self) -> dict:
        gt   = self.gt[self.active]
        pred = self.pred[self.active]

        gt_c   = gt   - gt.mean(axis=1,   keepdims=True)
        pred_c = pred - pred.mean(axis=1, keepdims=True)

        denom    = np.sqrt(np.sum(gt_c ** 2, axis=1) * np.sum(pred_c ** 2, axis=1))
        pearson  = np.sum(gt_c * pred_c, axis=1) / (denom + self.EPS)

        rel_l2 = np.linalg.norm(pred - gt, axis=1) / (np.linalg.norm(gt, axis=1) + self.EPS)

        return {
            "pearson_mean"      : float(np.mean(pearson)),
            "pearson_median"    : float(np.median(pearson)),
            "relative_l2_mean"  : float(np.mean(rel_l2)),
            "relative_l2_median": float(np.median(rel_l2)),
        }

    def _power_errors(self) -> dict:
        gt   = self.gt[self.active]
        pred = self.pred[self.active]

        power_gt   = np.trapezoid(gt,   self.x_axis, axis=1)
        power_pred = np.trapezoid(pred, self.x_axis, axis=1)

        rel = np.abs(power_pred - power_gt) / (np.abs(power_gt) + self.EPS)

        peak_gt   = self.x_axis[np.argmax(gt,   axis=1)]
        peak_pred = self.x_axis[np.argmax(pred, axis=1)]

        amp_gt   = np.max(gt,   axis=1)
        amp_pred = np.max(pred, axis=1)

        return {
            "power_rel_error_mean"        : float(np.mean(rel)),
            "power_rel_error_median"      : float(np.median(rel)),
            "peak_location_mae"           : float(np.mean(np.abs(peak_pred - peak_gt))),
            "peak_amplitude_rel_err_mean" : float(np.mean(np.abs(amp_pred - amp_gt) / (amp_gt + self.EPS))),
        }

    def compute(self) -> dict:
        metrics = {
            "n_curves"         : int(self.gt.shape[0]),
            "n_active_curves"  : int(self.active.sum()),
            "profile_length"   : int(self.gt.shape[1]),
            "embedding_dim"    : int(self.emb.shape[1]),
        }

        metrics.update(self._physical_errors())
        metrics.update(self._normalized_errors())
        metrics.update(self._shape_errors())
        metrics.update(self._power_errors())
        metrics.update(self._embedding_stats())

        return metrics
