import numpy as np
import torch
import torch.nn.functional as F


class MetricsAggregator:
    def __init__(self, metrics, deep: bool, gaussian_cfg, keep_pixel_arrays: bool = True, pixel_subsample: int = 0):
        self.metrics           = metrics
        self.deep              = deep
        self.gaussian_cfg      = gaussian_cfg
        self.keep_pixel_arrays = keep_pixel_arrays
        self.pixel_subsample   = int(pixel_subsample) if pixel_subsample else 0

        self.total_elements = 0
        self.total_pixels   = 0
        self.sum_squared    = 0.0
        self.sum_absolute   = 0.0

        self.expected_stats = {"count": 0, "mean": 0.0, "m2": 0.0}
        
        self.pixel_mse_store = []
        self.pixel_mae_store = []
        self.pixel_r2_store  = []
        self.pixel_cos_store = []
        self.pixel_coh_store = []
        
        self.param_stats    = {}
        self.gt_param_stats = {}

    @staticmethod
    def _welford_update(state: dict, values: torch.Tensor):
        num_new = values.numel()
        if num_new == 0:
            return
            
        mean_new = float(values.mean())
        m2_new   = float(((values - mean_new) ** 2).sum())
        
        num_existing  = state["count"]
        mean_existing = state["mean"]
        m2_existing   = state.get("m2", 0.0)
        
        total_count = num_existing + num_new
        delta       = mean_new - mean_existing
        
        state["mean"]  = mean_existing + delta * num_new / total_count
        state["m2"]    = m2_existing + m2_new + delta * delta * num_existing * num_new / total_count
        state["count"] = total_count
        
        min_new = float(values.min())
        max_new = float(values.max())
        state["min"] = min_new if "min" not in state else min(state["min"], min_new)
        state["max"] = max_new if "max" not in state else max(state["max"], max_new)

    def _store_pixel_metrics(self, store: list, tensor: torch.Tensor):
        if not self.keep_pixel_arrays:
            return
            
        flat_tensor = tensor.reshape(-1).detach().cpu()
        if self.pixel_subsample > 0 and flat_tensor.numel() > self.pixel_subsample:
            indices = torch.randint(0, flat_tensor.numel(), (self.pixel_subsample,))
            flat_tensor = flat_tensor[indices]
            
        store.append(flat_tensor)

    @torch.no_grad()
    def update(self, pred_params: torch.Tensor, exp_curves: torch.Tensor, gt_params=None):
        params_per_gaussian = self.gaussian_cfg.params_per_gaussian

        pred_curves    = self.metrics.reconstruct_gaussians(pred_params)
        curve_diff     = pred_curves - exp_curves
        abs_curve_diff = curve_diff.abs()

        self.sum_squared    += float((curve_diff ** 2).sum())
        self.sum_absolute   += float(abs_curve_diff.sum())
        self.total_elements += curve_diff.numel()

        pixel_mse   = (curve_diff ** 2).mean(dim=1)
        pixel_mae   = abs_curve_diff.mean(dim=1)
        ss_residual = (curve_diff ** 2).sum(dim=1)
        exp_mean    = exp_curves.mean(dim=1, keepdim=True)
        ss_total    = ((exp_curves - exp_mean) ** 2).sum(dim=1)
        pixel_r2    = 1.0 - ss_residual / (ss_total + 1e-8)
        
        self._store_pixel_metrics(self.pixel_mse_store, pixel_mse)
        self._store_pixel_metrics(self.pixel_mae_store, pixel_mae)
        self._store_pixel_metrics(self.pixel_r2_store,  pixel_r2)
        self.total_pixels += pixel_mse.numel()

        self._welford_update(self.expected_stats, exp_curves.reshape(-1))

        if self.deep:
            cos_sim  = F.cosine_similarity(pred_curves, exp_curves, dim=1)
            spec_coh = self.metrics.spectral_coherence(pred_curves, exp_curves)
            self._store_pixel_metrics(self.pixel_cos_store, cos_sim)
            self._store_pixel_metrics(self.pixel_coh_store, spec_coh)
            del cos_sim, spec_coh

        num_gaussians = pred_params.shape[1] // params_per_gaussian
        param_names   = self.gaussian_cfg.make_param_names(num_gaussians)
        
        for i, name in enumerate(param_names):
            state = self.param_stats.setdefault(name, {"count": 0, "mean": 0.0, "m2": 0.0})
            self._welford_update(state, pred_params[:, i].reshape(-1))

        if self.deep and gt_params is not None:
            num_channels = min(pred_params.shape[1], gt_params.shape[1])
            for i in range(num_channels):
                pred_flat = pred_params[:, i].reshape(-1)
                gt_flat   = gt_params[:, i].reshape(-1)
                
                stats = self.gt_param_stats.setdefault(i, {
                    "sum_squared": 0.0, "sum_absolute": 0.0, "sum_gt": 0.0, "sum_gt_squared": 0.0, "count": 0
                })
                
                diff = pred_flat - gt_flat
                stats["sum_squared"]    += float((diff ** 2).sum())
                stats["sum_absolute"]   += float(diff.abs().sum())
                stats["sum_gt"]         += float(gt_flat.sum())
                stats["sum_gt_squared"] += float((gt_flat ** 2).sum())
                stats["count"]          += diff.numel()

        del pred_curves, curve_diff, abs_curve_diff, pixel_mse, pixel_mae, ss_residual, ss_total, pixel_r2, exp_mean

    def _compute_array_stats(self, tensor_list: list) -> dict | None:
        if not tensor_list:
            return None
            
        concatenated = torch.cat(tensor_list)
        return {
            "mean":   float(concatenated.mean()), 
            "std":    float(concatenated.std()),
            "median": float(concatenated.median()), 
            "min":    float(concatenated.min()), 
            "max":    float(concatenated.max()),
        }

    def finalize(self, epoch: int, stage: str, last_channels: int) -> dict:
        curve_mse  = self.sum_squared / max(1, self.total_elements)
        curve_mae  = self.sum_absolute / max(1, self.total_elements)
        curve_rmse = float(np.sqrt(curve_mse))
        
        overall_ss_total = self.expected_stats["m2"]
        overall_r2       = 1.0 - (self.sum_squared / overall_ss_total) if overall_ss_total > 0 else 0.0

        results = {
            "curve_mse":  curve_mse,
            "curve_mae":  curve_mae,
            "curve_rmse": curve_rmse,
            "overall_r2": overall_r2,
        }
        
        metric_stores = [
            ("pixel_mse", self.pixel_mse_store), 
            ("pixel_mae", self.pixel_mae_store), 
            ("pixel_r2",  self.pixel_r2_store)
        ]
        
        for key, store in metric_stores:
            stats = self._compute_array_stats(store)
            if stats is None:
                fallback_mean = curve_mse if key == "pixel_mse" else (curve_mae if key == "pixel_mae" else 0.0)
                stats = {"mean": fallback_mean, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
            for stat_key, stat_value in stats.items():
                results[f"{key}_{stat_key}"] = stat_value

        if self.deep:
            cos_stats = self._compute_array_stats(self.pixel_cos_store)
            if cos_stats:
                for stat_key in ("mean", "std", "median"):
                    results[f"cos_sim_{stat_key}"] = cos_stats[stat_key]
                    
            coh_stats = self._compute_array_stats(self.pixel_coh_store)
            if coh_stats:
                for stat_key, stat_value in coh_stats.items():
                    results[f"spectral_coh_{stat_key}"] = stat_value

        for name, state in self.param_stats.items():
            results[f"{name}_mean"] = state["mean"]
            results[f"{name}_std"]  = float(np.sqrt(state["m2"] / max(1, state["count"] - 1))) if state["count"] > 1 else 0.0
            if self.deep and "min" in state:
                results[f"{name}_min"] = state["min"]
                results[f"{name}_max"] = state["max"]

        if self.deep and self.gt_param_stats:
            params_per_gaussian = self.gaussian_cfg.params_per_gaussian
            num_gaussians       = max(1, last_channels // params_per_gaussian)
            param_names         = self.gaussian_cfg.make_param_names(num_gaussians)
            
            total_mse = 0.0
            total_mae = 0.0
            
            for i, stats in self.gt_param_stats.items():
                count       = stats["count"]
                mean_gt     = stats["sum_gt"] / max(1, count)
                ss_total    = stats["sum_gt_squared"] - count * mean_gt * mean_gt
                mse         = stats["sum_squared"] / max(1, count)
                mae         = stats["sum_absolute"] / max(1, count)
                r2          = 1.0 - stats["sum_squared"] / (ss_total + 1e-8) if ss_total > 0 else 0.0
                
                name = param_names[i] if i < len(param_names) else f"ch{i}"
                results[f"gt_{name}_mse"] = mse
                results[f"gt_{name}_mae"] = mae
                results[f"gt_{name}_r2"]  = r2
                
                total_mse += mse
                total_mae += mae
                
            results["gt_param_mse_avg"] = total_mse / max(1, len(self.gt_param_stats))
            results["gt_param_mae_avg"] = total_mae / max(1, len(self.gt_param_stats))

        self.metrics.track_results(results, epoch, stage)
        return results
