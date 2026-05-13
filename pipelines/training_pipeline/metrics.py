import numpy as np
import torch
import torch.nn.functional as F
from configuration.training_config import GaussianConfig


class Metrics:
    def __init__(self, verbose: bool, tracker, logger, x_axis, gaussian_cfg: GaussianConfig):
        self.verbose      = verbose
        self.tracker      = tracker
        self.logger       = logger
        self.x_axis       = x_axis
        self.gaussian_cfg = gaussian_cfg

    def reconstruct_gaussians(self, params: torch.Tensor) -> torch.Tensor:
        C   = params.shape[1]
        ppg = self.gaussian_cfg.params_per_gaussian
        assert C % ppg == 0, (f"Gaussian param channels ({C}) must be divisible by {ppg}")
        
        n_gaussians = C // ppg
        x_axis = self.x_axis.to(params.device)
        x      = x_axis.reshape(1, -1, 1, 1)

        result = torch.zeros(params.shape[0], x_axis.shape[0], params.shape[2], params.shape[3], device=params.device, dtype=params.dtype)

        for i in range(n_gaussians):
            a      = params[:, 3 * i    : 3 * i + 1, :, :]
            mu     = params[:, 3 * i + 1: 3 * i + 2, :, :]
            sig    = params[:, 3 * i + 2: 3 * i + 3, :, :]
            result = result + a * torch.exp(-((x - mu) ** 2) / (2 * sig ** 2 + 1e-8))

        return result

    @staticmethod
    def spectral_coherence(pred_curves: torch.Tensor, exp_curves: torch.Tensor, win: int = 7) -> torch.Tensor:
        p = pred_curves.to(torch.complex64)           
        e = exp_curves.to(torch.complex64)

        kernel = torch.ones(1, 1, win, 1, 1, device=pred_curves.device) / win
        pad    = win // 2

        def _smooth(x: torch.Tensor) -> torch.Tensor:
            return F.conv3d(x.unsqueeze(1), kernel.to(x.dtype), padding=(pad, 0, 0)).squeeze(1)

        num = _smooth(p * torch.conj(e)).abs()      
        den = torch.sqrt(_smooth((p * torch.conj(p)).real) * _smooth((e * torch.conj(e)).real)).clamp(min=1e-8)
        coh = (num / den).mean(dim=1).real            
        return coh.clamp(0.0, 1.0)

    @staticmethod
    def compare_params(pred_params: torch.Tensor, gt_params: torch.Tensor, param_names: list[str]) -> dict:

        n_ch = min(pred_params.shape[1], gt_params.shape[1])
        stats     : dict  = {}
        total_mse : float = 0.0
        total_mae : float = 0.0

        for i in range(n_ch):
            p      = pred_params[:, i]
            g      = gt_params[:, i]
            diff   = p - g
            mse_i  = (diff ** 2).mean().item()
            mae_i  = diff.abs().mean().item()
            ss_res = (diff ** 2).sum().item()
            ss_tot = ((g - g.mean()) ** 2).sum().item()
            r2_i   = 1.0 - ss_res / (ss_tot + 1e-8) if ss_tot > 0 else 0.0

            name = param_names[i] if i < len(param_names) else f"ch{i}"
            stats[f"gt_{name}_mse"] = mse_i
            stats[f"gt_{name}_mae"] = mae_i
            stats[f"gt_{name}_r2"]  = r2_i
            
            total_mse += mse_i
            total_mae += mae_i

        stats["gt_param_mse_avg"] = total_mse / max(1, n_ch)
        stats["gt_param_mae_avg"] = total_mae / max(1, n_ch)
        return stats

    def track_results(self, results: dict, epoch: int, stage: str = "validation"):
        self.tracker.log_scalar(f"curve_mse/{stage}",   results["curve_mse"],   epoch)
        self.tracker.log_scalar(f"curve_mae/{stage}",   results["curve_mae"],   epoch)
        self.tracker.log_scalar(f"curve_rmse/{stage}",  results["curve_rmse"],  epoch)

        self.tracker.log_scalar(f"r2/{stage}",          results["pixel_r2_mean"],   epoch)
        self.tracker.log_scalar(f"r2_overall/{stage}",  results.get("overall_r2", 0.0), epoch)
        self.tracker.log_scalar(f"r2_median/{stage}",   results["pixel_r2_median"], epoch)
        self.tracker.log_scalar(f"r2_min/{stage}",      results["pixel_r2_min"],    epoch)
        
        self.tracker.log_scalar(f"pixel_mse_max/{stage}", results["pixel_mse_max"], epoch)
        self.tracker.log_scalar(f"pixel_mae_max/{stage}", results["pixel_mae_max"], epoch)

        if "cos_sim_mean" in results:
            self.tracker.log_scalar(f"cos_sim/{stage}", results["cos_sim_mean"], epoch)
        if "spectral_coh_mean" in results:
            self.tracker.log_scalar(f"spectral_coh/{stage}",     results["spectral_coh_mean"], epoch)
            self.tracker.log_scalar(f"spectral_coh_min/{stage}", results["spectral_coh_min"],  epoch)
        if "gt_param_mse_avg" in results:
            self.tracker.log_scalar(f"gt_param_mse/{stage}", results["gt_param_mse_avg"], epoch)
            self.tracker.log_scalar(f"gt_param_mae/{stage}", results["gt_param_mae_avg"], epoch)

        _exclude    = ("pixel_r2", "cos_sim", "pixel_mse", "pixel_mae", "sigma", "spectral_coh", "gt_param_mse", "gt_param_mae")
        gauss_names = [k.replace("_mean", "") for k in results if k.endswith("_mean") and k.replace("_mean", "") not in _exclude and not k.startswith("gt_")]
        
        if gauss_names:
            param_means = {name: float(results[f"{name}_mean"]) for name in gauss_names}
            self.tracker.log_metrics(f"params/{stage}", param_means, epoch)

    def calculate(self, epoch, pred_params, exp_curves, stage="validation", gt_params=None, deep: bool = False):
        ppg             = self.gaussian_cfg.params_per_gaussian
        n_gaussians     = pred_params.shape[1] // ppg

        pred_curves = self.reconstruct_gaussians(pred_params)
        diff        = pred_curves - exp_curves
        abs_diff    = torch.abs(diff)

        pixel_mse = (diff ** 2).mean(dim=1)
        pixel_mae = abs_diff.mean(dim=1)
        curve_mse = pixel_mse.mean().item()
        curve_mae = pixel_mae.mean().item()
        curve_rmse = float(np.sqrt(curve_mse))

        ss_res   = (diff ** 2).sum(dim=1)
        exp_mean = exp_curves.mean(dim=1, keepdim=True)
        ss_tot   = ((exp_curves - exp_mean) ** 2).sum(dim=1)
        pixel_r2 = 1 - ss_res / (ss_tot + 1e-8)

        overall_ss_res = torch.sum(diff ** 2).item()
        overall_ss_tot = torch.sum((exp_curves - exp_curves.mean()) ** 2).item()
        overall_r2     = 1 - (overall_ss_res / overall_ss_tot) if overall_ss_tot > 0 else 0

        if deep:
            cos_sim = F.cosine_similarity(pred_curves, exp_curves, dim=1)
            spec_coh = self.spectral_coherence(pred_curves, exp_curves)
        else:
            cos_sim  = None
            spec_coh = None

        param_stats = {}
        gauss_names = self.gaussian_cfg.make_param_names(n_gaussians)
        for i, name in enumerate(gauss_names):
            p                           = pred_params[:, i]
            param_stats[f"{name}_mean"] = p.mean().item()
            param_stats[f"{name}_std"]  = p.std().item()
            if deep:
                param_stats[f"{name}_min"] = p.min().item()
                param_stats[f"{name}_max"] = p.max().item()

        gt_param_stats = {}
        if deep and gt_params is not None:
            gt_param_stats = self.compare_params(pred_params, gt_params, gauss_names)

        if self.verbose:
            self.logger.section(f"[Epoch {epoch + 1}] Curve Fitting — {stage.capitalize()} ({'deep' if deep else 'light'})")
            self.logger.subsection(f"Curve : MSE={curve_mse:.6f}  MAE={curve_mae:.6f}  RMSE={curve_rmse:.6f}")
            self.logger.subsection(f"R²    : Overall={overall_r2:.4f}  Pixel Mean={pixel_r2.mean():.4f}  Median={pixel_r2.median():.4f}  Min={pixel_r2.min():.4f}")
            
            if deep and cos_sim is not None:
                self.logger.subsection(f"CosSim: Mean={cos_sim.mean():.4f}  Median={cos_sim.median():.4f}")
            if deep and spec_coh is not None:
                self.logger.subsection(f"SpCoh : Mean={spec_coh.mean():.4f}  Median={spec_coh.median():.4f}  Min={spec_coh.min():.4f}  Max={spec_coh.max():.4f}")
            if deep and gt_params is not None:
                self.logger.subsection(f"GT Param Comparison (avg MSE={gt_param_stats['gt_param_mse_avg']:.6f}  avg MAE={gt_param_stats['gt_param_mae_avg']:.6f}):")
                for name in gauss_names:
                    self.logger.subsection(f"  {name}: MSE={gt_param_stats.get(f'gt_{name}_mse', 0):.6f} MAE={gt_param_stats.get(f'gt_{name}_mae', 0):.6f} R²={gt_param_stats.get(f'gt_{name}_r2', 0):.4f}")
            self.logger.subsection("")

        results = {
            "curve_mse"  : curve_mse,
            "curve_mae"  : curve_mae,
            "curve_rmse" : curve_rmse,
            "overall_r2" : overall_r2,

            "pixel_r2_mean"   : pixel_r2.mean().item(),
            "pixel_r2_std"    : pixel_r2.std().item(),
            "pixel_r2_median" : pixel_r2.median().item(),
            "pixel_r2_min"    : pixel_r2.min().item(),
            "pixel_r2_max"    : pixel_r2.max().item(),

            "pixel_mse_mean"   : pixel_mse.mean().item(),
            "pixel_mse_std"    : pixel_mse.std().item(),
            "pixel_mse_median" : pixel_mse.median().item(),
            "pixel_mse_max"    : pixel_mse.max().item(),
            "pixel_mse_min"    : pixel_mse.min().item(),

            "pixel_mae_mean"   : pixel_mae.mean().item(),
            "pixel_mae_std"    : pixel_mae.std().item(),
            "pixel_mae_median" : pixel_mae.median().item(),
            "pixel_mae_max"    : pixel_mae.max().item(),
            "pixel_mae_min"    : pixel_mae.min().item(),
        }

        if deep:
            if cos_sim is not None:
                results.update({
                    "cos_sim_mean"   : cos_sim.mean().item(),
                    "cos_sim_std"    : cos_sim.std().item(),
                    "cos_sim_median" : cos_sim.median().item(),
                })
            
            if spec_coh is not None:
                results.update({
                    "spectral_coh_mean"   : spec_coh.mean().item(),
                    "spectral_coh_std"    : spec_coh.std().item(),
                    "spectral_coh_median" : spec_coh.median().item(),
                    "spectral_coh_min"    : spec_coh.min().item(),
                    "spectral_coh_max"    : spec_coh.max().item(),
                })

        results.update(param_stats)
        results.update(gt_param_stats)
        
        self.track_results(results, epoch, stage)
        return results
