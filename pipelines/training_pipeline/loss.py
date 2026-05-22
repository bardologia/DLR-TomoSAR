from __future__ import annotations

import torch
import torch.nn.functional as F

from configuration.training_config import LossConfig


class Loss:
    def __init__(self, x_axis, logger, tracker, gaussian_cfg, loss_cfg=None, norm_stats=None):
        self.x_axis       = x_axis
        self.logger       = logger
        self.tracker      = tracker
        self.gaussian_cfg = gaussian_cfg
        self.reconstruct  = self.reconstruct_gaussians
        self.loss_cfg     = loss_cfg if loss_cfg is not None else LossConfig()
        self.norm_stats   = norm_stats

        cfg = self.loss_cfg
        
        active_terms = [
            ("mse_curve",         cfg.use_mse_curve,          cfg.weight_mse_curve,          "weight_mse_curve"),
            ("l1_curve",          cfg.use_l1_curve,           cfg.weight_l1_curve,           "weight_l1_curve"),
            ("huber_curve",       cfg.use_huber_curve,        cfg.weight_huber_curve,        "weight_huber_curve"),
            ("charbonnier_curve", cfg.use_charbonnier_curve,  cfg.weight_charbonnier_curve,  "weight_charbonnier_curve"),
            ("cosine_curve",      cfg.use_cosine_curve,       cfg.weight_cosine_curve,       "weight_cosine_curve"),
            ("ssim_curve",        cfg.use_ssim_curve,         cfg.weight_ssim_curve,         "weight_ssim_curve"),
            ("param_l1",          cfg.use_param_l1,           cfg.weight_param_l1,           "weight_param_l1"),
            ("param_huber",       cfg.use_param_huber,        cfg.weight_param_huber,        "weight_param_huber"),
            ("smoothness_tv",     cfg.use_smoothness_tv,      cfg.weight_smoothness_tv,      "weight_smoothness_tv"),
        ]

        self.logger.section("[Loss Function]")
        self.logger.kv_table({
            "Sample points":  x_axis.shape[0],
            "Param matching": cfg.param_match,
        })

        active_rows = []
        for name, is_used, alpha, w_key in active_terms:
            if is_used:
                eff    = cfg.eff(w_key)
                factor = getattr(cfg.norm, w_key.removeprefix("weight_"), 1.0)
                extra  = f"  [axis={cfg.ssim_axis}]" if name == "ssim_curve" else ""
                active_rows.append({"Term": name, "Alpha": f"{alpha:g}", "Norm": f"{factor:g}", "Eff": f"{eff:g}{extra}"})
        self.logger.metrics_table(active_rows, ["Term", "Alpha", "Norm", "Eff"], title="Active Terms")

    def reconstruct_gaussians(self, params: torch.Tensor) -> torch.Tensor:
        B, C, H, W = params.shape
        ppg        = self.gaussian_cfg.params_per_gaussian
        assert C % ppg == 0, (f"Gaussian param channels ({C}) must be divisible by {ppg}")

        n_gaussians = C // ppg
    
        p   = params.reshape(B, n_gaussians, ppg, H, W)
        a   = F.relu(p[:, :, 0:1, :, :])                       
        mu  = p[:, :, 1:2, :, :]                                
        sig = p[:, :, 2:3, :, :]                                
        x   = self.x_axis.reshape(1, 1, -1, 1, 1)              

        curves = (a * torch.exp(-((x - mu) ** 2) / (2.0 * sig ** 2 + 1e-8))).sum(dim=1)  
     
        return curves
    
    @staticmethod
    def _charbonnier(diff: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sqrt(diff * diff + eps * eps).mean()

    @staticmethod
    def _huber(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
        return F.huber_loss(pred, target, reduction='mean', delta=delta)

    @staticmethod
    def _gaussian_window(window_size: int, sigma: float, dtype, device) -> torch.Tensor:
        coords    = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
        gaussian  = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        gaussian  = gaussian / gaussian.sum()
        kernel_2d = gaussian[:, None] * gaussian[None, :]
      
        return kernel_2d[None, None, :, :]

    @staticmethod
    def _ssim_loss(pred_curves, exp_curves, window_size, sigma, data_range, k1, k2, axis="elevation"):
        batch_size, num_points, height, width = pred_curves.shape
        dtype  = pred_curves.dtype
        device = pred_curves.device

        kernel  = Loss._gaussian_window(window_size, sigma, dtype, device)
        
        padding = window_size // 2
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2

        def conv(z):
            return F.conv2d(z, kernel, padding=padding)

        def ssim_one(x, y):
            mu_x       = conv(x);  mu_y = conv(y)
            mu_x_sq    = mu_x * mu_x;  mu_y_sq = mu_y * mu_y;  mu_x_y = mu_x * mu_y
            sigma_x_sq = torch.clamp(conv(x * x) - mu_x_sq, min=0.0)
            sigma_y_sq = torch.clamp(conv(y * y) - mu_y_sq, min=0.0)
            sigma_xy   = conv(x * y) - mu_x_y
            num        = (2.0 * mu_x_y + c1) * (2.0 * sigma_xy + c2)
            den        = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
            
            return (1.0 - num / torch.clamp(den, min=1e-12)).mean()

        if axis == "elevation":
            x_slices  = pred_curves.permute(1, 0, 2, 3).reshape(-1, 1, height, width)
            y_slices  = exp_curves.permute(1, 0, 2, 3).reshape(-1, 1, height, width)
            ssim_vals = ssim_one(x_slices, y_slices)               
        
        elif axis == "azimuth":
            x_slices  = pred_curves.permute(2, 0, 1, 3).reshape(-1, 1, num_points, width)
            y_slices  = exp_curves.permute(2, 0, 1, 3).reshape(-1, 1, num_points, width)
            ssim_vals = ssim_one(x_slices, y_slices)
        
        elif axis == "range":
            x_slices  = pred_curves.permute(3, 0, 1, 2).reshape(-1, 1, num_points, height)
            y_slices  = exp_curves.permute(3, 0, 1, 2).reshape(-1, 1, num_points, height)
            ssim_vals = ssim_one(x_slices, y_slices)
        
        else:
            raise ValueError(f"ssim_axis must be 'elevation', 'azimuth', or 'range', got '{axis}'")

        return ssim_vals

    @staticmethod
    def _tv_loss(params: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(params[..., 1:, :] - params[..., :-1, :]).mean()
        dy = torch.abs(params[..., :, 1:] - params[..., :, :-1]).mean()
       
        return dx + dy

    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, axis: int) -> torch.Tensor:
        num = (a * b).sum(dim=axis)
        den = torch.sqrt((a * a).sum(dim=axis) + 1e-8) * torch.sqrt((b * b).sum(dim=axis) + 1e-8)
        
        return num / torch.clamp(den, min=1e-8)

    def _match_params(self, pred_gauss, gt_gauss, gt_phys_gauss, pred_phys_gauss):
        ppg           = self.gaussian_cfg.params_per_gaussian
        batch_size, num_channels, height, width = pred_gauss.shape
        num_gaussians = num_channels // ppg
        pred          = pred_gauss.reshape(batch_size, num_gaussians, ppg, height, width)
        pred_phys     = pred_phys_gauss.reshape(batch_size, num_gaussians, ppg, height, width)

        gt_channels  = gt_gauss.shape[1]
        gt_gaussians = gt_channels // ppg
        gt           = gt_gauss[:, : gt_gaussians * ppg].reshape(batch_size, gt_gaussians, ppg, height, width)
        gt_phys      = gt_phys_gauss[:, : gt_gaussians * ppg].reshape(batch_size, gt_gaussians, ppg, height, width)

        strategy = self.loss_cfg.param_match

        if strategy == "sorted_mu":
            gt_phys_amp  = gt_phys[:, :, 0]
            is_active    = gt_phys_amp > 1e-7
            gt_mu_masked = torch.where(is_active, gt[:, :, 1], torch.full_like(gt[:, :, 1], float('inf')))
            pred_index   = torch.argsort(pred_phys[:, :, 1], dim=1)
            gt_index     = torch.argsort(gt_mu_masked, dim=1)
            pred_idx_b   = pred_index[:, :, None, :, :].expand_as(pred)
            gt_idx_b     = gt_index[:, :, None, :, :].expand_as(gt)
            pred         = torch.gather(pred, dim=1, index=pred_idx_b)
            pred_phys    = torch.gather(pred_phys, dim=1, index=pred_idx_b)
            gt           = torch.gather(gt, dim=1, index=gt_idx_b)
            gt_phys      = torch.gather(gt_phys, dim=1, index=gt_idx_b)

        elif strategy == "sorted_a":
            pred_a     = pred[:, :, 0]
            gt_a       = gt[:,   :, 0]
            pred_index = torch.argsort(pred_a, dim=1, descending=True)
            gt_index   = torch.argsort(gt_a,   dim=1, descending=True)
            pred_idx_b = pred_index[:, :, None, :, :].expand_as(pred)
            gt_idx_b   = gt_index[:,   :, None, :, :].expand_as(gt)
            pred       = torch.gather(pred, dim=1, index=pred_idx_b)
            pred_phys  = torch.gather(pred_phys, dim=1, index=pred_idx_b)
            gt         = torch.gather(gt,   dim=1, index=gt_idx_b)
            gt_phys    = torch.gather(gt_phys, dim=1, index=gt_idx_b)

        elif strategy == "sort_gt_by_mu":
            gt_phys_amp = gt_phys[:, :, 0]
            gt_mu       = gt[:, :, 1]
            is_active   = gt_phys_amp > 1e-7
            sort_key    = torch.where(is_active, gt_mu, torch.full_like(gt_mu, float('inf')))
            gt_index    = torch.argsort(sort_key, dim=1)
            gt_idx_b    = gt_index[:, :, None, :, :].expand_as(gt)
            gt          = torch.gather(gt, dim=1, index=gt_idx_b)
            gt_phys     = torch.gather(gt_phys, dim=1, index=gt_idx_b)

        effective_gaussians = min(num_gaussians, gt_gaussians)
        pred                = pred[:, :effective_gaussians]
        pred_phys           = pred_phys[:, :effective_gaussians]
        gt                  = gt[:, :effective_gaussians]
        gt_phys             = gt_phys[:, :effective_gaussians]

        return pred, pred_phys, gt, gt_phys

    def _param_term(self, pred_gauss, gt_gauss, gt_phys_gauss, pred_phys_gauss, kind):
        pred, pred_phys, gt, gt_phys = self._match_params(pred_gauss, gt_gauss, gt_phys_gauss, pred_phys_gauss)

        diff = pred - gt
        cfg  = self.loss_cfg

        weights = torch.tensor(self.loss_cfg.param_weights, dtype=diff.dtype, device=diff.device)
        if weights.numel() < pred.shape[2]:
            pad     = torch.ones(pred.shape[2] - weights.numel(), dtype=weights.dtype, device=weights.device)
            weights = torch.cat([weights, pad])

        weights = weights[: pred.shape[2]].reshape(1, 1, -1, 1, 1)

        if kind == "l1":
            weighted_diff = weights * torch.abs(diff)
            total         = weighted_diff.mean()
            param_names   = ["amp", "mu", "sigma"]
            per_param     = {pname: (weights[:, :, i:i+1] * torch.abs(diff[:, :, i:i+1])).mean() for i, pname in enumerate(param_names) if i < pred.shape[2]}
            return total, per_param

        if kind == "huber":
            delta        = self.loss_cfg.param_huber_delta
            abs_diff     = torch.abs(diff)
            quad         = 0.5 * diff * diff
            linear       = delta * (abs_diff - 0.5 * delta)
            val          = torch.where(abs_diff <= delta, quad, linear)
            weighted_val = weights * val
            return weighted_val.mean(), {}

        raise ValueError(f"Unknown param term kind: {kind}")

    def __call__(self, pred_params, gt_params):
        cfg = self.loss_cfg
        ppg = self.gaussian_cfg.params_per_gaussian
        C   = pred_params.shape[1]

        with torch.no_grad():
            if self.norm_stats is not None and self.norm_stats.stats.output_stats is not None:
                gt_phys = self.norm_stats.denormalize_output(gt_params)
            else:
                gt_phys = gt_params
            exp_curves = self.reconstruct(gt_phys)

        if self.norm_stats is not None and self.norm_stats.stats.output_stats is not None:
            pred_params_phys = self.norm_stats.denormalize_output(pred_params)
        else:
            pred_params_phys = pred_params

        pred_curves  = self.reconstruct(pred_params_phys)
        diff         = pred_curves - exp_curves

        components: dict = {}
        weighted:   dict = {}
        total_loss       = torch.zeros((), dtype=pred_curves.dtype, device=pred_curves.device)

        if cfg.use_mse_curve:
            val                     = (diff ** 2).mean()
            components["mse_curve"] = val
            weighted["mse_curve"]   = cfg.eff("weight_mse_curve") * val
            total_loss              = total_loss + weighted["mse_curve"]

        if cfg.use_l1_curve:
            val                    = torch.abs(diff).mean()
            components["l1_curve"] = val
            weighted["l1_curve"]   = cfg.eff("weight_l1_curve") * val
            total_loss             = total_loss + weighted["l1_curve"]

        if cfg.use_huber_curve:
            val                       = self._huber(pred_curves, exp_curves, cfg.huber_delta)
            components["huber_curve"] = val
            weighted["huber_curve"]   = cfg.eff("weight_huber_curve") * val
            total_loss                = total_loss + weighted["huber_curve"]

        if cfg.use_charbonnier_curve:
            val                             = self._charbonnier(diff, cfg.charbonnier_eps)
            components["charbonnier_curve"] = val
            weighted["charbonnier_curve"]   = cfg.eff("weight_charbonnier_curve") * val
            total_loss                      = total_loss + weighted["charbonnier_curve"]

        if cfg.use_cosine_curve:
            cos_sim                    = self._cosine_similarity(pred_curves, exp_curves, axis=1)
            val                        = (1.0 - cos_sim).mean()
            components["cosine_curve"] = val
            weighted["cosine_curve"]   = cfg.eff("weight_cosine_curve") * val
            total_loss                 = total_loss + weighted["cosine_curve"]

        if cfg.use_ssim_curve:
            val = self._ssim_loss(
                pred_curves, exp_curves,
                window_size = cfg.ssim_window_size,
                sigma       = cfg.ssim_sigma,
                data_range  = cfg.ssim_data_range,
                k1          = cfg.ssim_k1,
                k2          = cfg.ssim_k2,
                axis        = cfg.ssim_axis,
            )
            components["ssim_curve"] = val
            weighted["ssim_curve"]   = cfg.eff("weight_ssim_curve") * val
            total_loss               = total_loss + weighted["ssim_curve"]

        if cfg.use_param_l1:
            val, per_param         = self._param_term(pred_params, gt_params, gt_phys, pred_params_phys, "l1")
            components["param_l1"] = val
            weighted["param_l1"]   = cfg.eff("weight_param_l1") * val
            total_loss             = total_loss + weighted["param_l1"]

            for pname, pval in per_param.items():
                components[f"param_l1/{pname}"] = pval
                weighted[f"param_l1/{pname}"]   = cfg.eff("weight_param_l1") * pval

        if cfg.use_param_huber:
            val, _                    = self._param_term(pred_params, gt_params, gt_phys, pred_params_phys, "huber")
            components["param_huber"] = val
            weighted["param_huber"]   = cfg.eff("weight_param_huber") * val
            total_loss                = total_loss + weighted["param_huber"]

        if cfg.use_smoothness_tv:
            val                         = self._tv_loss(pred_params)
            components["smoothness_tv"] = val
            weighted["smoothness_tv"]   = cfg.eff("weight_smoothness_tv") * val
            total_loss                  = total_loss + weighted["smoothness_tv"]

        return {
            "total_loss" : total_loss,
            "components" : components,
            "weighted"   : weighted,
        }
