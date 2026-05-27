from __future__ import annotations

from functools import lru_cache as _lru_cache

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

from configuration.training_config import LossConfig
from tools.param_matcher           import ParamMatcher


@_lru_cache(maxsize=8)
def _cached_gaussian_kernel(size: int, sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g      = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g      = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel[None, None, :, :].contiguous()


class LossComponents:
    @staticmethod
    def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target) ** 2).mean()

    @staticmethod
    def l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(pred - target).mean()

    @staticmethod
    def huber(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
        return F.huber_loss(pred, target, reduction="mean", delta=delta)

    @staticmethod
    def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
        diff = pred - target
        return torch.sqrt((diff * diff + eps * eps).clamp(min=eps * eps)).mean()

    @staticmethod
    def cosine(pred: torch.Tensor, target: torch.Tensor, axis: int) -> torch.Tensor:
        pred_norm   = torch.norm(pred,   dim=axis, keepdim=True)
        target_norm = torch.norm(target, dim=axis, keepdim=True)

        valid = (target_norm > 1e-3).squeeze(axis).float()

        p   = pred   / pred_norm.clamp(min=1e-3)
        t   = target / target_norm.clamp(min=1e-3)
        sim = (p * t).sum(dim=axis).clamp(-1.0, 1.0)
        n   = valid.sum().clamp(min=1.0)

        return ((1.0 - sim) * valid).sum() / n

    @staticmethod
    def spectral_coherence(pred: torch.Tensor, target: torch.Tensor, window: int) -> torch.Tensor:
        B, N, H, W = pred.shape
        p = pred.permute(  0, 2, 3, 1).reshape(B * H * W, 1, N)
        t = target.permute(0, 2, 3, 1).reshape(B * H * W, 1, N)

        pt = F.avg_pool1d(p * t, window, stride=1) * window
        p2 = F.avg_pool1d(p * p, window, stride=1) * window
        t2 = F.avg_pool1d(t * t, window, stride=1) * window

        coh = (pt.abs() / (p2 * t2).clamp(min=1e-16).sqrt()).clamp(0.0, 1.0)
        
        return (1.0 - coh).mean()

    @staticmethod
    def tv(params: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(params[..., 1:, :] - params[..., :-1, :]).mean()
        dy = torch.abs(params[..., :, 1:] - params[..., :, :-1]).mean()
        
        return dx + dy

    @staticmethod
    def _gaussian_window(size: int, sigma: float, dtype, device) -> torch.Tensor:
        coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
        g      = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g      = g / g.sum()
        kernel = g[:, None] * g[None, :]
       
        return kernel[None, None, :, :]

    @staticmethod
    def ssim(
        pred: torch.Tensor,
        target: torch.Tensor,
        window_size: int,
        sigma: float,
        data_range: float,
        k1: float,
        k2: float,
        axis: str = "elevation",
    ) -> torch.Tensor:
        B, N, H, W = pred.shape
        dtype      = pred.dtype
        device     = pred.device

        kernel  = _cached_gaussian_kernel(window_size, sigma, dtype, device)
        padding = window_size // 2
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2

        def conv(z: torch.Tensor) -> torch.Tensor:
            return F.conv2d(z, kernel, padding=padding)

        def ssim_slice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            xy_min = torch.min(x.min(), y.min()).detach()
            xy_max = torch.max(x.max(), y.max()).detach()
            rng    = (xy_max - xy_min).clamp(min=1e-6)
            x      = (x - xy_min) / rng
            y      = (y - xy_min) / rng

            mu_x  = conv(x);  mu_y = conv(y)
            mu_x2 = mu_x * mu_x;  mu_y2 = mu_y * mu_y;  mu_xy = mu_x * mu_y
            sx2   = torch.clamp(conv(x * x) - mu_x2, min=0.0)
            sy2   = torch.clamp(conv(y * y) - mu_y2, min=0.0)
            sxy   = conv(x * y) - mu_xy
            num   = (2.0 * mu_xy + c1) * (2.0 * sxy + c2)
            den   = (mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2)
            return (1.0 - num / den.clamp(min=1e-12)).mean()

        if axis == "elevation":
            xs = pred.permute(1, 0, 2, 3).reshape(-1, 1, H, W)
            ys = target.permute(1, 0, 2, 3).reshape(-1, 1, H, W)
        elif axis == "azimuth":
            xs = pred.permute(2, 0, 1, 3).reshape(-1, 1, N, W)
            ys = target.permute(2, 0, 1, 3).reshape(-1, 1, N, W)
        elif axis == "range":
            xs = pred.permute(3, 0, 1, 2).reshape(-1, 1, N, H)
            ys = target.permute(3, 0, 1, 2).reshape(-1, 1, N, H)
        else:
            raise ValueError(f"ssim_axis must be 'elevation', 'azimuth', or 'range', got '{axis}'")

        return ssim_slice(xs, ys)

    @staticmethod
    def param_l1(
        pred: torch.Tensor,
        gt: torch.Tensor,
        weights: torch.Tensor,
        param_names: list[str],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        diff          = pred - gt
        weighted_diff = weights * torch.abs(diff)
        total         = weighted_diff.mean()
   
        per_param     = {
            name: (weights[:, :, i:i+1] * torch.abs(diff[:, :, i:i+1])).mean()
            for i, name in enumerate(param_names)
            if i < pred.shape[2]
        }
        return total, per_param

    @staticmethod
    def param_huber(
        pred: torch.Tensor,
        gt: torch.Tensor,
        weights: torch.Tensor,
        delta: float,
    ) -> torch.Tensor:
        diff     = pred - gt
        abs_diff = torch.abs(diff)
        val      = torch.where(abs_diff <= delta, 0.5 * diff * diff, delta * (abs_diff - 0.5 * delta))
        return (weights * val).mean()


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
            ("spectral_coh",      cfg.use_spectral_coherence, cfg.weight_spectral_coh,       "weight_spectral_coh"),
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

        self.matcher = ParamMatcher(
            strategy = cfg.param_match,
            logger   = self.logger,
        )

    def reconstruct_gaussians(self, params: torch.Tensor) -> torch.Tensor:
        B, C, H, W = params.shape
        ppg        = self.gaussian_cfg.params_per_gaussian
        assert C % ppg == 0, (f"Gaussian param channels ({C}) must be divisible by {ppg}")

        n_gaussians = C // ppg
    
        p   = params.reshape(B, n_gaussians, ppg, H, W)
        a   = F.softplus(p[:, :, 0:1, :, :], beta=10).clamp(max=1e6)
        mu  = p[:, :, 1:2, :, :]                                
        sig = p[:, :, 2:3, :, :]                                
        x   = self.x_axis.reshape(1, 1, -1, 1, 1)

        sig_safe = F.softplus(sig, beta=100, threshold=20).clamp(min=1e-3)
        sig2     = (sig_safe ** 2).clamp(min=1e-6)

        exponent = ((x - mu) ** 2) / (2.0 * sig2)
        exponent = torch.nan_to_num(exponent, nan=500.0, posinf=500.0, neginf=0.0).clamp(max=500.0)
        curves   = (a * torch.exp(-exponent)).sum(dim=1)
     
        return curves
    
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

        pred, pred_phys, gt, gt_phys = self.matcher.match_torch(pred, pred_phys, gt, gt_phys)

        effective_gaussians = min(num_gaussians, gt_gaussians)
        pred                = pred[:,      :effective_gaussians]
        pred_phys           = pred_phys[:, :effective_gaussians]
        gt                  = gt[:,        :effective_gaussians]
        gt_phys             = gt_phys[:,   :effective_gaussians]

        return pred, pred_phys, gt, gt_phys

    def _param_term(self, pred_gauss, gt_gauss, gt_phys_gauss, pred_phys_gauss, kind):
        pred, pred_phys, gt, gt_phys = self._match_params(pred_gauss, gt_gauss, gt_phys_gauss, pred_phys_gauss)
        cfg     = self.loss_cfg
        w       = torch.tensor(cfg.param_weights, dtype=pred.dtype, device=pred.device)

        if w.numel() < pred.shape[2]:
            w = torch.cat([w, torch.ones(pred.shape[2] - w.numel(), dtype=w.dtype, device=w.device)])

        w       = w[:pred.shape[2]]
        weights = w.reshape(1, 1, -1, 1, 1)

        hard_active          = (gt_phys[:, :, 0:1] > cfg.amp_zero_thr).to(pred.dtype)
        param_mask           = torch.ones_like(pred)
        param_mask[:, :, 1:] = hard_active.expand_as(pred[:, :, 1:])

        if kind == "l1":
            return LossComponents.param_l1(pred, gt, weights * param_mask, ["amp", "mu", "sigma"])

        if kind == "huber":
            return LossComponents.param_huber(pred, gt, weights * param_mask, cfg.param_huber_delta), {}

        raise ValueError(f"Unknown param_term kind: {kind!r}. Expected 'l1' or 'huber'.")

    def __call__(self, pred_params, gt_params):
        cfg = self.loss_cfg
        ppg = self.gaussian_cfg.params_per_gaussian
        C   = pred_params.shape[1]

        with torch.no_grad():
            if self.norm_stats is not None and self.norm_stats.stats.output_stats is not None:
                gt_phys = self.norm_stats.denormalize_output(gt_params.float())
            else:
                gt_phys = gt_params.float()
            exp_curves = self.reconstruct(gt_phys.float())

        if self.norm_stats is not None and self.norm_stats.stats.output_stats is not None:
            pred_params_phys = self.norm_stats.denormalize_output(pred_params.float())
        else:
            pred_params_phys = pred_params.float()

        pred_curves  = self.reconstruct(pred_params_phys.float())
        exp_curves   = exp_curves.float()

        components: dict = {}
        weighted:   dict = {}
        total_loss       = torch.zeros((), dtype=pred_curves.dtype, device=pred_curves.device)

        lc = LossComponents

        if cfg.use_mse_curve:
            val                     = lc.mse(pred_curves, exp_curves)
            components["mse_curve"] = val
            weighted["mse_curve"]   = cfg.eff("weight_mse_curve") * val
            total_loss              = total_loss + weighted["mse_curve"]

        if cfg.use_l1_curve:
            val                    = lc.l1(pred_curves, exp_curves)
            components["l1_curve"] = val
            weighted["l1_curve"]   = cfg.eff("weight_l1_curve") * val
            total_loss             = total_loss + weighted["l1_curve"]

        if cfg.use_huber_curve:
            val                       = lc.huber(pred_curves, exp_curves, cfg.huber_delta)
            components["huber_curve"] = val
            weighted["huber_curve"]   = cfg.eff("weight_huber_curve") * val
            total_loss                = total_loss + weighted["huber_curve"]

        if cfg.use_charbonnier_curve:
            val                             = lc.charbonnier(pred_curves, exp_curves, cfg.charbonnier_eps)
            components["charbonnier_curve"] = val
            weighted["charbonnier_curve"]   = cfg.eff("weight_charbonnier_curve") * val
            total_loss                      = total_loss + weighted["charbonnier_curve"]

        if cfg.use_cosine_curve:
            val                        = lc.cosine(pred_curves, exp_curves, axis=1)
            components["cosine_curve"] = val
            weighted["cosine_curve"]   = cfg.eff("weight_cosine_curve") * val
            total_loss                 = total_loss + weighted["cosine_curve"]

        if cfg.use_spectral_coherence:
            val                        = lc.spectral_coherence(pred_curves, exp_curves, cfg.spectral_coh_window)
            components["spectral_coh"] = val
            weighted["spectral_coh"]   = cfg.eff("weight_spectral_coh") * val
            total_loss                 = total_loss + weighted["spectral_coh"]

        if cfg.use_ssim_curve:
            val = lc.ssim(
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
            val                         = lc.tv(pred_params)
            components["smoothness_tv"] = val
            weighted["smoothness_tv"]   = cfg.eff("weight_smoothness_tv") * val
            total_loss                  = total_loss + weighted["smoothness_tv"]

        return {
            "total_loss" : total_loss,
            "components" : components,
            "weighted"   : weighted,
        }
