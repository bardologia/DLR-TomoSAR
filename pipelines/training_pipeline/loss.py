import torch
import torch.nn.functional as F
from configuration.training_config import GaussianConfig, LossConfig
from pipelines.training_pipeline.param_mask import MaskedParamLoss


class Loss:
    def __init__(self, x_axis, logger, tracker, reconstruct_fn, gaussian_cfg, loss_cfg = None, norm_stats=None):
        self.x_axis       = x_axis
        self.logger       = logger
        self.tracker      = tracker
        self.reconstruct  = reconstruct_fn
        self.gaussian_cfg = gaussian_cfg
        self.loss_cfg     = loss_cfg if loss_cfg is not None else LossConfig()
        self.norm_stats   = norm_stats

        cfg = self.loss_cfg
        active_terms = [
            ("mse_curve",         cfg.use_mse_curve,          cfg.weight_mse_curve),
            ("l1_curve",          cfg.use_l1_curve,           cfg.weight_l1_curve),
            ("huber_curve",       cfg.use_huber_curve,        cfg.weight_huber_curve),
            ("charbonnier_curve", cfg.use_charbonnier_curve,  cfg.weight_charbonnier_curve),
            ("cosine_curve",      cfg.use_cosine_curve,       cfg.weight_cosine_curve),
            ("spectral_coh",      cfg.use_spectral_coherence, cfg.weight_spectral_coh),
            ("ssim_curve",        cfg.use_ssim_curve,         cfg.weight_ssim_curve),
            ("param_l1",          cfg.use_param_l1,           cfg.weight_param_l1),
            ("param_huber",       cfg.use_param_huber,        cfg.weight_param_huber),
            ("smoothness_tv",     cfg.use_smoothness_tv,      cfg.weight_smoothness_tv),
            ("masked_param",      cfg.use_masked_param,       cfg.weight_masked_param),
        ]

        self.logger.section("[Loss Function]")
        self.logger.subsection(f"Reconstruction over K-Gaussian sum on {len(x_axis)} sample points")
        self.logger.subsection(f"Param matching strategy : {cfg.param_match}")
        self.logger.subsection("Active terms (term : weight):")
        for name, is_used, weight in active_terms:
            if is_used:
                self.logger.subsection(f"  • {name:<20s} weight={weight:g}")
        self.logger.subsection("")

        self.masked_param_fn = (
            MaskedParamLoss(
                gaussian_cfg = gaussian_cfg,
                loss_cfg     = cfg,
                norm_stats   = norm_stats,
                logger       = logger,
            )
            if cfg.use_masked_param
            else None
        )

    @staticmethod
    def _charbonnier(diff: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sqrt(diff * diff + eps * eps).mean()

    @staticmethod
    def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords    = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
        gaussian  = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        gaussian  = gaussian / gaussian.sum()
        kernel_2d = gaussian.unsqueeze(1) * gaussian.unsqueeze(0)             
        return kernel_2d.unsqueeze(0).unsqueeze(0)               

    @staticmethod
    def _ssim_loss(pred_curves: torch.Tensor, exp_curves: torch.Tensor, window_size: int, sigma: float, data_range: float, k1: float, k2: float) -> torch.Tensor:
        batch_size, num_points, height, width = pred_curves.shape
        device, dtype = pred_curves.device, pred_curves.dtype

        x = pred_curves.reshape(batch_size * num_points, 1, height, width)
        y = exp_curves.reshape(batch_size * num_points, 1, height, width)

        kernel  = Loss._gaussian_window(window_size, sigma, device, dtype)
        padding = window_size // 2

        mu_x = F.conv2d(x, kernel, padding=padding)
        mu_y = F.conv2d(y, kernel, padding=padding)

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_x_y  = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, kernel, padding=padding) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, kernel, padding=padding) - mu_y_sq
        sigma_xy   = F.conv2d(x * y, kernel, padding=padding) - mu_x_y

        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2

        numerator   = (2.0 * mu_x_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map    = numerator / denominator.clamp(min=1e-12)

        return 1.0 - ssim_map.mean()

    @staticmethod
    def _tv_loss(params: torch.Tensor) -> torch.Tensor:
        dx = (params[..., 1:, :] - params[..., :-1, :]).abs().mean()
        dy = (params[..., :, 1:] - params[..., :, :-1]).abs().mean()
        return dx + dy

    def _match_params(self, pred_gauss: torch.Tensor, gt_gauss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params_per_gaussian = self.gaussian_cfg.params_per_gaussian
        batch_size, num_channels, height, width = pred_gauss.shape
        num_gaussians = num_channels // params_per_gaussian
        pred = pred_gauss.view(batch_size, num_gaussians, params_per_gaussian, height, width)

        gt_channels  = gt_gauss.shape[1]
        gt_gaussians = gt_channels // params_per_gaussian
        gt           = gt_gauss[:, : gt_gaussians * params_per_gaussian].view(batch_size, gt_gaussians, params_per_gaussian, height, width)

        effective_gaussians = min(num_gaussians, gt_gaussians)
        pred = pred[:, :effective_gaussians]
        gt   = gt[:,   :effective_gaussians]

        strategy = self.loss_cfg.param_match
        if strategy == "index" or effective_gaussians <= 1:
            return pred, gt

        if strategy == "sorted_mu":
            pred_mu    = pred[:, :, 1]                   
            gt_mu      = gt[:,   :, 1]
            pred_index = pred_mu.argsort(dim=1)           
            gt_index   = gt_mu.argsort(dim=1)
            pred = torch.gather(pred, 1, pred_index.unsqueeze(2).expand(-1, -1, params_per_gaussian, -1, -1))
            gt   = torch.gather(gt,   1, gt_index.unsqueeze(2).expand(-1, -1, params_per_gaussian, -1, -1))
            return pred, gt

        return self._match_params(pred_gauss, gt_gauss) if False else (pred, gt)

    def _param_term(self, pred_gauss: torch.Tensor, gt_gauss: torch.Tensor, kind: str) -> torch.Tensor:
        pred_gauss = self.norm_stats.normalize_output(pred_gauss)
        pred, gt   = self._match_params(pred_gauss, gt_gauss)        
        diff       = pred - gt

        weights = torch.tensor(self.loss_cfg.param_weights, device=diff.device, dtype=diff.dtype)
        if weights.numel() < pred.shape[2]:
            pad     = torch.ones(pred.shape[2] - weights.numel(), device=weights.device, dtype=weights.dtype)
            weights = torch.cat([weights, pad])
        
        weights = weights[: pred.shape[2]].view(1, 1, -1, 1, 1)

        if kind == "l1":
            return (weights * diff.abs()).mean()
        if kind == "huber":
            delta    = self.loss_cfg.param_huber_delta
            abs_diff = diff.abs()
            quad     = 0.5 * diff * diff
            linear   = delta * (abs_diff - 0.5 * delta)
            val      = torch.where(abs_diff <= delta, quad, linear)
            return (weights * val).mean()
        
        raise ValueError(f"Unknown param term kind: {kind}")

    def __call__(self, pred_params: torch.Tensor, exp_curves: torch.Tensor, step: int | None, gt_params: torch.Tensor | None = None) -> dict:
        from pipelines.training_pipeline.metrics import Metrics
        
        cfg = self.loss_cfg
        pred_curves = self.reconstruct(pred_params)               
        diff        = pred_curves - exp_curves

        components: dict[str, torch.Tensor] = {}
        weighted:   dict[str, torch.Tensor] = {}
        total_loss  = pred_curves.new_zeros(())

        if cfg.use_mse_curve:
            val = (diff ** 2).mean()
            components["mse_curve"] = val
            weighted["mse_curve"]   = cfg.weight_mse_curve * val
            total_loss = total_loss + weighted["mse_curve"]

        if cfg.use_l1_curve:
            val = diff.abs().mean()
            components["l1_curve"] = val
            weighted["l1_curve"]   = cfg.weight_l1_curve * val
            total_loss = total_loss + weighted["l1_curve"]

        if cfg.use_huber_curve:
            val = F.huber_loss(pred_curves, exp_curves, delta=cfg.huber_delta)
            components["huber_curve"] = val
            weighted["huber_curve"]   = cfg.weight_huber_curve * val
            total_loss = total_loss + weighted["huber_curve"]

        if cfg.use_charbonnier_curve:
            val = self._charbonnier(diff, cfg.charbonnier_eps)
            components["charbonnier_curve"] = val
            weighted["charbonnier_curve"]   = cfg.weight_charbonnier_curve * val
            total_loss = total_loss + weighted["charbonnier_curve"]

        if cfg.use_cosine_curve:
            cos_sim = F.cosine_similarity(pred_curves, exp_curves, dim=1)
            val     = (1.0 - cos_sim).mean()
            components["cosine_curve"] = val
            weighted["cosine_curve"]   = cfg.weight_cosine_curve * val
            total_loss = total_loss + weighted["cosine_curve"]

        if cfg.use_spectral_coherence:
            spectral_coh = Metrics.spectral_coherence(pred_curves, exp_curves, win=cfg.spectral_coh_window)
            val          = (1.0 - spectral_coh).mean()
            components["spectral_coh"] = val
            weighted["spectral_coh"]   = cfg.weight_spectral_coh * val
            total_loss = total_loss + weighted["spectral_coh"]

        if cfg.use_ssim_curve:
            val = self._ssim_loss(
                pred_curves, exp_curves,
                window_size = cfg.ssim_window_size,
                sigma       = cfg.ssim_sigma,
                data_range  = cfg.ssim_data_range,
                k1          = cfg.ssim_k1,
                k2          = cfg.ssim_k2,
            )
            components["ssim_curve"] = val
            weighted["ssim_curve"]   = cfg.weight_ssim_curve * val
            total_loss = total_loss + weighted["ssim_curve"]

        if gt_params is not None:
            if cfg.use_param_l1:
                val = self._param_term(pred_params, gt_params, "l1")
                components["param_l1"] = val
                weighted["param_l1"]   = cfg.weight_param_l1 * val
                total_loss = total_loss + weighted["param_l1"]

            if cfg.use_param_huber:
                val = self._param_term(pred_params, gt_params, "huber")
                components["param_huber"] = val
                weighted["param_huber"]   = cfg.weight_param_huber * val
                total_loss = total_loss + weighted["param_huber"]

        if cfg.use_smoothness_tv:
            val = self._tv_loss(pred_params)
            components["smoothness_tv"] = val
            weighted["smoothness_tv"]   = cfg.weight_smoothness_tv * val
            total_loss = total_loss + weighted["smoothness_tv"]

        if cfg.use_masked_param and gt_params is not None and self.masked_param_fn is not None:
            val = self.masked_param_fn(pred_params, gt_params)
            components["masked_param"] = val
            weighted["masked_param"]   = cfg.weight_masked_param * val
            total_loss = total_loss + weighted["masked_param"]

        if cfg.log_components_every > 0 and step is not None and step % cfg.log_components_every == 0:
            comp_scalars = {key: float(val.item()) for key, val in components.items()}
            self.tracker.log_metrics("train/components", comp_scalars, step)
            self.tracker.log_scalar("train/loss_total", float(total_loss.item()), step)

        output_dict = {
            "total_loss" : total_loss,
            "components" : {key: val.detach() for key, val in components.items()},
            "weighted"   : {key: val.detach() for key, val in weighted.items()},
        }

        if "mse_curve" in components:
            output_dict["mse_loss"] = components["mse_curve"]
            
        return output_dict
