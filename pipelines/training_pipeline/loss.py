from __future__ import annotations

import torch

from configuration.training_config                import LossConfig
from pipelines.training_pipeline.loss_components  import LossComponents
from tools.param_matcher                          import ParamMatcher
from tools.gaussian_utils                         import clamp_gaussian_params


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

    def set_curriculum(self, complete_cfg) -> None:
        self.loss_cfg         = complete_cfg
        self.matcher.strategy = complete_cfg.param_match

    def reconstruct_gaussians(self, params: torch.Tensor) -> torch.Tensor:
        B, C, H, W = params.shape
        ppg        = self.gaussian_cfg.params_per_gaussian
        assert C % ppg == 0, (f"Gaussian param channels ({C}) must be divisible by {ppg}")

        n_gaussians = C // ppg
        p           = params.reshape(B, n_gaussians, ppg, H, W)

        a   = p[:, :, 0, :, :]
        mu  = p[:, :, 1, :, :]
        sig = p[:, :, 2, :, :]
        x   = self.x_axis.reshape(1, -1, 1, 1)

        curves = torch.zeros((B, x.shape[1], H, W), dtype=params.dtype, device=params.device)

        for g in range(n_gaussians):
            a_g   = a[:, g:g+1, :, :]
            mu_g  = mu[:, g:g+1, :, :]
            sig_g = sig[:, g:g+1, :, :]

            sig2_g     = sig_g ** 2
            exponent_g = ((x - mu_g) ** 2) / (2.0 * sig2_g)
            curves     = curves + a_g * torch.exp(-exponent_g)

        return curves
    
    def _match_params(self, pred_gauss, gt_gauss, gt_phys_gauss, pred_phys_gauss):
        batch_size, num_channels, height, width = pred_gauss.shape
        num_gaussians = num_channels // 3
       
        pred          = pred_gauss.reshape(     batch_size, num_gaussians, 3, height, width)
        pred_phys     = pred_phys_gauss.reshape(batch_size, num_gaussians, 3, height, width)

        gt_gaussians = gt_gauss.shape[1] // 3
       
        gt           = gt_gauss[     :, : gt_gaussians * 3].reshape(batch_size, gt_gaussians, 3, height, width)
        gt_phys      = gt_phys_gauss[:, : gt_gaussians * 3].reshape(batch_size, gt_gaussians, 3, height, width)

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

        active               = (gt_phys[:, :, 0:1] > cfg.amp_zero_thr).to(pred.dtype)
        param_mask           = torch.ones_like(pred)
        param_mask[:, :, 1:] = active.expand_as(pred[:, :, 1:])

        if kind == "l1":
            return LossComponents.param_l1(pred, gt, weights * param_mask, ["amp", "mu", "sigma"])

        if kind == "huber":
            return LossComponents.param_huber(pred, gt, weights * param_mask, cfg.param_huber_delta), {}

        raise ValueError(f"Unknown param_term kind: {kind!r}. Expected 'l1' or 'huber'.")

    def __call__(self, pred_params, gt_params):
        cfg = self.loss_cfg
        ppg = self.gaussian_cfg.params_per_gaussian
        C   = pred_params.shape[1]

        pred_params_phys = clamp_gaussian_params(
            self.norm_stats.denormalize_output(pred_params.float()),
            x_axis      = self.x_axis,
            amp_max     = self.gaussian_cfg.amp_max,
            ppg         = self.gaussian_cfg.params_per_gaussian,
            leaky_slope = 0.01,
        )
        
        pred_params_norm = self.norm_stats.normalize_output(pred_params_phys)

        with torch.no_grad():
            gt_phys    = self.norm_stats.denormalize_output(gt_params.float())
            exp_curves = self.reconstruct(gt_phys)

        pred_curves  = self.reconstruct(pred_params_phys.float())
        exp_curves   = exp_curves.float()

        components:  dict  = {}
        weighted:    dict  = {}
        total_loss         = torch.zeros((), dtype=pred_curves.dtype, device=pred_curves.device)
        weight_sum:  float = 0.0  

        lc = LossComponents

        needs_diff = cfg.use_mse_curve or cfg.use_l1_curve or cfg.use_huber_curve or cfg.use_charbonnier_curve
        diff       = (pred_curves - exp_curves) if needs_diff else None

        if cfg.use_mse_curve:
            eff_w                           = cfg.eff("weight_mse_curve")
            val                             = lc.mse_diff(diff)
            components["mse_curve"]         = val
            weighted["mse_curve"]           = eff_w * val
            total_loss                      = total_loss + weighted["mse_curve"]
            weight_sum                     += eff_w

        if cfg.use_l1_curve:
            eff_w                           = cfg.eff("weight_l1_curve")
            val                             = lc.l1_diff(diff)
            components["l1_curve"]          = val
            weighted["l1_curve"]            = eff_w * val
            total_loss                      = total_loss + weighted["l1_curve"]
            weight_sum                     += eff_w

        if cfg.use_huber_curve:
            eff_w                           = cfg.eff("weight_huber_curve")
            val                             = lc.huber_diff(diff, cfg.huber_delta)
            components["huber_curve"]       = val
            weighted["huber_curve"]         = eff_w * val
            total_loss                      = total_loss + weighted["huber_curve"]
            weight_sum                     += eff_w

        if cfg.use_charbonnier_curve:
            eff_w                           = cfg.eff("weight_charbonnier_curve")
            val                             = lc.charbonnier_diff(diff, cfg.charbonnier_eps)
            components["charbonnier_curve"] = val
            weighted["charbonnier_curve"]   = eff_w * val
            total_loss                      = total_loss + weighted["charbonnier_curve"]
            weight_sum                     += eff_w

        if cfg.use_cosine_curve:
            eff_w                           = cfg.eff("weight_cosine_curve")
            val                             = lc.cosine(pred_curves, exp_curves, axis=1)
            components["cosine_curve"]      = val
            weighted["cosine_curve"]        = eff_w * val
            total_loss                      = total_loss + weighted["cosine_curve"]
            weight_sum                     += eff_w

        if cfg.use_spectral_coherence:
            eff_w                           = cfg.eff("weight_spectral_coh")
            val                             = lc.spectral_coherence(pred_curves, exp_curves, cfg.spectral_coh_window)
            components["spectral_coh"]      = val
            weighted["spectral_coh"]        = eff_w * val
            total_loss                      = total_loss + weighted["spectral_coh"]
            weight_sum                     += eff_w

        if cfg.use_ssim_curve:
            eff_w                           = cfg.eff("weight_ssim_curve")
            val                             = lc.ssim(pred_curves, exp_curves, cfg)
            components["ssim_curve"]        = val
            weighted["ssim_curve"]          = eff_w * val
            total_loss                      = total_loss + weighted["ssim_curve"]
            weight_sum                     += eff_w

        if cfg.use_param_huber:
            eff_w                           = cfg.eff("weight_param_huber")
            val, _                          = self._param_term(pred_params_norm, gt_params, gt_phys, pred_params_phys, "huber")
            components["param_huber"]       = val
            weighted["param_huber"]         = eff_w * val
            total_loss                      = total_loss + weighted["param_huber"]
            weight_sum                     += eff_w
        
        if cfg.use_smoothness_tv:
            eff_w                           = cfg.eff("weight_smoothness_tv")
            val                             = lc.tv(pred_params_norm)
            components["smoothness_tv"]     = val
            weighted["smoothness_tv"]       = eff_w * val
            total_loss                      = total_loss + weighted["smoothness_tv"]
            weight_sum                     += eff_w

        if cfg.use_param_l1:
            eff_w                           = cfg.eff("weight_param_l1")
            val, per_param                  = self._param_term(pred_params_norm, gt_params, gt_phys, pred_params_phys, "l1")
            components["param_l1"]          = val
            weighted["param_l1"]            = eff_w * val
            total_loss                      = total_loss + weighted["param_l1"]
            weight_sum                     += eff_w

            for pname, pval in per_param.items():
                components[f"param_l1/{pname}"] = pval
                weighted[f"param_l1/{pname}"]   = eff_w * pval

        if weight_sum > 0.0:
            total_loss = total_loss / weight_sum

        return {
            "total_loss" : total_loss,
            "components" : components,
            "weighted"   : weighted,
        }
