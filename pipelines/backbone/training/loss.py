from __future__ import annotations

from collections import namedtuple

import torch

from tools.data.gaussians     import GaussianClamp, GaussianCurve
from tools.loss.curve_loss    import CurveLoss
from tools.loss.param_loss    import ParamLoss
from tools.loss.physical_loss import PhysicalLoss
from tools.sar.tomo_geometry  import TomoGeometry


LossTerm = namedtuple("LossTerm", ["name", "use_flag", "weight_key", "space"])

LOSS_TERMS = (
    LossTerm("mse_curve",          "use_mse_curve",          "weight_mse_curve",          "denorm"),
    LossTerm("l1_curve",           "use_l1_curve",           "weight_l1_curve",           "denorm"),
    LossTerm("huber_curve",        "use_huber_curve",        "weight_huber_curve",        "denorm"),
    LossTerm("charbonnier_curve",  "use_charbonnier_curve",  "weight_charbonnier_curve",  "denorm"),
    LossTerm("cosine_curve",       "use_cosine_curve",       "weight_cosine_curve",       "denorm"),
    LossTerm("spectral_coh",       "use_spectral_coherence", "weight_spectral_coh",       "denorm"),
    LossTerm("ssim_curve",         "use_ssim_curve",         "weight_ssim_curve",         "denorm"),
    LossTerm("total_power_relerr", "use_total_power",        "weight_total_power",        "denorm"),
    LossTerm("moments",            "use_moments",            "weight_moments",            "denorm"),
    LossTerm("coherence_resyn",    "use_coherence_resyn",    "weight_coherence_resyn",    "denorm"),
    LossTerm("covariance_match",   "use_covariance_match",   "weight_covariance_match",   "denorm"),
    LossTerm("capon_cycle",        "use_capon_cycle",        "weight_capon_cycle",        "denorm"),
    LossTerm("param_huber",        "use_param_huber",        "weight_param_huber",        "norm"),
    LossTerm("smoothness_tv",      "use_smoothness_tv",      "weight_smoothness_tv",      "norm"),
    LossTerm("param_l1",           "use_param_l1",           "weight_param_l1",           "norm"),
)


class Loss:
    def __init__(self, x_axis, logger, tracker, gaussian_cfg, loss_cfg, norm_stats, geometry_cfg, log_all_losses=False):
        self.x_axis          = x_axis
        self.logger          = logger
        self.tracker         = tracker
        self.gaussian_cfg    = gaussian_cfg
        self.reconstruct     = self.reconstruct_gaussians
        self.loss_cfg        = loss_cfg
        self.norm_stats      = norm_stats
        self.geometry_cfg    = geometry_cfg
        self.geometry        = TomoGeometry(self.geometry_cfg, x_axis)
        self.dx              = float(x_axis[1] - x_axis[0])
        self.log_all_losses  = log_all_losses
        self.loss_generation = 0

        cfg = self.loss_cfg

        self.match_strategy = cfg.param_match

        self.logger.section("[Loss Function]")
        self.logger.kv_table({
            "Sample points":  x_axis.shape[0],
            "Param matching": cfg.param_match,
            "Log all losses": self.log_all_losses,
        })
        self.logger.kv_table(self.geometry.describe(), title="Tomographic Geometry")

        self.log_active_terms(cfg, title="Active Terms")

    def log_active_terms(self, cfg, title: str) -> None:
        active_rows = []

        for term in LOSS_TERMS:
            if getattr(cfg, term.use_flag):
                alpha  = getattr(cfg, term.weight_key)
                eff    = cfg.eff(term.weight_key)
                factor = getattr(cfg.norm, term.weight_key.removeprefix("weight_"), 1.0)
                extra  = f"  [axis={cfg.ssim_axis}]" if term.name == "ssim_curve" else ""
                active_rows.append({"Term": term.name, "Alpha": f"{alpha:g}", "Norm": f"{factor:g}", "Eff": f"{eff:g}{extra}"})

        self.logger.metrics_table(active_rows, ["Term", "Alpha", "Norm", "Eff"], title=title)

    def set_curriculum(self, complete_cfg) -> None:
        self.loss_cfg       = complete_cfg
        self.match_strategy = complete_cfg.param_match
        self.loss_generation += 1

        self.logger.subsection("Active loss composition changes at the curriculum swap; train/val loss curves are not comparable across the swap epoch.")
        self.log_active_terms(complete_cfg, title="Active Terms (curriculum complete)")

    def reconstruct_gaussians(self, params: torch.Tensor) -> torch.Tensor:
        return GaussianCurve.reconstruct(params, self.x_axis, self.gaussian_cfg.params_per_gaussian)

    def _param_term(self, pred_gauss, gt_gauss, gt_phys_gauss, pred_phys_gauss, kind):
        pred, pred_phys, gt, gt_phys = self._match_params(pred_gauss, gt_gauss, gt_phys_gauss, pred_phys_gauss)
        cfg = self.loss_cfg
        w   = torch.tensor(cfg.param_weights, dtype=pred.dtype, device=pred.device)

        if w.numel() < pred.shape[2]:
            w = torch.cat([w, torch.ones(pred.shape[2] - w.numel(), dtype=w.dtype, device=w.device)])

        w       = w[:pred.shape[2]]
        weights = w.reshape(1, 1, -1, 1, 1)

        active     = (gt_phys[:, :, 0:1] > cfg.amp_zero_thr).to(pred.dtype)
        param_mask = torch.ones_like(pred)
        param_mask[:, :, 1:] = active.expand_as(pred[:, :, 1:])

        if kind == "l1":
            return ParamLoss.l1(pred, gt, weights * param_mask, ["amp", "mu", "sigma"])

        if kind == "huber":
            return ParamLoss.huber(pred, gt, weights * param_mask, cfg.param_huber_delta), {}

        raise ValueError(f"Unknown param_term kind: {kind!r}. Expected 'l1' or 'huber'.")

    def _match_params(self, pred_gauss, gt_gauss, gt_phys_gauss, pred_phys_gauss):
        batch_size, num_channels, height, width = pred_gauss.shape
        num_gaussians = num_channels // 3

        pred      = pred_gauss.reshape(     batch_size, num_gaussians, 3, height, width)
        pred_phys = pred_phys_gauss.reshape(batch_size, num_gaussians, 3, height, width)

        gt_gaussians = gt_gauss.shape[1] // 3

        gt      = gt_gauss[     :, : gt_gaussians * 3].reshape(batch_size, gt_gaussians, 3, height, width)
        gt_phys = gt_phys_gauss[:, : gt_gaussians * 3].reshape(batch_size, gt_gaussians, 3, height, width)

        pred, pred_phys, gt, gt_phys = ParamLoss.match(self.match_strategy, pred, pred_phys, gt, gt_phys)

        effective_gaussians = min(num_gaussians, gt_gaussians)

        pred      = pred[:,      :effective_gaussians]
        pred_phys = pred_phys[:, :effective_gaussians]
        gt        = gt[:,        :effective_gaussians]
        gt_phys   = gt_phys[:,   :effective_gaussians]

        return pred, pred_phys, gt, gt_phys

    def _prepare(self, pred_params, gt_params):
        pred_params_phys = GaussianClamp.apply(
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

        pred_curves = self.reconstruct(pred_params_phys.float())
        exp_curves  = exp_curves.float()

        return pred_params_norm, pred_params_phys, gt_phys, pred_curves, exp_curves

    def _term_computes(self, cfg, diff, pred_curves, exp_curves, pred_params_norm, gt_params, gt_phys, pred_params_phys) -> dict:
        lc = CurveLoss
        pc = PhysicalLoss

        return {
            "mse_curve":          lambda: lc.mse_diff(diff),
            "l1_curve":           lambda: lc.l1_diff(diff),
            "huber_curve":        lambda: lc.huber_diff(diff, cfg.huber_delta),
            "charbonnier_curve":  lambda: lc.charbonnier_diff(diff, cfg.charbonnier_eps),
            "cosine_curve":       lambda: lc.cosine(pred_curves, exp_curves, axis=1),
            "spectral_coh":       lambda: lc.spectral_coherence(pred_curves, exp_curves, cfg.spectral_coh_window),
            "ssim_curve":         lambda: lc.ssim(pred_curves, exp_curves, cfg.ssim_window_size, cfg.ssim_sigma, cfg.ssim_data_range, cfg.ssim_k1, cfg.ssim_k2, cfg.ssim_axis),
            "total_power_relerr": lambda: pc.total_power(pred_curves, exp_curves, self.dx, cfg.physics_floor),
            "moments":            lambda: pc.moments(pred_curves, exp_curves, self.x_axis, self.dx, cfg.physics_floor, cfg.moments_weights),
            "coherence_resyn":    lambda: pc.coherence_resynthesis(pred_curves, exp_curves, self.geometry.steering, self.dx, cfg.physics_floor),
            "covariance_match":   lambda: pc.covariance_matching(pred_curves, exp_curves, self.geometry.outer, self.dx, cfg.physics_floor),
            "capon_cycle":        lambda: pc.capon_cycle(pred_curves, exp_curves, self.geometry.steering, self.geometry.outer, self.dx, cfg.capon_loading, cfg.physics_floor),
            "param_huber":        lambda: self._param_term(pred_params_norm, gt_params, gt_phys, pred_params_phys, "huber")[0],
            "smoothness_tv":      lambda: ParamLoss.tv(pred_params_norm),
        }

    def __call__(self, pred_params, gt_params):
        cfg = self.loss_cfg

        pred_params_norm, pred_params_phys, gt_phys, pred_curves, exp_curves = self._prepare(pred_params, gt_params)

        needs_diff = self.log_all_losses or cfg.use_mse_curve or cfg.use_l1_curve or cfg.use_huber_curve or cfg.use_charbonnier_curve
        diff       = (pred_curves - exp_curves) if needs_diff else None

        computes = self._term_computes(cfg, diff, pred_curves, exp_curves, pred_params_norm, gt_params, gt_phys, pred_params_phys)

        components : dict = {}
        weighted   : dict = {}
        monitor    : dict = {}
        total_loss            = torch.zeros((), dtype=pred_curves.dtype, device=pred_curves.device)
        weight_sum:    float  = 0.0

        per_param_l1:  dict   = {}

        for term in LOSS_TERMS:
            is_used = getattr(cfg, term.use_flag)

            if term.name == "param_l1":
                if is_used:
                    val, per_param_l1 = self._param_term(pred_params_norm, gt_params, gt_phys, pred_params_phys, "l1")
                elif self.log_all_losses:
                    with torch.no_grad():
                        val, per_param_l1 = self._param_term(pred_params_norm, gt_params, gt_phys, pred_params_phys, "l1")
                else:
                    continue
            else:
                compute = computes[term.name]

                if is_used:
                    val = compute()
                elif self.log_all_losses:
                    with torch.no_grad():
                        val = compute()
                else:
                    continue

            if is_used:
                eff_w                  = cfg.eff(term.weight_key)
                components[term.name] = val
                weighted[term.name]   = eff_w * val
                total_loss             = total_loss + weighted[term.name]
                weight_sum            += eff_w

                if self.log_all_losses:
                    monitor[f"{term.name}_{term.space}"] = val

            else:
                monitor[f"{term.name}_{term.space}"] = val

        if cfg.use_param_l1:
            eff_w = cfg.eff("weight_param_l1")
            for pname, pval in per_param_l1.items():
                components[f"param_l1/{pname}"] = pval
                weighted[f"param_l1/{pname}"]   = eff_w * pval

        if self.log_all_losses:
            for pname, pval in per_param_l1.items():
                monitor[f"param_l1/{pname}_norm"] = pval

        if weight_sum > 0.0:
            total_loss = total_loss / weight_sum

        return {
            "total_loss" : total_loss,
            "components" : components,
            "weighted"   : weighted,
            "monitor"    : monitor,
        }
