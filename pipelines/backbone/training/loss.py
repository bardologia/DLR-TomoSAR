from __future__ import annotations

import torch

from pipelines.backbone.training.loss_terms import LOSS_TERMS
from tools.data.gaussians     import GaussianClamp, GaussianCurve
from tools.loss.curve_loss    import CurveLoss
from tools.loss.param_loss    import ParamLoss, ParamMatcher
from tools.loss.physical_loss import PhysicalLoss
from tools.sar.tomo_geometry  import TomoGeometry


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

        self.logger.section("[Loss Function]")
        self.logger.kv_table({
            "Sample points":  x_axis.shape[0],
            "Log all losses": self.log_all_losses,
            "Param matching": cfg.param_matching.value,
        })
        self.logger.kv_table(self.geometry.describe(), title="Tomographic Geometry")

        self.log_active_terms(cfg, title="Active Terms")
        self.log_slot_presence_config(cfg, title="Slot-Presence Loss Config")

    @property
    def slot_presence_active(self) -> bool:
        cfg = self.loss_cfg
        return bool(cfg.presence_balance or cfg.use_active_normalization or cfg.amp_focal_gamma > 0.0 or self.log_all_losses)

    def log_active_terms(self, cfg, title: str) -> None:
        active_rows = []

        for term in LOSS_TERMS:
            if getattr(cfg, term.use_flag):
                weight = getattr(cfg, term.weight_key)
                extra  = f"  [axis={cfg.ssim_axis}]" if term.name == "ssim_curve" else ""
                active_rows.append({"Term": term.name, "Weight": f"{weight:g}{extra}"})

        self.logger.metrics_table(active_rows, ["Term", "Weight"], title=title)

    def log_slot_presence_config(self, cfg, title: str) -> None:
        self.logger.kv_table({
            "presence_balance":         cfg.presence_balance,
            "active_weight":            cfg.active_weight,
            "inactive_weight":          cfg.inactive_weight,
            "amp_focal_gamma":          cfg.amp_focal_gamma,
            "amp_focal_delta":          cfg.amp_focal_delta,
            "use_active_normalization": cfg.use_active_normalization,
        }, title=title)

    def set_curriculum(self, complete_cfg) -> None:
        self.loss_cfg       = complete_cfg
        self.loss_generation += 1

        self.logger.subsection("Active loss composition changes at the curriculum swap; train/val loss curves are not comparable across the swap epoch.")
        self.logger.subsection(f"Param matching (curriculum complete): {complete_cfg.param_matching.value}")
        self.log_active_terms(complete_cfg, title="Active Terms (curriculum complete)")
        self.log_slot_presence_config(complete_cfg, title="Slot-Presence Loss Config (curriculum complete)")

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

        presence = ParamLoss.presence_scale(active, cfg.presence_balance, cfg.active_weight, cfg.inactive_weight)
        focal    = ParamLoss.focal_scale(pred[:, :, 0:1], gt[:, :, 0:1], cfg.amp_focal_gamma, cfg.amp_focal_delta)

        elem_w               = weights * param_mask * presence
        elem_w[:, :, 0:1]    = elem_w[:, :, 0:1] * focal

        physical = self._physical_errors(pred_phys, gt_phys, active)

        if kind == "l1":
            total, per_param = ParamLoss.l1(pred, gt, elem_w, ["amp", "mu", "sigma"], cfg.use_active_normalization)
            return total, per_param, physical

        if kind == "huber":
            return ParamLoss.huber(pred, gt, elem_w, cfg.param_huber_delta, cfg.use_active_normalization), {}, physical

        if kind == "mse":
            return ParamLoss.mse(pred, gt, elem_w, cfg.use_active_normalization), {}, physical

        raise ValueError(f"Unknown param_term kind: {kind!r}. Expected 'l1', 'huber', or 'mse'.")

    @torch.no_grad()
    def _physical_errors(self, pred_phys, gt_phys, active) -> dict:
        diff = (pred_phys - gt_phys).abs()

        active_mask = active[:, :, 0]
        n_active    = active_mask.sum().clamp(min=1.0)

        return {
            "amp_mae"     : diff[:, :, 0].mean(),
            "mu_mae_m"    : (diff[:, :, 1] * active_mask).sum() / n_active,
            "sigma_mae_m" : (diff[:, :, 2] * active_mask).sum() / n_active,
        }

    def _match_params(self, pred_gauss, gt_gauss, gt_phys_gauss, pred_phys_gauss):
        batch_size, num_channels, height, width = pred_gauss.shape

        ppg           = self.gaussian_cfg.params_per_gaussian
        num_gaussians = num_channels // ppg

        pred      = pred_gauss.reshape(     batch_size, num_gaussians, ppg, height, width)
        pred_phys = pred_phys_gauss.reshape(batch_size, num_gaussians, ppg, height, width)

        gt_gaussians = gt_gauss.shape[1] // ppg

        gt      = gt_gauss[     :, : gt_gaussians * ppg].reshape(batch_size, gt_gaussians, ppg, height, width)
        gt_phys = gt_phys_gauss[:, : gt_gaussians * ppg].reshape(batch_size, gt_gaussians, ppg, height, width)

        pred, pred_phys, gt, gt_phys = ParamMatcher.match(pred, pred_phys, gt, gt_phys, method=self.loss_cfg.param_matching.value, active_thr=self.loss_cfg.amp_zero_thr)

        effective_gaussians = min(num_gaussians, gt_gaussians)

        pred      = pred[:,      :effective_gaussians]
        pred_phys = pred_phys[:, :effective_gaussians]
        gt        = gt[:,        :effective_gaussians]
        gt_phys   = gt_phys[:,   :effective_gaussians]

        return pred, pred_phys, gt, gt_phys

    def _prepare(self, pred_params, gt_params):
        leaky_slope = self.gaussian_cfg.clamp_leaky_slope

        pred_params_phys = GaussianClamp.apply(
            self.norm_stats.denormalize_output(pred_params.float(), leaky_slope=leaky_slope),
            x_axis      = self.x_axis,
            amp_max     = self.gaussian_cfg.amp_max,
            ppg         = self.gaussian_cfg.params_per_gaussian,
            leaky_slope = leaky_slope,
        )

        pred_params_norm = self.norm_stats.normalize_output(pred_params_phys)

        with torch.no_grad():
            gt_phys    = self.norm_stats.denormalize_output(gt_params.float())
            exp_curves = self.reconstruct(gt_phys)

        pred_curves = self.reconstruct(pred_params_phys.float())
        exp_curves  = exp_curves.float()

        return pred_params_norm, pred_params_phys, gt_phys, pred_curves, exp_curves

    def _term_computes(self, cfg, diff, pred_curves, exp_curves, pred_params_norm, gt_params, gt_phys, pred_params_phys, kz_map) -> dict:
        lc = CurveLoss
        pc = PhysicalLoss

        if kz_map is not None:
            coherence_resyn  = lambda: pc.coherence_resynthesis_pp(pred_curves, exp_curves, kz_map, self.x_axis, self.dx, cfg.physics_floor)
            covariance_match = lambda: pc.covariance_matching_pp(pred_curves, exp_curves, kz_map, self.x_axis, self.dx, cfg.physics_floor)
            capon_cycle      = lambda: pc.capon_cycle_pp(pred_curves, exp_curves, kz_map, self.x_axis, self.dx, cfg.capon_loading, cfg.physics_floor)
        else:
            coherence_resyn  = lambda: pc.coherence_resynthesis(pred_curves, exp_curves, self.geometry.steering, self.dx, cfg.physics_floor)
            covariance_match = lambda: pc.covariance_matching(pred_curves, exp_curves, self.geometry.outer, self.dx, cfg.physics_floor)
            capon_cycle      = lambda: pc.capon_cycle(pred_curves, exp_curves, self.geometry.steering, self.geometry.outer, self.dx, cfg.capon_loading, cfg.physics_floor)

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
            "coherence_resyn":    coherence_resyn,
            "covariance_match":   covariance_match,
            "capon_cycle":        capon_cycle,
            "param_huber":        lambda: self._param_term(pred_params_norm, gt_params, gt_phys, pred_params_phys, "huber")[0],
            "param_mse":          lambda: self._param_term(pred_params_norm, gt_params, gt_phys, pred_params_phys, "mse")[0],
            "smoothness_tv":      lambda: ParamLoss.tv(pred_params_norm),
        }

    @torch.no_grad()
    def _occupancy(self, pred_params_phys, gt_phys) -> dict:
        cfg = self.loss_cfg
        ppg = self.gaussian_cfg.params_per_gaussian
        thr = cfg.amp_zero_thr

        Bp, Cp, Hp, Wp = pred_params_phys.shape
        pred_amp       = pred_params_phys.reshape(Bp, Cp // ppg, ppg, Hp, Wp)[:, :, 0]
        pred_active    = (pred_amp > thr).to(pred_amp.dtype)

        Bg, Cg, Hg, Wg = gt_phys.shape
        gt_amp         = gt_phys.reshape(Bg, Cg // ppg, ppg, Hg, Wg)[:, :, 0]
        gt_active      = (gt_amp > thr).to(gt_amp.dtype)

        pred_slot = pred_active.mean(dim=(0, 2, 3))
        gt_slot   = gt_active.mean(dim=(0, 2, 3))

        out : dict = {}
        out["gt_active_frac"]   = gt_active.mean()
        out["pred_active_frac"] = pred_active.mean()

        for g in range(pred_slot.shape[0]):
            out[f"pred_active_slot{g}"] = pred_slot[g]
        for g in range(gt_slot.shape[0]):
            out[f"gt_active_slot{g}"]   = gt_slot[g]

        pred_count = pred_active.sum(dim=1)
        gt_count   = gt_active.sum(dim=1)

        out["count/exact_frac"] = (pred_count == gt_count).to(pred_amp.dtype).mean()
        out["count/under_frac"] = (pred_count <  gt_count).to(pred_amp.dtype).mean()
        out["count/over_frac"]  = (pred_count >  gt_count).to(pred_amp.dtype).mean()

        for k in range(1, gt_active.shape[1] + 1):
            mask_k = gt_count == k
            denom  = mask_k.sum()
            if denom > 0:
                out[f"count/acc_gt{k}"] = ((pred_count == gt_count) & mask_k).to(pred_amp.dtype).sum() / denom.to(pred_amp.dtype)

        return out

    def __call__(self, pred_output, gt_params, kz_map=None):
        cfg = self.loss_cfg

        pred_params_norm, pred_params_phys, gt_phys, pred_curves, exp_curves = self._prepare(pred_output, gt_params)

        if kz_map is not None:
            kz_map = kz_map.to(device=pred_curves.device, dtype=pred_curves.dtype)

        needs_diff = self.log_all_losses or cfg.use_mse_curve or cfg.use_l1_curve or cfg.use_huber_curve or cfg.use_charbonnier_curve
        diff       = (pred_curves - exp_curves) if needs_diff else None

        computes = self._term_computes(cfg, diff, pred_curves, exp_curves, pred_params_norm, gt_params, gt_phys, pred_params_phys, kz_map)

        components : dict = {}
        monitor    : dict = {}
        occupancy  : dict = {}
        physical   : dict = {}
        total_loss            = torch.zeros((), dtype=pred_curves.dtype, device=pred_curves.device)
        weight_sum:    float  = 0.0

        per_param_l1:  dict   = {}

        for term in LOSS_TERMS:
            is_used = getattr(cfg, term.use_flag)

            if term.name == "param_l1":
                if is_used:
                    val, per_param_l1, physical = self._param_term(pred_params_norm, gt_params, gt_phys, pred_params_phys, "l1")
                elif self.log_all_losses:
                    with torch.no_grad():
                        val, per_param_l1, physical = self._param_term(pred_params_norm, gt_params, gt_phys, pred_params_phys, "l1")
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
                eff_w                 = getattr(cfg, term.weight_key)
                components[term.name] = val
                total_loss            = total_loss + eff_w * val
                weight_sum           += eff_w

            else:
                monitor[f"{term.name}_{term.space}"] = val

        per_param_target = components if cfg.use_param_l1 else monitor
        for pname, pval in per_param_l1.items():
            per_param_target[f"param_l1/{pname}"] = pval

        if self.slot_presence_active:
            occupancy = self._occupancy(pred_params_phys, gt_phys)

        if weight_sum > 0.0:
            total_loss = total_loss / weight_sum

        return {
            "total_loss" : total_loss,
            "components" : components,
            "monitor"    : monitor,
            "occupancy"  : occupancy,
            "physical"   : physical,
        }
