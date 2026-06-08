from __future__ import annotations

from collections import namedtuple
from functools   import lru_cache

import torch
import torch.nn.functional as F

from tools.gaussians     import GaussianClamp
from tools.logger        import Logger
from tools.tomo_geometry import TomoGeometry


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


class ParamMatcher:
    def __init__(
        self,
        strategy : str,
        logger   : Logger | None = None,
    ) -> None:
        self.strategy = strategy

        if logger is not None:
            logger.kv_table({"Param match strategy": strategy})

    @classmethod
    def silent(cls, strategy: str) -> ParamMatcher:
        return cls(strategy, logger=None)

    def match_torch(
        self,
        pred      : torch.Tensor,
        pred_phys : torch.Tensor,
        gt        : torch.Tensor,
        gt_phys   : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        strategy = self.strategy

        if strategy == "sort_gt_by_mu":
            gt_phys_amp = gt_phys[:, :, 0]
            gt_mu       = gt[:, :, 1]
            is_active   = gt_phys_amp > 1e-3
            sort_key    = torch.where(is_active, gt_mu, torch.full_like(gt_mu, float("inf")))
            gt_index    = torch.argsort(sort_key, dim=1)
            gt_idx_b    = gt_index[:, :, None, :, :].expand_as(gt)
            gt          = torch.gather(gt,      dim=1, index=gt_idx_b)
            gt_phys     = torch.gather(gt_phys, dim=1, index=gt_idx_b)

        return pred, pred_phys, gt, gt_phys


class LossComponents:
    @staticmethod
    @lru_cache(maxsize=8)
    def gaussian_kernel(size: int, sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
        g      = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g      = g / g.sum()
        kernel = g[:, None] * g[None, :]
        return kernel[None, None, :, :].contiguous()

    @staticmethod
    def mse_diff(diff: torch.Tensor) -> torch.Tensor:
        return (diff * diff).mean()

    @staticmethod
    def l1_diff(diff: torch.Tensor) -> torch.Tensor:
        return diff.abs().mean()

    @staticmethod
    def huber_diff(diff: torch.Tensor, delta: float) -> torch.Tensor:
        abs_diff = diff.abs()
        val      = torch.where(abs_diff <= delta, 0.5 * diff * diff, delta * (abs_diff - 0.5 * delta))
        return val.mean()

    @staticmethod
    def charbonnier_diff(diff: torch.Tensor, eps: float) -> torch.Tensor:
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
    def ssim(
        pred: torch.Tensor,
        target: torch.Tensor,
        cfg,
    ) -> torch.Tensor:
        window_size = cfg.ssim_window_size
        sigma       = cfg.ssim_sigma
        data_range  = cfg.ssim_data_range
        k1          = cfg.ssim_k1
        k2          = cfg.ssim_k2
        axis        = cfg.ssim_axis

        B, N, H, W = pred.shape
        dtype      = pred.dtype
        device     = pred.device

        kernel  = LossComponents.gaussian_kernel(window_size, sigma, dtype, device)
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


class PhysicsComponents:
    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp(min=1.0)

    @staticmethod
    def moment_sums(curves: torch.Tensor, x_axis: torch.Tensor, dx: float) -> tuple:
        x  = x_axis.reshape(1, -1, 1, 1)

        s0 = curves.sum(dim=1) * dx
        s1 = (curves * x).sum(dim=1) * dx
        s2 = (curves * x * x).sum(dim=1) * dx

        return s0, s1, s2

    @staticmethod
    def total_power(pred: torch.Tensor, target: torch.Tensor, dx: float, floor: float) -> torch.Tensor:
        p0 = pred.sum(dim=1) * dx
        t0 = target.sum(dim=1) * dx

        mask = (t0 > floor).to(pred.dtype)
        rel  = (p0 - t0).abs() / t0.clamp(min=floor)

        return PhysicsComponents.masked_mean(rel, mask)

    @staticmethod
    def moments(pred: torch.Tensor, target: torch.Tensor, x_axis: torch.Tensor, dx: float, floor: float, weights: tuple) -> torch.Tensor:
        p0, p1, p2 = PhysicsComponents.moment_sums(pred,   x_axis, dx)
        t0, t1, t2 = PhysicsComponents.moment_sums(target, x_axis, dx)

        x_range = float(x_axis[-1] - x_axis[0])
        mask    = (t0 > floor).to(pred.dtype)

        p0c = p0.clamp(min=floor)
        t0c = t0.clamp(min=floor)

        p_mean = p1 / p0c
        t_mean = t1 / t0c

        p_var = (p2 / p0c - p_mean * p_mean).clamp(min=0.0)
        t_var = (t2 / t0c - t_mean * t_mean).clamp(min=0.0)

        p_spread = torch.sqrt(p_var + 1e-8)
        t_spread = torch.sqrt(t_var + 1e-8)

        mass_term   = (p0 - t0).abs() / t0c
        mean_term   = (p_mean - t_mean).abs() / x_range
        spread_term = (p_spread - t_spread).abs() / x_range

        w_mass, w_mean, w_spread = weights
        w_sum                    = max(w_mass + w_mean + w_spread, 1e-12)

        combined = (w_mass * mass_term + w_mean * mean_term + w_spread * spread_term) / w_sum

        return PhysicsComponents.masked_mean(combined, mask)

    @staticmethod
    def coherence_resynthesis(pred: torch.Tensor, target: torch.Tensor, steering: torch.Tensor, dx: float, floor: float) -> torch.Tensor:
        p0 = pred.sum(dim=1) * dx
        t0 = target.sum(dim=1) * dx

        mask = (t0 > floor).to(pred.dtype)

        gp = torch.einsum("nk,bkhw->bnhw", steering, pred.to(steering.dtype)) * dx
        gt = torch.einsum("nk,bkhw->bnhw", steering, target.to(steering.dtype)) * dx

        gp = gp / p0.clamp(min=floor).unsqueeze(1)
        gt = gt / t0.clamp(min=floor).unsqueeze(1)

        val = ((gp - gt).abs() ** 2).mean(dim=1)

        return PhysicsComponents.masked_mean(val, mask)

    @staticmethod
    def covariance_matching(pred: torch.Tensor, target: torch.Tensor, outer: torch.Tensor, dx: float, floor: float) -> torch.Tensor:
        t0   = target.sum(dim=1) * dx
        mask = (t0 > floor).to(pred.dtype)

        delta = torch.einsum("ijk,bkhw->bijhw", outer, (pred - target).to(outer.dtype)) * dx
        ref   = torch.einsum("ijk,bkhw->bijhw", outer, target.to(outer.dtype)) * dx

        num = (delta.abs() ** 2).sum(dim=(1, 2))
        den = (ref.abs() ** 2).sum(dim=(1, 2)).clamp(min=1e-12)

        return PhysicsComponents.masked_mean(num / den, mask)

    @staticmethod
    def capon_cycle(pred: torch.Tensor, target: torch.Tensor, steering: torch.Tensor, outer: torch.Tensor, dx: float, loading: float, floor: float) -> torch.Tensor:
        n_tracks = steering.shape[0]

        t0   = target.sum(dim=1) * dx
        mask = (t0 > floor).to(pred.dtype)

        cov   = torch.einsum("ijk,bkhw->bhwij", outer, pred.to(outer.dtype)) * dx
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1).real / n_tracks
        eye   = torch.eye(n_tracks, dtype=cov.dtype, device=cov.device)
        cov   = cov + (loading * trace.clamp(min=floor)).unsqueeze(-1).unsqueeze(-1) * eye

        inv   = torch.linalg.inv(cov)
        denom = torch.einsum("ik,bhwij,jk->bhwk", steering.conj(), inv, steering).real
        spec  = (1.0 / denom.clamp(min=1e-12)).permute(0, 3, 1, 2)

        p0     = spec.sum(dim=1) * dx
        spec_n = spec / p0.clamp(min=floor).unsqueeze(1)
        targ_n = target / t0.clamp(min=floor).unsqueeze(1)

        val = ((spec_n - targ_n) ** 2).mean(dim=1)

        return PhysicsComponents.masked_mean(val, mask)


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
            "Param matching": cfg.param_match,
            "Log all losses": self.log_all_losses,
        })
        self.logger.kv_table(self.geometry.describe(), title="Tomographic Geometry")

        self.log_active_terms(cfg, title="Active Terms")

        self.matcher = ParamMatcher(
            strategy = cfg.param_match,
            logger   = self.logger,
        )

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
        self.loss_cfg         = complete_cfg
        self.matcher.strategy = complete_cfg.param_match
        self.loss_generation += 1

        self.logger.subsection("Active loss composition changes at the curriculum swap; train/val loss curves are not comparable across the swap epoch.")
        self.log_active_terms(complete_cfg, title="Active Terms (curriculum complete)")

    def reconstruct_gaussians(self, params: torch.Tensor) -> torch.Tensor:
        B, C, H, W = params.shape
        ppg        = self.gaussian_cfg.params_per_gaussian

        if C % ppg != 0:
            raise ValueError(f"Gaussian param channels ({C}) must be divisible by {ppg}")

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

        pred_curves  = self.reconstruct(pred_params_phys.float())
        exp_curves   = exp_curves.float()

        lc = LossComponents
        pc = PhysicsComponents

        needs_diff = self.log_all_losses or cfg.use_mse_curve or cfg.use_l1_curve or cfg.use_huber_curve or cfg.use_charbonnier_curve
        diff       = (pred_curves - exp_curves) if needs_diff else None

        computes = {
            "mse_curve":          lambda: lc.mse_diff(diff),
            "l1_curve":           lambda: lc.l1_diff(diff),
            "huber_curve":        lambda: lc.huber_diff(diff, cfg.huber_delta),
            "charbonnier_curve":  lambda: lc.charbonnier_diff(diff, cfg.charbonnier_eps),
            "cosine_curve":       lambda: lc.cosine(pred_curves, exp_curves, axis=1),
            "spectral_coh":       lambda: lc.spectral_coherence(pred_curves, exp_curves, cfg.spectral_coh_window),
            "ssim_curve":         lambda: lc.ssim(pred_curves, exp_curves, cfg),
            "total_power_relerr": lambda: pc.total_power(pred_curves, exp_curves, self.dx, cfg.physics_floor),
            "moments":            lambda: pc.moments(pred_curves, exp_curves, self.x_axis, self.dx, cfg.physics_floor, cfg.moments_weights),
            "coherence_resyn":    lambda: pc.coherence_resynthesis(pred_curves, exp_curves, self.geometry.steering, self.dx, cfg.physics_floor),
            "covariance_match":   lambda: pc.covariance_matching(pred_curves, exp_curves, self.geometry.outer, self.dx, cfg.physics_floor),
            "capon_cycle":        lambda: pc.capon_cycle(pred_curves, exp_curves, self.geometry.steering, self.geometry.outer, self.dx, cfg.capon_loading, cfg.physics_floor),
            "param_huber":        lambda: self._param_term(pred_params_norm, gt_params, gt_phys, pred_params_phys, "huber")[0],
            "smoothness_tv":      lambda: lc.tv(pred_params_norm),
        }

        components:    dict  = {}
        weighted:      dict  = {}
        monitor:       dict  = {}
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
                components[term.name]  = val
                weighted[term.name]    = eff_w * val
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
