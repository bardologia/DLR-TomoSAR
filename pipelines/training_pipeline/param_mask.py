from __future__ import annotations

import torch
from configuration.training_config import GaussianConfig, LossConfig


class MaskedParamLoss:
    def __init__(
        self,
        gaussian_cfg : GaussianConfig,
        loss_cfg     : LossConfig,
        norm_stats   = None,
        logger       = None,
    ) -> None:
        self.gaussian_cfg = gaussian_cfg
        self.loss_cfg     = loss_cfg
        self.norm_stats   = norm_stats
        self.logger       = logger

        self.kind        : str   = loss_cfg.masked_param_kind
        self.huber_delta : float = loss_cfg.masked_param_huber_delta
        self.a_ths       : float = loss_cfg.masked_param_a_ths
        self.w_ths       : float = loss_cfg.masked_param_w_ths
        self.percent     : float = loss_cfg.masked_param_percent

        self.upper_bound_w: float = self.w_ths * (1.0 + self.percent)

        if self.kind not in ("mse", "l1", "huber"):
            raise ValueError(f"MaskedParamLoss: unknown kind '{self.kind}'. Use 'mse', 'l1', or 'huber'.")

        if logger is not None:
            logger.subsection(
                f"  MaskedParamLoss  kind={self.kind}  a_ths={self.a_ths}"
                f"  w_ths={self.w_ths}  percent={self.percent}"
                f"  upper_bound_w={self.upper_bound_w}"
                + (f"  huber_delta={self.huber_delta}" if self.kind == "huber" else "")
            )

    def _elem_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        if self.kind == "mse":
            return diff ** 2
        if self.kind == "l1":
            return diff.abs()
        abs_diff = diff.abs()
        delta    = self.huber_delta
        return torch.where(abs_diff <= delta, 0.5 * diff * diff, delta * (abs_diff - 0.5 * delta))

    def _masked_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        elem = self._elem_loss(pred, target)
        return (elem * mask).sum() / (mask.sum() + 1e-8)

    def _unpack_component(self, params: torch.Tensor, idx: int):
        ppg = self.gaussian_cfg.params_per_gaussian
        a   = params[:, ppg * idx    ]
        mu  = params[:, ppg * idx + 1]
        sig = params[:, ppg * idx + 2]
        return a, mu, sig

    def _active_mask(self, pred_norm: torch.Tensor, idx: int) -> torch.Tensor:
        a, _, sig = self._unpack_component(pred_norm, idx)
        return ((a > self.a_ths) & (sig >= self.upper_bound_w)).float()

    def __call__(
        self,
        pred_params : torch.Tensor,
        gt_params   : torch.Tensor,
    ) -> torch.Tensor:
        ppg         = self.gaussian_cfg.params_per_gaussian
        n_pred      = pred_params.shape[1] // ppg
        n_gt        = gt_params.shape[1]   // ppg
        n_gaussians = min(n_pred, n_gt)

        if n_gaussians < 1:
            return pred_params.new_zeros(())

        if self.norm_stats is not None:
            pred_norm = self.norm_stats.normalize_output(pred_params)
        else:
            pred_norm = pred_params

        total = pred_params.new_zeros(())

        for i in range(n_gaussians):
            a_p, mu_p, sig_p = self._unpack_component(pred_norm,  i)
            a_t, mu_t, sig_t = self._unpack_component(gt_params,  i)

            if i == 0:
                mask = torch.ones_like(a_p)
            else:
                mask = self._active_mask(pred_norm, i)

            total = total + (
                self._masked_loss(a_p,   a_t,   mask) +
                self._masked_loss(mu_p,  mu_t,  mask) +
                self._masked_loss(sig_p, sig_t, mask)
            )

        return total
