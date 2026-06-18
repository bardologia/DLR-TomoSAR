from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F


class ParamLoss:
    MAX_MATCH_GAUSSIANS = 6
    ACTIVE_AMP_THR      = 1e-3
    DENOM_FLOOR         = 1e-6

    @staticmethod
    def match(
        strategy  : str,
        pred      : torch.Tensor,
        pred_phys : torch.Tensor,
        gt        : torch.Tensor,
        gt_phys   : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if strategy == "sort_gt_by_mu":
            gt, gt_phys = ParamLoss._sort_gt_by_mu(gt, gt_phys)

        elif strategy == "hungarian_active":
            gt, gt_phys         = ParamLoss._sort_gt_by_mu(gt, gt_phys)
            pred, pred_phys     = ParamLoss._assign_pred_to_gt(pred, pred_phys, gt, gt_phys)

        return pred, pred_phys, gt, gt_phys

    @staticmethod
    def _sort_gt_by_mu(gt: torch.Tensor, gt_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gt_phys_amp = gt_phys[:, :, 0]
        gt_mu       = gt[:, :, 1]
        is_active   = gt_phys_amp > ParamLoss.ACTIVE_AMP_THR
        sort_key    = torch.where(is_active, gt_mu, torch.full_like(gt_mu, float("inf")))
        gt_index    = torch.argsort(sort_key, dim=1)
        gt_idx_b    = gt_index[:, :, None, :, :].expand_as(gt)

        return torch.gather(gt, dim=1, index=gt_idx_b), torch.gather(gt_phys, dim=1, index=gt_idx_b)

    @staticmethod
    def _assign_pred_to_gt(pred: torch.Tensor, pred_phys: torch.Tensor, gt: torch.Tensor, gt_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, G, P, H, W = pred.shape

        if gt.shape[1] != G:
            raise ValueError(f"hungarian_active requires equal pred/gt gaussian counts, got {G} and {gt.shape[1]}")

        if G > ParamLoss.MAX_MATCH_GAUSSIANS:
            raise ValueError(f"hungarian_active enumerates G! permutations; G={G} exceeds MAX_MATCH_GAUSSIANS={ParamLoss.MAX_MATCH_GAUSSIANS}")

        active = (gt_phys[:, :, 0] > ParamLoss.ACTIVE_AMP_THR).to(pred.dtype)

        pred_e = pred.permute(0, 3, 4, 1, 2)[:, :, :, :, None, :]
        gt_e   = gt.permute(  0, 3, 4, 1, 2)[:, :, :, None, :, :]
        cost   = (pred_e - gt_e).abs().sum(-1) * active.permute(0, 2, 3, 1)[:, :, :, None, :]

        perms      = list(itertools.permutations(range(G)))
        gt_arange  = torch.arange(G, device=pred.device)
        perm_costs = []

        for perm in perms:
            pidx = torch.tensor(perm, device=pred.device)
            perm_costs.append(cost[:, :, :, pidx, gt_arange].sum(-1))

        best   = torch.stack(perm_costs, dim=-1).argmin(dim=-1)
        perm_t = torch.tensor(perms, device=pred.device)
        chosen = perm_t[best].permute(0, 3, 1, 2)
        idx_b  = chosen[:, :, None, :, :].expand(B, G, P, H, W)

        return torch.gather(pred, dim=1, index=idx_b), torch.gather(pred_phys, dim=1, index=idx_b)

    @staticmethod
    def presence_scale(active: torch.Tensor, balance: bool, active_weight: float, inactive_weight: float) -> torch.Tensor:
        if balance:
            frac            = active.mean()
            active_weight   = 0.5 / (frac + ParamLoss.DENOM_FLOOR)
            inactive_weight = 0.5 / (1.0 - frac + ParamLoss.DENOM_FLOOR)

        return active * active_weight + (1.0 - active) * inactive_weight

    @staticmethod
    def presence_bce(presence_logits: torch.Tensor, active_target: torch.Tensor, balance: bool) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(presence_logits, active_target, reduction="none")

        if not balance:
            return bce.mean()

        frac   = active_target.mean()
        w_pos  = 0.5 / (frac + ParamLoss.DENOM_FLOOR)
        w_neg  = 0.5 / (1.0 - frac + ParamLoss.DENOM_FLOOR)
        weight = active_target * w_pos + (1.0 - active_target) * w_neg

        return (bce * weight).sum() / weight.sum().clamp(min=ParamLoss.DENOM_FLOOR)

    @staticmethod
    def focal_scale(amp_pred: torch.Tensor, amp_gt: torch.Tensor, gamma: float, delta: float) -> torch.Tensor:
        if gamma <= 0.0:
            return torch.ones_like(amp_pred)

        abs_diff = (amp_pred - amp_gt).abs().detach()

        return (abs_diff / (abs_diff + delta + ParamLoss.DENOM_FLOOR)) ** gamma

    @staticmethod
    def l1(
        pred        : torch.Tensor,
        gt          : torch.Tensor,
        weights     : torch.Tensor,
        param_names : list[str],
        active_norm : bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        diff          = pred - gt
        weighted_diff = weights * torch.abs(diff)
        total         = ParamLoss._reduce(weighted_diff, weights, active_norm)

        per_param     = {
            name: ParamLoss._reduce(weights[:, :, i:i+1] * torch.abs(diff[:, :, i:i+1]), weights[:, :, i:i+1], active_norm)
            for i, name in enumerate(param_names)
            if i < pred.shape[2]
        }
        return total, per_param

    @staticmethod
    def huber(
        pred        : torch.Tensor,
        gt          : torch.Tensor,
        weights     : torch.Tensor,
        delta       : float,
        active_norm : bool = False,
    ) -> torch.Tensor:
        diff     = pred - gt
        abs_diff = torch.abs(diff)
        val      = torch.where(abs_diff <= delta, 0.5 * diff * diff, delta * (abs_diff - 0.5 * delta))

        return ParamLoss._reduce(weights * val, weights, active_norm)

    @staticmethod
    def _reduce(weighted: torch.Tensor, weights: torch.Tensor, active_norm: bool) -> torch.Tensor:
        if active_norm:
            return weighted.sum() / weights.sum().clamp(min=ParamLoss.DENOM_FLOOR)

        return weighted.mean()

    @staticmethod
    def tv(params: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(params[..., 1:, :] - params[..., :-1, :]).mean()
        dy = torch.abs(params[..., :, 1:] - params[..., :, :-1]).mean()

        return dx + dy
