from __future__ import annotations

import torch


class ParamLoss:
    @staticmethod
    def match(
        strategy  : str,
        pred      : torch.Tensor,
        pred_phys : torch.Tensor,
        gt        : torch.Tensor,
        gt_phys   : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    @staticmethod
    def l1(
        pred        : torch.Tensor,
        gt          : torch.Tensor,
        weights     : torch.Tensor,
        param_names : list[str],
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
    def huber(
        pred    : torch.Tensor,
        gt      : torch.Tensor,
        weights : torch.Tensor,
        delta   : float,
    ) -> torch.Tensor:
        diff     = pred - gt
        abs_diff = torch.abs(diff)
        val      = torch.where(abs_diff <= delta, 0.5 * diff * diff, delta * (abs_diff - 0.5 * delta))
        return (weights * val).mean()

    @staticmethod
    def tv(params: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(params[..., 1:, :] - params[..., :-1, :]).mean()
        dy = torch.abs(params[..., :, 1:] - params[..., :, :-1]).mean()

        return dx + dy
