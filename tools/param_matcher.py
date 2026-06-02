from __future__ import annotations

import torch

from tools.logger import Logger


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

