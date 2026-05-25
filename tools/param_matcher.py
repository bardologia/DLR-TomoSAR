from __future__ import annotations

import itertools
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

        elif strategy == "hungarian":
            B, G_p, ppg, H, W = pred.shape
            G_g = gt.shape[1]
            G   = min(G_p, G_g)

            p_hw      = pred.permute(      0, 3, 4, 1, 2).reshape(B, H * W, G_p, ppg)
            g_hw      = gt.permute(        0, 3, 4, 1, 2).reshape(B, H * W, G_g, ppg)
            p_phys_hw = pred_phys.permute( 0, 3, 4, 1, 2).reshape(B, H * W, G_p, ppg)
            g_phys_hw = gt_phys.permute(   0, 3, 4, 1, 2).reshape(B, H * W, G_g, ppg)

            cost_mat = torch.cdist(p_hw[:, :, :G, :], g_hw[:, :, :G, :])
            cost_mat = torch.nan_to_num(cost_mat, nan=100, posinf=100, neginf=0.0)

            is_active    = gt_phys[:, :, 0] > 1e-3                          
            active_hw    = is_active.permute(0, 2, 3, 1).reshape(B, H * W, G_g)  
            inactive_pen = (~active_hw[:, :, :G]).float() * 100             
            cost_mat     = cost_mat + inactive_pen.unsqueeze(2)               

            perms      = torch.tensor(list(itertools.permutations(range(G))), dtype=torch.long, device=pred.device)
            perm_mask  = torch.nn.functional.one_hot(perms, num_classes=G).to(cost_mat.dtype)
            perm_costs = torch.einsum("bsij,pij->bsp", cost_mat, perm_mask)
            best       = perm_costs.argmin(-1)                             
            best_perm  = perms[best]                                       
          
            pred_perm_e          = best_perm.unsqueeze(-1).expand(B, H * W, G, ppg)
            pred_matched_hw      = torch.gather(p_hw[:, :, :G, :],      dim=2, index=pred_perm_e)
            pred_phys_matched_hw = torch.gather(p_phys_hw[:, :, :G, :], dim=2, index=pred_perm_e)
          
            gt_matched_hw      = g_hw[:, :, :G, :]
            gt_phys_matched_hw = g_phys_hw[:, :, :G, :]

            pred      = pred_matched_hw.reshape(     B, H, W, G, ppg).permute(0, 3, 4, 1, 2)
            gt        = gt_matched_hw.reshape(       B, H, W, G, ppg).permute(0, 3, 4, 1, 2)
            pred_phys = pred_phys_matched_hw.reshape(B, H, W, G, ppg).permute(0, 3, 4, 1, 2)
            gt_phys   = gt_phys_matched_hw.reshape(  B, H, W, G, ppg).permute(0, 3, 4, 1, 2)

        return pred, pred_phys, gt, gt_phys

