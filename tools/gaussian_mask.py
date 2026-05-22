from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from tools.logger import Logger


class GaussianMask:
    def __init__(self, cfg, logger: Optional[Logger] = None) -> None:
        self.amp_zero_thr       = cfg.amp_zero_thr
        self.amp_zero_thr_torch = getattr(cfg, "amp_zero_thr_torch", 1e-7)

        if logger is not None:
            logger.section("[GaussianMask]")
            logger.kv_table({
                "amp_zero_thr":       self.amp_zero_thr,
                "amp_zero_thr_torch": self.amp_zero_thr_torch,
            })

    def suppress_np(self, params_gt : np.ndarray, params_pred : np.ndarray, k : int) -> np.ndarray:
        ch_a   = 3 * k
        ch_sig = 3 * k + 2
        n      = params_gt[ch_a].size

        suppress = np.zeros(n, dtype=bool)

        if ch_a < params_gt.shape[0] and ch_a < params_pred.shape[0]:
            a_gt      = params_gt  [ch_a].reshape(-1)
            a_pred    = params_pred[ch_a].reshape(-1)
            suppress |= (np.abs(a_gt) < self.amp_zero_thr) & (np.abs(a_pred) < self.amp_zero_thr)

        return suppress

    def active_mask_torch(self, gt_phys : torch.Tensor, ppg : int) -> torch.Tensor:
        gt_phys_amp = gt_phys[:, :, 0:1, :, :]
        is_active   = (gt_phys_amp > self.amp_zero_thr_torch).to(gt_phys.dtype)
        mask_list   = [torch.ones_like(is_active)] + [is_active] * (ppg - 1)
        active_mask = torch.cat(mask_list, dim=2)

        return active_mask
