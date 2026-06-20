from __future__ import annotations

from itertools import permutations

import numpy as np


class GaussianMatcher:
    def __init__(self, ppg: int = 3, amp_threshold: float = 1e-3, big: float = 1e7, chunk_size: int = 250_000) -> None:
        self.ppg           = ppg
        self.amp_threshold = amp_threshold
        self.big           = big
        self.chunk_size    = chunk_size

    def _channel(self, params: np.ndarray, n_K: int, offset: int) -> np.ndarray:
        return np.stack([params[self.ppg * k + offset] for k in range(n_K)], axis=0).reshape(n_K, -1)

    @staticmethod
    def _best_perm(cost: np.ndarray, perms: np.ndarray, n_K: int) -> np.ndarray:
        idx        = np.arange(n_K)
        perm_costs = np.stack([cost[:, idx, perm].sum(axis=1) for perm in perms], axis=1)
        return perm_costs.argmin(axis=1)

    def assignment(self, params_pred: np.ndarray, params_gt: np.ndarray, n_K: int) -> np.ndarray:
        mu_pred  = self._channel(params_pred, n_K, 1)
        mu_gt    = self._channel(params_gt,   n_K, 1)
        act_pred = self._channel(params_pred, n_K, 0) >= self.amp_threshold
        act_gt   = self._channel(params_gt,   n_K, 0) >= self.amp_threshold

        perms    = np.asarray(list(permutations(range(n_K))))
        n_pixels = mu_pred.shape[1]
        sel      = np.empty((n_pixels, n_K), dtype=np.int64)

        for start in range(0, n_pixels, self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            mp = mu_pred [:, sl].T
            mg = mu_gt   [:, sl].T
            ap = act_pred[:, sl].T
            ag = act_gt  [:, sl].T

            base = np.nan_to_num(np.abs(mp[:, :, None] - mg[:, None, :]), nan=self.big, posinf=self.big)
            pair = ap[:, :, None] & ag[:, None, :]
            cost = np.where(pair, base, self.big)

            sel[sl] = perms[self._best_perm(cost, perms, n_K)]

        return sel

    def aligned_prediction(self, params_pred: np.ndarray, params_gt: np.ndarray, n_K: int) -> np.ndarray:
        ppg  = self.ppg
        H, W = params_pred.shape[-2:]
        sel  = self.assignment(params_pred, params_gt, n_K)

        act_pred = self._channel(params_pred, n_K, 0) >= self.amp_threshold
        n_pixels = sel.shape[0]
        rows     = np.arange(n_pixels)
        aligned  = np.full((n_K, ppg, n_pixels), np.nan, dtype=np.float32)

        for i in range(n_K):
            active = act_pred[i]
            if not active.any():
                continue

            target = sel[active, i]
            block  = np.stack([params_pred[ppg * i + c].reshape(-1)[active] for c in range(ppg)], axis=1)
            aligned[target, :, rows[active]] = block

        return aligned.reshape(n_K * ppg, H, W)
