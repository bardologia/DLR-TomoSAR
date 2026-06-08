from __future__ import annotations

import numpy as np


class ProfilePreprocessor:
    @staticmethod
    def apply(profile : np.ndarray, threshold_factor : float, truncation_index : int) -> np.ndarray:
        out    = np.asarray(profile)
        copied = False

        if threshold_factor > 0.0:
            out    = np.where(out > out.max(axis=0, keepdims=True) * threshold_factor, out, 0.0)
            copied = True

        if truncation_index < out.shape[0]:
            if not copied:
                out = out.copy()
            out[truncation_index:] = 0.0

        return out
