from __future__ import annotations

import torch

from .config import DataConfig


class ProfileNormalizer:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.mode         = config.normalize
        self.log_compress = config.log_compress
        self.log_eps      = config.log_eps

    def apply(self, profile: torch.Tensor) -> tuple[torch.Tensor, float]:
        out = profile
        if self.log_compress:
            out = torch.log(out.clamp(min=self.log_eps))

        if self.mode == "none":
            return out, 1.0
       
        if self.mode == "per_profile_max":
            denom = float(out.abs().max().clamp(min=1e-8).item())
            return out / denom, denom
      
        if self.mode == "per_profile_zscore":
            mu  = out.mean()
            std = out.std().clamp(min=1e-8)
            return (out - mu) / std, 1.0
       
        if self.mode == "global":
            return out, 1.0
       
        raise ValueError(f"Unknown normalisation mode '{self.mode}'.")

    def invert_numpy(self, profiles, scales):
        import numpy as np
        scales = scales.reshape(-1, 1)
       
        if self.mode == "none":
            data = profiles
       
        elif self.mode == "per_profile_max":
            data = profiles * scales
       
        elif self.mode == "per_profile_zscore":
            data = profiles
       
        elif self.mode == "global":
            data = profiles
        
        else:
            raise ValueError(f"Unknown normalisation mode '{self.mode}'.")

        if self.log_compress:
            data = np.exp(data)
        return data
