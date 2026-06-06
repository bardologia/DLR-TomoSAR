from __future__ import annotations

import math

import torch


class TomoGeometry:
    def __init__(self, cfg, x_axis: torch.Tensor):
        self.cfg = cfg

        if len(cfg.kz_values) > 0:
            kz_list = [float(v) for v in cfg.kz_values]
        else:
            scale   = 4.0 * math.pi / (cfg.wavelength * cfg.slant_range)
            kz_list = [scale * float(b) for b in cfg.baselines]

        self.kz       = torch.tensor(kz_list, dtype=x_axis.dtype, device=x_axis.device)
        self.n_tracks = self.kz.shape[0]

        phase         = self.kz.reshape(-1, 1) * x_axis.reshape(1, -1)
        self.steering = torch.polar(torch.ones_like(phase), phase)
        self.outer    = torch.einsum("ik,jk->ijk", self.steering, self.steering.conj())

    def describe(self) -> dict:
        return {
            "Tracks":       self.n_tracks,
            "kz min":       f"{float(self.kz.min()):.4f} rad/m" if self.n_tracks else "n/a",
            "kz max":       f"{float(self.kz.max()):.4f} rad/m" if self.n_tracks else "n/a",
            "Wavelength":   f"{self.cfg.wavelength} m",
            "Slant range":  f"{self.cfg.slant_range} m",
            "kz source":    "explicit kz_values" if len(self.cfg.kz_values) > 0 else "baselines via 4*pi*b/(lambda*r0)",
            "Baselines origin": getattr(self.cfg, "baselines_origin", "config"),
        }
