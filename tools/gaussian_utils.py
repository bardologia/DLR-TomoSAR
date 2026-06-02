from __future__ import annotations

import torch


def clamp_gaussian_params(
    params      : torch.Tensor,
    x_axis      : torch.Tensor,
    amp_max     : float,
    ppg         : int   = 3,
    leaky_slope : float = 0.0,
) -> torch.Tensor:
    x_min   = x_axis.min()
    x_max   = x_axis.max()
    x_step  = (x_max - x_min) / (x_axis.shape[0] - 1)
    x_range = x_max - x_min
    n_gauss = params.shape[1] // ppg
    slices  = []

    def _clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        if leaky_slope > 0.0:
            clamped = x.clamp(lo, hi)
            return clamped + leaky_slope * (x - clamped).detach()
        return x.clamp(lo, hi)

    for _g in range(n_gauss):
        _b = _g * ppg
        slices.append(_clamp(params[:, _b + 0:_b + 1], 0.0,          amp_max      ))  
        slices.append(_clamp(params[:, _b + 1:_b + 2], x_min,        x_max        ))  
        slices.append(_clamp(params[:, _b + 2:_b + 3], x_step * 0.5, x_range * 0.5)) 

    return torch.cat(slices, dim=1)
