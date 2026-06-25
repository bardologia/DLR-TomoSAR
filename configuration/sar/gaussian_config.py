from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GaussianConfig:
    n_default_gaussians : int
    x_min               : float
    x_max               : float
    amp_max             : float = 1000
    params_per_gaussian : int   = 3
    predict_presence    : bool  = False
