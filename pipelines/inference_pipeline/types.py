from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

import numpy as np


@dataclass
class Result:
    pred_curves        : np.ndarray
    gt_curves          : np.ndarray
    params_pred        : np.ndarray
    params_gt          : np.ndarray

    pixel_mse          : np.ndarray
    pixel_mae          : np.ndarray
    pixel_r2           : np.ndarray
    pixel_cosine       : np.ndarray
    pixel_peak_err_idx : np.ndarray

    cube_directory     : Path
    azimuth_offset     : int
    range_offset       : int
