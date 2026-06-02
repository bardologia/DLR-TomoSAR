from __future__ import annotations

import gc
from pathlib import Path
from typing  import Dict, Tuple

import numpy as np

from tools.logger import Logger


class FittingMetricsCalculator:
    def __init__(self, n_gaussians : int, logger : Logger, amp_threshold : float = 1e-3) -> None:
        self.n_gaussians   = n_gaussians
        self.logger        = logger
        self.amp_threshold = amp_threshold

    @staticmethod
    def _load_tomogram(tomogram_path: Path) -> np.ndarray:
        return np.load(str(tomogram_path)).astype(np.float32, copy=False)        

    @staticmethod
    def _build_height_axis(height_range: Tuple[float, float], n_elev: int) -> np.ndarray:
        return np.linspace(float(height_range[0]), float(height_range[1]), n_elev, dtype=np.float32)

    def _reconstruct_slice(self, parameters_array : np.ndarray, h_val : float) -> np.ndarray:
        Az, R         = parameters_array.shape[1:]
        reconstructed = np.zeros((Az, R), dtype=np.float32)

        for k in range(self.n_gaussians):
            amp = parameters_array[3 * k    ]
            mu  = parameters_array[3 * k + 1]
            sig = parameters_array[3 * k + 2]
            reconstructed += amp * np.exp(-((h_val - mu) ** 2) / (2.0 * sig ** 2 + 1e-12))

        return reconstructed

    def _compute_r2_map(self, tomogram : np.ndarray, parameters_array : np.ndarray, height_axis : np.ndarray) -> np.ndarray:
        n_elev = height_axis.size
        Az, R  = tomogram.shape[1:]

        tom_mean = tomogram.mean(axis=0, dtype=np.float64)

        ss_res = np.zeros((Az, R), dtype=np.float64)
        ss_tot = np.zeros((Az, R), dtype=np.float64)

        for j in range(n_elev):
            tom_h = tomogram[j].astype(np.float64)
            rec_h = self._reconstruct_slice(parameters_array, float(height_axis[j])).astype(np.float64)
            ss_res += (rec_h - tom_h) ** 2
            ss_tot += (tom_h - tom_mean) ** 2

        ss_tot += 1e-12

        return (1.0 - ss_res / ss_tot).astype(np.float32)

    def _compute_activity_map(self, parameters_array: np.ndarray) -> np.ndarray:
        active = np.zeros(parameters_array.shape[1:], dtype=np.int32)              
        for k in range(self.n_gaussians):
            active += (parameters_array[3 * k] >= self.amp_threshold).astype(np.int32)
        
        return active

    def _compute_per_gaussian_maps(self, parameters_array: np.ndarray) -> Dict[str, np.ndarray]:
        maps: Dict[str, np.ndarray] = {}
        for k in range(self.n_gaussians):
            amp_k  = parameters_array[3 * k    ].astype(np.float32)
            active = amp_k >= self.amp_threshold
            maps[f"amp_{k}"]   = np.where(active, amp_k,                                           np.nan).astype(np.float32)
            maps[f"mu_{k}"]    = np.where(active, parameters_array[3 * k + 1].astype(np.float32),  np.nan).astype(np.float32)
            maps[f"sigma_{k}"] = np.where(active, parameters_array[3 * k + 2].astype(np.float32),  np.nan).astype(np.float32)
       
        return maps

    def _compute_mu_separation_maps(self, parameters_array: np.ndarray) -> Dict[str, np.ndarray]:
        maps: Dict[str, np.ndarray] = {}
        if self.n_gaussians < 2:
            return maps
        
        for k in range(self.n_gaussians - 1):
            a_k   = parameters_array[3 * k            ]
            a_kp1 = parameters_array[3 * (k + 1)      ]
            m_k   = parameters_array[3 * k         + 1]
            m_kp1 = parameters_array[3 * (k + 1)   + 1]
            both  = (a_k >= self.amp_threshold) & (a_kp1 >= self.amp_threshold)
            maps[f"mu_sep_{k}_{k + 1}"] = np.where(both, np.abs(m_kp1.astype(np.float32) - m_k.astype(np.float32)), np.nan).astype(np.float32)
        
        return maps

    def _compute_global_summary(self, r2_map : np.ndarray, activity_map : np.ndarray) -> Dict[str, float]:
        r2_flat  = r2_map.reshape(-1).astype(np.float64)
        r2_valid = r2_flat[np.isfinite(r2_flat)]
        n_total  = int(r2_map.size)
        act_flat = activity_map.reshape(-1)

        def _pct(q: float) -> float:
            return float(np.percentile(r2_valid, q)) if r2_valid.size > 0 else float("nan")

        summary: Dict[str, float] = {
            "n_pixels"    : float(n_total),
            "n_gaussians" : float(self.n_gaussians),
            "r2_mean"     : float(r2_valid.mean())      if r2_valid.size > 0 else float("nan"),
            "r2_median"   : float(np.median(r2_valid))  if r2_valid.size > 0 else float("nan"),
            "r2_std"      : float(r2_valid.std())       if r2_valid.size > 0 else float("nan"),
            "r2_p10"      : _pct(10),
            "r2_p25"      : _pct(25),
            "r2_p50"      : _pct(50),
            "r2_p75"      : _pct(75),
            "r2_p90"      : _pct(90),
            "r2_neg_frac" : float((r2_valid < 0.0).mean()) if r2_valid.size > 0 else float("nan"),
        }

        for k in range(self.n_gaussians + 1):
            n_k = int((act_flat == k).sum())
            summary[f"frac_{k}_active"] = float(n_k) / n_total if n_total > 0 else float("nan")

        return summary

    def run(self, parameters_array : np.ndarray, metadata : dict, tomogram_path : Path) -> dict:
        self.logger.section("[Fitting Metrics Calculation]")


        self.logger.subsection(f"Loading tomogram : {Path(tomogram_path).name}")
        tomogram     = self._load_tomogram(Path(tomogram_path))                     
        height_range = tuple(metadata["height_range"])
        height_axis  = self._build_height_axis(height_range, tomogram.shape[0])    

        self.logger.subsection(f"Tomogram shape  : {tomogram.shape}")
        self.logger.subsection(f"Height range    : [{height_range[0]:.1f}, {height_range[1]:.1f}] m")
        self.logger.subsection("Computing per-pixel R\u00b2 map (single-pass, float32)")
        r2_map = self._compute_r2_map(tomogram, parameters_array, height_axis)

        del tomogram
        gc.collect()

        self.logger.subsection("Computing activity and spatial parameter maps")
        activity_map = self._compute_activity_map(parameters_array)
        gauss_maps   = self._compute_per_gaussian_maps(parameters_array)
        sep_maps     = self._compute_mu_separation_maps(parameters_array)
        summary      = self._compute_global_summary(r2_map, activity_map)

        self.logger.subsection(f"R\u00b2 \u2014 mean={summary['r2_mean']:.4f} median={summary['r2_median']:.4f} neg_frac={summary['r2_neg_frac']:.4f}")

        return {
            "r2_map"         : r2_map,
            "activity_map"   : activity_map,
            "height_axis"    : height_axis,
            "global_summary" : summary,
            **gauss_maps,
            **sep_maps,
        }
