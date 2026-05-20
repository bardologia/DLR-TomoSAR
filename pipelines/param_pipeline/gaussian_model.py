from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


class GaussianModel:
    @staticmethod
    def multi_gaussian(height_axis: np.ndarray, *parameters) -> np.ndarray:
        params_array = np.asarray(parameters, dtype=np.float64).reshape(-1, 3)
        amplitudes   = params_array[:, 0:1]
        means        = params_array[:, 1:2]
        sigmas       = params_array[:, 2:3]

        safe_sigma_sq = 2.0 * (sigmas * sigmas) + 1e-10
        diff          = height_axis[None, :] - means
        exponent      = -(diff * diff) / safe_sigma_sq
        
        np.clip(exponent, -100.0, 0.0, out=exponent)

        return (amplitudes * np.exp(exponent)).sum(axis=0)

    @staticmethod
    def estimate_initial_parameters(height_axis: np.ndarray, profile: np.ndarray, number_of_gaussians: int) -> list[float]:
        smoothed_profile = uniform_filter1d(profile, size=5, mode="nearest")
        sigma_guess      = (height_axis[-1] - height_axis[0]) / (4.0 * number_of_gaussians)
        working_profile  = smoothed_profile.copy()

        initial_parameters = []
        for _ in range(number_of_gaussians):
            peak_index     = int(np.argmax(working_profile))
            peak_mean      = float(height_axis[peak_index])
            peak_amplitude = float(working_profile[peak_index])

            initial_parameters.extend([max(peak_amplitude, 1e-10), peak_mean, sigma_guess])

            suppression_mask                  = np.abs(height_axis - peak_mean) < 2 * sigma_guess
            working_profile[suppression_mask] = 0.0

        return initial_parameters
