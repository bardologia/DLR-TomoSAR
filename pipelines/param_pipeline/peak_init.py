from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


class PeakInitialiser:
    @staticmethod
    def _prominence_worker(smoothed_chunk : np.ndarray, height_axis : np.ndarray, K : int, sigma_guess : float, min_dist : int, prominence_frac : float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        chunk_N, H = smoothed_chunk.shape
        amps = np.zeros((chunk_N, K), dtype=np.float32)
        mus  = np.zeros((chunk_N, K), dtype=np.float32)
        sigs = np.full ((chunk_N, K), sigma_guess, dtype=np.float32)

        for n in range(chunk_N):
            smo  = smoothed_chunk[n]
            pmax = smo.max()
            if pmax < 1e-10:
                idxs = np.linspace(0, H - 1, K, dtype=int)
            else:
                peaks, props = find_peaks(smo, prominence=pmax * prominence_frac, distance=min_dist)
                if len(peaks) >= K:
                    idxs = peaks[np.argsort(props["prominences"])[::-1][:K]]

                elif len(peaks) > 0:
                    residual        = smo.copy()
                    residual[peaks] = 0.0
                    extra           = []

                    for _ in range(K - len(peaks)):
                        ei = int(np.argmax(residual))
                        extra.append(ei)
                        lo = max(0, ei - min_dist)
                        hi = min(H, ei + min_dist + 1)
                        residual[lo:hi] = 0.0

                    idxs = np.concatenate([peaks, np.array(extra, dtype=int)])

                else:
                    idxs = np.linspace(0, H - 1, K, dtype=int)

            for g, idx in enumerate(idxs[:K]):
                amps[n, g] = max(float(smo[idx]), 1e-10)
                mus [n, g] = float(height_axis[idx])

        return amps, mus, sigs

    def run(self, prof_raw : np.ndarray, height_axis : np.ndarray, K : int, prominence_frac : float = 0.05, n_workers : int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N, H        = prof_raw.shape
        h_span      = float(height_axis[-1] - height_axis[0])
        dh          = float(height_axis[1] - height_axis[0])
        sigma_guess = max(2.0 * dh, h_span / (8.0 * K))
        min_dist    = max(1, int(sigma_guess / dh))
        smoothed    = uniform_filter1d(prof_raw.astype(np.float32), size=5, mode="nearest", axis=1).copy()

        chunk_size = max(1, -(-N // (n_workers * 2)))
        chunks     = [smoothed[i:i + chunk_size] for i in range(0, N, chunk_size)]

        worker_fn  = partial(
            self._prominence_worker,
            height_axis     = height_axis,
            K               = K,
            sigma_guess     = sigma_guess,
            min_dist        = min_dist,
            prominence_frac = prominence_frac,
        )

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            chunk_results = list(pool.map(worker_fn, chunks))

        amps = np.concatenate([r[0] for r in chunk_results], axis=0)
        mus  = np.concatenate([r[1] for r in chunk_results], axis=0)
        sigs = np.concatenate([r[2] for r in chunk_results], axis=0)

        return amps, mus, sigs
