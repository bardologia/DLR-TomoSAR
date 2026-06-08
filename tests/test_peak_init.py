from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from pipelines.param_pipeline.sigma import PeakInitialiser


class TestPeakInitialiserLocalisation:
    def _sharp_with_noise_spur(self, H : int = 170, sigma_bins : float = 0.3, peak_bin : int = 85) -> tuple:
        h    = np.linspace(-30.0, 30.0, H)
        dh   = h[1] - h[0]
        rng  = np.random.default_rng(0)
        prof = np.exp(-((h - h[peak_bin]) ** 2) / (2.0 * (sigma_bins * dh) ** 2))
        prof = prof + 0.008 * rng.standard_normal(H)
        return h, np.maximum(prof, 0.0).astype(np.float32), peak_bin

    def test_sharp_peak_mean_sits_on_raw_argmax(self):
        h, prof, _ = self._sharp_with_noise_spur()
        init       = PeakInitialiser(n_workers=1)
        amps, mus, _ = init.run(prof[None, :], h, K=5)
        init.close()

        dominant = int(np.argmax(amps[0]))
        mu_bin   = int(np.argmin(np.abs(h - mus[0, dominant])))

        assert abs(mu_bin - int(np.argmax(prof))) <= 1

    def test_most_prominent_peak_takes_first_slot(self):
        h, prof, peak_bin = self._sharp_with_noise_spur()
        init              = PeakInitialiser(n_workers=1)
        amps, mus, _      = init.run(prof[None, :], h, K=5)
        init.close()

        first_mu_bin = int(np.argmin(np.abs(h - mus[0, 0])))

        assert amps[0, 0] == pytest.approx(float(amps[0].max()))
        assert abs(first_mu_bin - peak_bin) <= 1

    def test_distinct_components_have_distinct_means(self):
        h, prof, _   = self._sharp_with_noise_spur()
        init         = PeakInitialiser(n_workers=1)
        amps, mus, _ = init.run(prof[None, :], h, K=5)
        init.close()

        active = mus[0, amps[0] > 1e-6]

        assert np.unique(active).size == active.size
