from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pipelines.autoencoder_common.inference.metrics import AeMetricsBase
from pipelines.image_autoencoder.inference.metrics import ImageAeMetrics
from pipelines.profile_autoencoder.inference.metrics import ProfileAeMetrics


class _Norm:
    def normalize_input(self, x):
        return x

    def normalize(self, x):
        return x


def test_metrics_share_base():
    assert issubclass(ImageAeMetrics, AeMetricsBase)
    assert issubclass(ProfileAeMetrics, AeMetricsBase)


def test_image_metrics_compute_keys():
    rng = np.random.default_rng(0)
    res = SimpleNamespace(
        gt         = rng.random((6, 2, 4, 4)).astype(np.float32),
        pred       = rng.random((6, 2, 4, 4)).astype(np.float32),
        embeddings = rng.random((6, 8)).astype(np.float32),
    )

    out = ImageAeMetrics(res, _Norm()).compute()

    assert out["n_patches"] == 6
    assert out["n_channels"] == 2
    for key in ("mse_mean", "psnr", "channel_mse", "mse_mean_normalized", "embedding_norm_mean"):
        assert key in out


def test_profile_metrics_compute_keys():
    rng = np.random.default_rng(1)
    res = SimpleNamespace(
        gt         = rng.random((10, 20)).astype(np.float32),
        pred       = rng.random((10, 20)).astype(np.float32),
        embeddings = rng.random((10, 4)).astype(np.float32),
    )

    out = ProfileAeMetrics(res, np.linspace(0.0, 1.0, 20), _Norm(), 1e-3).compute()

    assert out["n_curves"] == 10
    assert out["profile_length"] == 20
    for key in ("mse_mean", "pearson_mean", "power_rel_error_mean", "peak_location_mae", "embedding_norm_mean"):
        assert key in out


def test_write_json_roundtrip(tmp_path):
    path = AeMetricsBase.write_json({"a": 1, "b": 2.5}, tmp_path / "m.json")
    assert path.exists()
