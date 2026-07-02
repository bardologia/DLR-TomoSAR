from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.normalization.general     import ChannelStats, ChannelStrategy, NormMethod
from pipelines.backbone.dataset.normalizer    import Normalizer
from pipelines.backbone.dataset.stats         import Stats


def _zscore_stats() -> Stats:
    input_stats = ChannelStats(
        loc        = [1.0, -2.0, 0.0],
        scale      = [3.0, 0.5, 2.0],
        names      = ["pass/phase", "pass/phase", "ifg/phase"],
        strategies = [ChannelStrategy(NormMethod.ZSCORE)] * 3,
    )

    return Stats(input_stats=input_stats, output_stats=None)


def _log1p_stats() -> Stats:
    input_stats = ChannelStats(
        loc        = [0.5, 1.2],
        scale      = [0.8, 1.5],
        names      = ["pass/mag", "ifg/mag"],
        strategies = [ChannelStrategy(NormMethod.ZSCORE, apply_log1p=True)] * 2,
    )

    return Stats(input_stats=input_stats, output_stats=None)


def test_zscore_roundtrip_numpy():
    norm = Normalizer(_zscore_stats())
    x    = np.random.default_rng(0).standard_normal((3, 5, 5)).astype(np.float32)

    back = norm.denormalize_input(norm.normalize_input(x))

    np.testing.assert_allclose(back, x, atol=1e-4)


def test_zscore_normalized_output_matches_formula():
    norm = Normalizer(_zscore_stats())
    x    = np.random.default_rng(1).standard_normal((3, 4, 4)).astype(np.float32)

    out      = norm.normalize_input(x)
    expected = (x - np.array([1.0, -2.0, 0.0]).reshape(-1, 1, 1)) / np.array([3.0, 0.5, 2.0]).reshape(-1, 1, 1)

    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_log1p_roundtrip_numpy():
    norm = Normalizer(_log1p_stats())
    x    = np.abs(np.random.default_rng(2).standard_normal((2, 6, 6))).astype(np.float32) * 10.0

    back = norm.denormalize_input(norm.normalize_input(x))

    np.testing.assert_allclose(back, x, atol=1e-3)


def test_torch_and_numpy_normalization_agree():
    norm = Normalizer(_zscore_stats())
    x    = np.random.default_rng(3).standard_normal((3, 4, 4)).astype(np.float32)

    out_np    = norm.normalize_input(x)
    out_torch = norm.normalize_input(torch.from_numpy(x))

    np.testing.assert_allclose(out_np, out_torch.numpy(), atol=1e-5)


def test_batched_4d_normalization_roundtrip():
    norm = Normalizer(_zscore_stats())
    x    = torch.randn(2, 3, 5, 5)

    back = norm.denormalize_input(norm.normalize_input(x))

    assert torch.allclose(back, x, atol=1e-4)


def test_missing_output_stats_raises():
    norm = Normalizer(_zscore_stats())

    with pytest.raises(ValueError):
        norm.normalize_output(np.zeros((1, 2, 2), dtype=np.float32))


@pytest.mark.real_data
@pytest.mark.slow
def test_fit_real_window_stats_finite_and_roundtrip(data_dir, interferograms, parameters):
    from configuration.dataset                      import InputConfig, OutputConfig, Representation
    from pipelines.backbone.dataset.datasets        import PatchDataset
    from pipelines.backbone.dataset.stats_computer  import StatsComputer
    from pipelines.backbone.dataset.spatial         import Patcher
    from tools.monitoring.logger                    import Logger

    ifg     = np.ascontiguousarray(np.asarray(interferograms[:4, :24, :24]))
    primary = ifg[:1]
    inputs  = np.concatenate([primary, ifg], axis=0)
    params  = np.ascontiguousarray(np.asarray(parameters[:, :24, :24]))

    patcher = Patcher.build(spatial_size=(24, 24), patch_size=(8, 8), stride=8)
    ic      = InputConfig(use_primary=True, primary_representation=Representation.MAG_ONLY,
                          use_secondaries=False,
                          use_interferograms=True, interferograms_representation=Representation.ANGLE_ONLY)
    oc      = OutputConfig()

    ds = PatchDataset(
        inputs=inputs, gt_parameters=params, grid=patcher,
        input_config=ic, output_config=oc, split_name="train",
        n_secondaries=0, n_interferograms=4, n_gaussians=5,
    )

    logger = Logger(log_dir="logs", name="norm_real", level="ERROR")
    stats  = StatsComputer.compute(
        dataset=ds, logger=logger, input_config=ic, output_config=oc,
        n_secondaries=0, n_interferograms=4, n_gaussians=5,
    )

    assert np.all(np.isfinite(stats.input_stats.loc))
    assert np.all(np.isfinite(stats.input_stats.scale))
    assert np.all(np.asarray(stats.input_stats.scale) > 0.0)
    assert np.all(np.isfinite(stats.output_stats.loc))
    assert np.all(np.isfinite(stats.output_stats.scale))

    norm   = Normalizer(stats)
    sample = ds[0][0]
    out    = norm.normalize_input(sample)
    back   = norm.denormalize_input(out)

    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(back, sample, atol=1e-2)
