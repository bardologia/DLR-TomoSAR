from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.backbone.inference.analysis.input_attribution import TrackChannels
from pipelines.backbone.inference.analysis.robustness        import RobustnessCore
from pipelines.backbone.inference.probes                     import PredictionCurves

from types import SimpleNamespace


H, W, N_ELEV = 8, 8, 10


def _model(images):
    x      = np.asarray(images, dtype=np.float32)
    params = np.zeros((x.shape[0], 3, H, W), dtype=np.float32)

    params[:, 0] = np.abs(x[:, 1]) + np.abs(x[:, 2]) + 0.5
    params[:, 1] = 5.0
    params[:, 2] = 2.0

    return params


def _batches():
    rng      = np.random.default_rng(0)
    images   = torch.from_numpy(rng.uniform(0.4, 1.0, size=(2, 3, H, W))).float()
    renderer = PredictionCurves(1, np.linspace(-5.0, 15.0, N_ELEV))

    gt_curves = renderer.render(_model(images))
    return [(images, gt_curves)], renderer


def test_noise_curve_starts_clean_and_degrades():
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    rows = core.noise_curve(batches, [0.0, 0.5, 2.0])

    assert rows[0]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert rows[1]["mse"] > 0.0
    assert rows[2]["mse"] > rows[1]["mse"]


def test_noise_curve_requires_the_clean_anchor():
    batches, renderer = _batches()

    with pytest.raises(ValueError):
        RobustnessCore(_model, renderer).noise_curve(batches, [0.5, 2.0])


def test_noise_on_unused_channels_is_harmless():
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    unused = core.noise_curve(batches, [0.0, 1.0], channels=[0])
    used   = core.noise_curve(batches, [0.0, 1.0], channels=[1, 2])

    assert unused[1]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert used[1]["mse"]   > 0.0


def test_noise_draws_report_spread():
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    rows = core.noise_curve(batches, [0.0, 0.5], draws=3)

    assert rows[0]["mse_std"] is None
    assert rows[1]["mse_std"] is not None


def test_gain_curve_always_holds_the_reference():
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    rows  = core.gain_curve(batches, [0.5, 2.0])
    gains = [row["gain"] for row in rows]

    assert gains == [0.5, 1.0, 2.0]
    assert rows[1]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert rows[0]["mse"] > 0.0
    assert rows[2]["mse"] > 0.0


def test_drop_curve_degrades_with_more_tracks():
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    rows = core.drop_curve(batches, [[1], [2]], draws=2)

    assert rows[0]["dropped"] == 0
    assert rows[0]["mse"]     == pytest.approx(0.0, abs=1e-12)
    assert rows[1]["mse"]     > 0.0
    assert rows[2]["mse"]     > rows[1]["mse"]
    assert rows[2]["dropped"] == 2


def test_drop_curve_evaluates_each_distinct_subset_once():
    batches, renderer = _batches()
    calls             = []

    def counting_model(images):
        calls.append(images)
        return _model(images)

    core = RobustnessCore(counting_model, renderer)
    rows = core.drop_curve(batches, [[1], [2]], draws=8)

    assert len(calls) == 4

    assert rows[1]["mse_std"] is not None
    assert rows[2]["mse_std"] is None


def test_shift_consistency_is_zero_for_a_pointwise_model():
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    rows = core.shift_consistency(batches, [1, 3])

    assert rows[0]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert rows[1]["mse"] == pytest.approx(0.0, abs=1e-12)


def test_shift_consistency_detects_position_dependence():
    batches, renderer = _batches()

    def positional_model(images):
        params = _model(images)
        params[:, 0] += np.arange(H, dtype=np.float32)[None, :, None]
        return params

    rows = RobustnessCore(positional_model, renderer).shift_consistency(batches, [2])

    assert rows[0]["mse"] > 1e-6


def test_shift_consistency_rejects_out_of_range_shifts():
    batches, renderer = _batches()

    with pytest.raises(ValueError):
        RobustnessCore(_model, renderer).shift_consistency(batches, [H])


def test_tolerance_sigma_interpolates_and_censors():
    crossing = [
        {"sigma": 0.0, "mse": 1.0, "mse_std": None},
        {"sigma": 1.0, "mse": 1.5, "mse_std": None},
        {"sigma": 2.0, "mse": 3.5, "mse_std": None},
    ]
    flat = [
        {"sigma": 0.0, "mse": 1.0, "mse_std": None},
        {"sigma": 1.0, "mse": 1.2, "mse_std": None},
    ]

    tolerance = RobustnessCore.tolerance_sigma(crossing)

    assert tolerance["sigma"]    == pytest.approx(1.25)
    assert tolerance["censored"] is False

    censored = RobustnessCore.tolerance_sigma(flat)

    assert censored["sigma"]    == pytest.approx(1.0)
    assert censored["censored"] is True


def test_track_channels_map_secondaries_and_interferograms():
    input_config = SimpleNamespace(
        use_primary                      = True,
        use_secondaries                  = True,
        use_interferograms               = True,
        primary_channels_per_pass        = 1,
        secondaries_channels_per_pass    = 1,
        interferograms_channels_per_pass = 1,
    )
    run = SimpleNamespace(dataset=SimpleNamespace(input_config=input_config), n_secondaries=2)

    per_track = TrackChannels.build(run)

    assert per_track == [[1, 3], [2, 4]]


def test_track_channels_without_tracks_raise():
    input_config = SimpleNamespace(
        use_primary                      = True,
        use_secondaries                  = False,
        use_interferograms               = False,
        primary_channels_per_pass        = 1,
        secondaries_channels_per_pass    = 1,
        interferograms_channels_per_pass = 1,
    )
    run = SimpleNamespace(dataset=SimpleNamespace(input_config=input_config), n_secondaries=0)

    with pytest.raises(ValueError):
        TrackChannels.build(run)
