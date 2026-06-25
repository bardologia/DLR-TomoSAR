from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.normalization.general    import ChannelStats, ChannelStrategy, NormMethod
from pipelines.backbone.dataset.normalizer    import Normalizer
from pipelines.backbone.dataset.stats         import Stats
from pipelines.backbone.dataset.stats_computer import StrategyFitter

ROBUST_LOG1P_SLOTS = ["pass/mag", "ifg/mag", "out/amp", "out/sigma", "dem/elevation"]
ZSCORE_SLOTS       = ["out/mu", "pass/phase", "ifg/phase", "pass/raw_re_im", "ifg/raw_re_im"]

AMP_THR        = 1e-2
SCALE_FLOOR    = 1e-8
REL_AGREE_TOL  = 0.05


def _active(parameters):
    par = np.asarray(parameters)
    amp = par[0::3].astype(np.float64)
    mu  = par[1::3].astype(np.float64)
    sig = par[2::3].astype(np.float64)
    m   = amp > AMP_THR
    return amp[m], mu[m], sig[m]


def _random_split_agreement(strat, vals, seed):
    r       = np.random.default_rng(seed).random(vals.shape) < 0.5
    l1, s1  = StrategyFitter.fit(strat, vals[r])
    l2, s2  = StrategyFitter.fit(strat, vals[~r])
    loc_rel   = abs(l1 - l2) / (abs(l1) + abs(l2) + 1e-9)
    scale_rel = abs(s1 - s2) / (s1 + s2 + 1e-9)
    return loc_rel, scale_rel


def _channel_stats(slot_keys, pools):
    strategies = [ChannelStrategy.from_slot(k) for k in slot_keys]
    fits       = [StrategyFitter.fit(strat, pool) for strat, pool in zip(strategies, pools)]
    return ChannelStats(
        loc        = [f[0] for f in fits],
        scale      = [f[1] for f in fits],
        names      = list(slot_keys),
        strategies = strategies,
    )


def test_heavy_tailed_channels_use_robust_iqr_log1p():
    for slot in ROBUST_LOG1P_SLOTS:
        strat = ChannelStrategy.from_slot(slot)
        assert strat.norm_method is NormMethod.ROBUST_IQR
        assert strat.apply_log1p is True


def test_mu_and_signed_channels_stay_zscore():
    assert ChannelStrategy.from_slot("out/mu").norm_method is NormMethod.ZSCORE
    assert ChannelStrategy.from_slot("out/mu").apply_log1p is False

    for slot in ZSCORE_SLOTS:
        assert ChannelStrategy.from_slot(slot).norm_method is NormMethod.ZSCORE


@pytest.mark.real_data
def test_mu_robust_iqr_is_degenerate_which_is_why_mu_stays_zscore(parameters):
    _, mu, _ = _active(parameters)

    _, robust_scale = StrategyFitter.fit(ChannelStrategy(NormMethod.ROBUST_IQR, apply_log1p=False), mu)
    _, zscore_scale = StrategyFitter.fit(ChannelStrategy.from_slot("out/mu"), mu)

    assert robust_scale <= 1e-7
    assert zscore_scale > 1.0


@pytest.mark.real_data
@pytest.mark.parametrize("role", ["amp", "mu", "sigma"])
def test_output_stats_reproducible_across_random_halves(parameters, role):
    amp, mu, sig = _active(parameters)
    vals  = {"amp": amp, "mu": mu, "sigma": sig}[role]
    slot  = {"amp": "out/amp", "mu": "out/mu", "sigma": "out/sigma"}[role]
    strat = ChannelStrategy.from_slot(slot)

    for seed in range(4):
        loc_rel, scale_rel = _random_split_agreement(strat, vals, seed)
        assert loc_rel   < REL_AGREE_TOL
        assert scale_rel < REL_AGREE_TOL


@pytest.mark.real_data
def test_input_magnitude_stats_reproducible_across_random_halves(primary, secondaries, interferograms):
    pass_mag = np.concatenate([np.abs(np.asarray(primary)).ravel(), np.abs(np.asarray(secondaries)).ravel()]).astype(np.float64)
    ifg_mag  = np.abs(np.asarray(interferograms)).ravel().astype(np.float64)

    for slot, vals in [("pass/mag", pass_mag), ("ifg/mag", ifg_mag)]:
        strat = ChannelStrategy.from_slot(slot)
        loc_rel, scale_rel = _random_split_agreement(strat, vals, seed=0)
        assert loc_rel   < REL_AGREE_TOL
        assert scale_rel < REL_AGREE_TOL


@pytest.mark.real_data
def test_configured_robust_scales_are_nondegenerate_on_real_data(parameters, primary, secondaries, interferograms, dem_full):
    amp, _, sig = _active(parameters)
    pass_mag    = np.abs(np.asarray(primary)).ravel().astype(np.float64)
    ifg_mag     = np.abs(np.asarray(interferograms)).ravel().astype(np.float64)
    dem         = np.asarray(dem_full).ravel().astype(np.float64)

    pools = {"out/amp": amp, "out/sigma": sig, "pass/mag": pass_mag, "ifg/mag": ifg_mag, "dem/elevation": dem}

    for slot, vals in pools.items():
        _, scale = StrategyFitter.fit(ChannelStrategy.from_slot(slot), vals)
        assert scale > 1e-3
        assert scale > SCALE_FLOOR * 100


@pytest.mark.real_data
def test_output_normalizer_round_trip_is_exact(parameters):
    par   = np.asarray(parameters)
    amp, mu, sig = _active(parameters)
    stats = _channel_stats(["out/amp", "out/mu", "out/sigma"], [amp, mu, sig])

    norm = Normalizer(Stats(output_stats=stats))

    t = torch.from_numpy(par[0:3, :64, :64].copy()).unsqueeze(0).float()
    n = norm.normalize_output(t)
    d = norm.denormalize_output(n)

    assert torch.isfinite(n).all()
    assert torch.allclose(d, t, rtol=1e-4, atol=1e-3)


@pytest.mark.real_data
def test_input_normalizer_round_trip_is_exact(primary, interferograms, secondaries):
    pass_mag = np.concatenate([np.abs(np.asarray(primary)).ravel(), np.abs(np.asarray(secondaries)).ravel()]).astype(np.float64)
    ifg_mag  = np.abs(np.asarray(interferograms)).ravel().astype(np.float64)
    stats    = _channel_stats(["pass/mag", "ifg/mag"], [pass_mag, ifg_mag])

    norm = Normalizer(Stats(input_stats=stats))

    p = np.abs(np.asarray(primary)[:64, :64]).astype(np.float32)
    i = np.abs(np.asarray(interferograms)[0, :64, :64]).astype(np.float32)
    t = torch.from_numpy(np.stack([p, i])).unsqueeze(0)

    n = norm.normalize_input(t)
    d = norm.denormalize_input(n)

    assert torch.isfinite(n).all()
    assert torch.allclose(d, t, rtol=1e-4, atol=1e-3)


def test_robust_iqr_scale_is_more_outlier_stable_than_zscore():
    rng  = np.random.default_rng(0)
    base = rng.normal(5.0, 1.0, size=20000)

    outliers = np.concatenate([base, np.full(400, 500.0)])

    z_clean  = StrategyFitter.fit(ChannelStrategy(NormMethod.ZSCORE,     apply_log1p=False), base)[1]
    z_out    = StrategyFitter.fit(ChannelStrategy(NormMethod.ZSCORE,     apply_log1p=False), outliers)[1]
    r_clean  = StrategyFitter.fit(ChannelStrategy(NormMethod.ROBUST_IQR, apply_log1p=False), base)[1]
    r_out    = StrategyFitter.fit(ChannelStrategy(NormMethod.ROBUST_IQR, apply_log1p=False), outliers)[1]

    z_drift = abs(z_out - z_clean) / z_clean
    r_drift = abs(r_out - r_clean) / r_clean

    assert r_drift < z_drift
    assert r_drift < 0.05
