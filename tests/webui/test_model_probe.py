from __future__ import annotations

import sys
from pathlib import Path
from types   import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from model_probe   import ModelProbe
from project_paths import ProjectPaths
from web_logger    import WebLogger

from pipelines.backbone.inference.probes import PredictionCurves


N_AZ, N_RG, PH, PW, N_ELEV = 20, 16, 8, 8, 12


class _SilentLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _BareModel(torch.nn.Module):
    def forward(self, x):
        B, _, H, W = x.shape
        out        = torch.zeros(B, 3, H, W)

        out[:, 0] = x[:, 0] + 1.0
        out[:, 1] = 5.0 * x[:, 0]
        out[:, 2] = 3.0 + 0.0 * x[:, 0]

        return out


class _Wrapper:
    def __init__(self) -> None:
        self.module = _BareModel()

    def __call__(self, x):
        with torch.no_grad():
            return self.module(torch.as_tensor(np.asarray(x, dtype=np.float32))).numpy()


def _probe() -> ModelProbe:
    probe = ModelProbe(ProjectPaths(), _SilentLogger())

    rng            = np.random.default_rng(0)
    complex_inputs = (rng.uniform(0.5, 1.0, size=(2, N_AZ, N_RG)) + 1j * rng.uniform(0.0, 0.2, size=(2, N_AZ, N_RG))).astype(np.complex64)
    gt_params      = np.zeros((3, N_AZ, N_RG), dtype=np.float32)

    gt_params[0] = 1.5
    gt_params[1] = 20.0
    gt_params[2] = 4.0

    dataset = SimpleNamespace(
        dem             = None,
        gt_parameters   = gt_params,
        assemble_window = lambda window, dem: np.abs(window).astype(np.float32),
    )

    run = SimpleNamespace(
        model          = _Wrapper(),
        dataset        = dataset,
        complex_inputs = complex_inputs,
        full_curves    = np.ones((N_ELEV, N_AZ, N_RG), dtype=np.float32),
        x_axis         = np.linspace(-10.0, 40.0, N_ELEV).astype(np.float32),
        n_gaussians    = 1,
        split_region   = SimpleNamespace(azimuth_size=N_AZ, range_size=N_RG, azimuth_start=0, range_start=0),
    )

    probe.loaded = {
        "run"      : run,
        "labels"   : ["primary", "sec PS04"],
        "layers"   : ["conv"],
        "types"    : {"conv": "Conv2d"},
        "renderer" : PredictionCurves(1, run.x_axis),
        "patch"    : (PH, PW),
    }
    probe.status = {"state": "ready", "path": "fake", "progress": 1.0, "stage": "ready", "error": "", "info": None}

    return probe


def test_predict_returns_slots_curves_and_references():
    result = _probe().predict({"az": 10, "rg": 8})

    assert result["ok"] is True
    assert len(result["slots"])     == 1
    assert result["slots"][0]["active"] is True
    assert len(result["curve"])     == N_ELEV
    assert len(result["gt_curve"])  == N_ELEV
    assert result["gt_slots"][0]["mu"] == pytest.approx(20.0)
    assert len(result["raw_curve"]) == N_ELEV


def test_predict_outside_region_fails():
    result = _probe().predict({"az": 99, "rg": 0})

    assert result["ok"] is False
    assert "outside" in result["error"]


def test_predict_without_loaded_model_fails():
    probe = ModelProbe(ProjectPaths(), _SilentLogger())

    assert probe.predict({"az": 0, "rg": 0})["ok"] is False


def test_saliency_concentrates_on_the_used_channel():
    result = _probe().saliency({"az": 10, "rg": 8, "family": "mu"})

    assert result["ok"] is True
    assert result["shares"][0] == pytest.approx(1.0)
    assert result["shares"][1] == pytest.approx(0.0)
    assert len(result["map"])  == PH
    assert max(max(row) for row in result["map"]) == pytest.approx(1.0)


def test_saliency_dead_family_fails():
    result = _probe().saliency({"az": 10, "rg": 8, "family": "sigma"})

    assert result["ok"] is False


def test_whatif_drop_of_used_channel_shifts_the_prediction():
    probe = _probe()

    dropped   = probe.whatif({"az": 10, "rg": 8, "perturbation": {"kind": "drop_channel", "channel": 0}})
    untouched = probe.whatif({"az": 10, "rg": 8, "perturbation": {"kind": "drop_channel", "channel": 1}})

    assert dropped["ok"] and untouched["ok"]
    assert dropped["delta_mse"]   > 1e-6
    assert untouched["delta_mse"] == pytest.approx(0.0, abs=1e-12)


def test_whatif_unknown_perturbation_fails():
    result = _probe().whatif({"az": 10, "rg": 8, "perturbation": {"kind": "warp"}})

    assert result["ok"] is False
    assert "unknown perturbation" in result["error"]


def test_layers_and_map_need_a_loaded_model():
    probe = ModelProbe(ProjectPaths(), _SilentLogger())

    assert probe.layers()["ok"] is False
    assert probe.map_png() is None


def test_map_png_renders_for_a_loaded_run():
    png = _probe().map_png()

    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
