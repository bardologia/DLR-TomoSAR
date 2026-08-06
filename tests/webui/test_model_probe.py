from __future__ import annotations

import base64

from types   import SimpleNamespace

import numpy as np
import pytest
import torch

from model_probe import ModelProbe
from web_logger  import WebLogger

from pipelines.backbone.inference.probes import PredictionCurves

from tests.conftest import SilentLogger


N_AZ, N_RG, PH, PW, N_ELEV = 20, 16, 8, 8, 12


class _BareModel(torch.nn.Module):
    def forward(self, x):
        B, _, H, W = x.shape
        out        = torch.zeros(B, 3, H, W)

        out[:, 0] = x[:, 0] + 1.0
        out[:, 1] = 5.0 * x[:, 0]
        out[:, 2] = 3.0 + 0.0 * x[:, 0]

        return out


class _TwoSlotModel(torch.nn.Module):
    def forward(self, x):
        B, _, H, W = x.shape
        out        = torch.zeros(B, 6, H, W)

        out[:, 0] = x[:, 0] + 1.0
        out[:, 1] = 5.0 * x[:, 0]
        out[:, 2] = 3.0 + 0.0 * x[:, 0]
        out[:, 3] = x[:, 1] + 1.0
        out[:, 4] = 5.0 * x[:, 1]
        out[:, 5] = 3.0 + 0.0 * x[:, 1]

        return out


class _Wrapper:
    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module

    def __call__(self, x):
        with torch.no_grad():
            return self.module(torch.as_tensor(np.asarray(x, dtype=np.float32))).numpy()


def _probe(n_slots: int = 1, with_gt: bool = True) -> ModelProbe:
    probe = ModelProbe(SilentLogger())

    rng            = np.random.default_rng(0)
    complex_inputs = (rng.uniform(0.5, 1.0, size=(2, N_AZ, N_RG)) + 1j * rng.uniform(0.0, 0.2, size=(2, N_AZ, N_RG))).astype(np.complex64)
    gt_params      = np.zeros((3 * n_slots, N_AZ, N_RG), dtype=np.float32)

    for k in range(n_slots):
        gt_params[3 * k]     = 1.5
        gt_params[3 * k + 1] = 20.0
        gt_params[3 * k + 2] = 4.0

    dataset = SimpleNamespace(
        dem             = None,
        gt_parameters   = gt_params if with_gt else None,
        assemble_window = lambda window, dem: np.abs(window).astype(np.float32),
    )

    run = SimpleNamespace(
        model          = _Wrapper(_BareModel() if n_slots == 1 else _TwoSlotModel()),
        dataset        = dataset,
        complex_inputs = complex_inputs,
        full_curves    = np.ones((N_ELEV, N_AZ, N_RG), dtype=np.float32),
        x_axis         = np.linspace(-10.0, 40.0, N_ELEV).astype(np.float32),
        n_gaussians    = n_slots,
        split_region   = SimpleNamespace(azimuth_size=N_AZ, range_size=N_RG, azimuth_start=0, range_start=0),
    )

    probe.loaded = {
        "run"      : run,
        "labels"   : ["primary", "sec PS04"],
        "layers"   : ["conv"],
        "types"    : {"conv": "Conv2d"},
        "renderer" : PredictionCurves(n_slots, run.x_axis),
        "patch"    : (PH, PW),
        "device"   : "cpu",
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


def test_predict_reports_fit_and_hungarian_matches():
    result = _probe().predict({"az": 10, "rg": 8})

    fit = result["fit"]
    assert fit["curve_mse_gt"]  > 0.0
    assert fit["curve_mse_raw"] > 0.0
    assert fit["pred_active"] == 1
    assert fit["gt_active"]   == 1

    match = result["matches"]
    assert len(match) == 1
    assert match[0]["gt_slot"]   == 0
    assert match[0]["pred_slot"] == 0
    assert match[0]["d_amp"]   == pytest.approx(result["slots"][0]["amp"] - 1.5, abs=1e-5)
    assert match[0]["d_mu"]    == pytest.approx(result["slots"][0]["mu"] - 20.0, abs=1e-4)
    assert match[0]["d_sigma"] == pytest.approx(-1.0, abs=1e-5)


def test_predict_matches_cover_every_active_gt_slot():
    result = _probe(n_slots=2).predict({"az": 10, "rg": 8})

    match = result["matches"]
    assert [m["gt_slot"] for m in match]        == [0, 1]
    assert sorted(m["pred_slot"] for m in match) == [0, 1]


def test_predict_without_gt_omits_matches_and_gt_fit():
    result = _probe(with_gt=False).predict({"az": 10, "rg": 8})

    assert result["ok"] is True
    assert result["gt_curve"] is None
    assert result["gt_slots"] is None
    assert result["matches"]  is None
    assert result["fit"]["curve_mse_gt"] is None
    assert result["fit"]["gt_active"]    is None


def test_predict_outside_region_fails():
    result = _probe().predict({"az": 99, "rg": 0})

    assert result["ok"] is False
    assert "outside" in result["error"]


def test_predict_without_loaded_model_fails():
    probe = ModelProbe(SilentLogger())

    assert probe.predict({"az": 0, "rg": 0})["ok"]     is False
    assert probe.attribution({"az": 0, "rg": 0})["ok"] is False


def test_fields_returns_per_slot_maps_and_overviews():
    result = _probe().fields({"az": 10, "rg": 8})

    assert result["ok"]     is True
    assert result["center"] == [4, 4]
    assert result["patch"]  == [PH, PW]
    assert len(result["slots"]) == 1

    slot = result["slots"][0]
    assert slot["active"] is True
    assert [f["family"] for f in slot["families"]] == ["amp", "mu", "sigma"]

    amp = slot["families"][0]
    assert base64.b64decode(amp["png"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert amp["min"] >= 1.5
    assert amp["max"] <= 2.5

    sigma = slot["families"][2]
    assert sigma["min"] == pytest.approx(3.0)
    assert sigma["max"] == pytest.approx(3.0)

    assert result["activity"]["min"] == pytest.approx(1.0)
    assert result["activity"]["max"] == pytest.approx(1.0)
    assert result["error"]["max"] > 0.0


def test_fields_without_gt_has_no_error_map():
    result = _probe(with_gt=False).fields({"az": 10, "rg": 8})

    assert result["ok"]    is True
    assert result["error"] is None


def test_fields_without_loaded_model_fails():
    assert ModelProbe(SilentLogger()).fields({"az": 0, "rg": 0})["ok"] is False


def test_attribution_concentrates_on_the_used_channel():
    result = _probe().attribution({"az": 10, "rg": 8})

    assert result["ok"]       is True
    assert result["channels"] == ["primary", "sec PS04"]
    assert result["center"]   == [4, 4]
    assert result["patch"]    == [PH, PW]
    assert result["slot"]     == -1

    families = {payload["family"]: payload for payload in result["families"]}

    assert families["mu"]["dead"] is False
    assert families["mu"]["shares"][0] == pytest.approx(1.0)
    assert families["mu"]["shares"][1] == pytest.approx(0.0)
    assert base64.b64decode(families["mu"]["cells"][0])[:8] == b"\x89PNG\r\n\x1a\n"
    assert families["mu"]["cells"][1] is None


def test_attribution_per_slot_isolates_each_scatterers_inputs():
    probe = _probe(n_slots=2)

    slot0    = probe.attribution({"az": 10, "rg": 8, "slot": 0})
    slot1    = probe.attribution({"az": 10, "rg": 8, "slot": 1})
    combined = probe.attribution({"az": 10, "rg": 8})

    assert slot0["ok"] and slot1["ok"] and combined["ok"]
    assert (slot0["slot"], slot1["slot"], combined["slot"]) == (0, 1, -1)

    mu0  = {payload["family"]: payload for payload in slot0["families"]}["mu"]
    mu1  = {payload["family"]: payload for payload in slot1["families"]}["mu"]
    both = {payload["family"]: payload for payload in combined["families"]}["mu"]

    assert mu0["shares"][0]  == pytest.approx(1.0)
    assert mu0["shares"][1]  == pytest.approx(0.0)
    assert mu1["shares"][0]  == pytest.approx(0.0)
    assert mu1["shares"][1]  == pytest.approx(1.0)
    assert both["shares"][0] == pytest.approx(0.5)
    assert both["shares"][1] == pytest.approx(0.5)


def test_attribution_rejects_out_of_range_slots():
    probe = _probe()

    high = probe.attribution({"az": 10, "rg": 8, "slot": 1})
    low  = probe.attribution({"az": 10, "rg": 8, "slot": -2})

    assert high["ok"] is False
    assert "out of range" in high["error"]
    assert low["ok"] is False
    assert "out of range" in low["error"]


def test_attribution_marks_dead_families_instead_of_failing():
    result = _probe().attribution({"az": 10, "rg": 8})

    families = {payload["family"]: payload for payload in result["families"]}

    assert families["sigma"]["dead"]   is True
    assert families["sigma"]["shares"] == [0.0, 0.0]
    assert families["sigma"]["cells"]  == [None, None]


def test_ablation_ranks_the_used_channel_first():
    result = _probe().ablation({"az": 10, "rg": 8})

    assert result["ok"] is True
    assert result["base_power"] > 0.0

    ranked = result["channels"]
    assert [c["label"] for c in ranked] == ["primary", "sec PS04"]
    assert ranked[0]["channel"]   == 0
    assert ranked[0]["delta_mse"] > 1e-6
    assert ranked[0]["max_mu_shift"] > 1.0
    assert ranked[1]["delta_mse"] == pytest.approx(0.0, abs=1e-12)
    assert ranked[1]["flips"]     == 0


def test_ablation_without_loaded_model_fails():
    assert ModelProbe(SilentLogger()).ablation({"az": 0, "rg": 0})["ok"] is False


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


class _WithContainer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv       = torch.nn.Conv2d(1, 1, 1)
        self.downsample = torch.nn.ModuleList()


def test_probe_layers_skips_container_modules():
    layers = ModelProbe(SilentLogger())._probe_layers(_WithContainer())

    assert layers == ["conv"]


def test_layers_and_map_need_a_loaded_model():
    probe = ModelProbe(SilentLogger())

    assert probe.layers()["ok"] is False
    assert probe.map_png() is None


def test_map_png_renders_for_a_loaded_run():
    png = _probe().map_png()

    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def _loadable_run(tmp_path, group: str = "backbone", name: str = "run_ok", config_name: str = "model_config.json"):
    run_dir = tmp_path / group / name
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pt").write_bytes(b"x")
    (run_dir / "meta").mkdir()
    (run_dir / "meta" / config_name).write_text("{}")
    return run_dir


def test_runs_lists_loadable_backbone_and_dual_runs(tmp_path):
    backbone = _loadable_run(tmp_path)
    dual     = _loadable_run(tmp_path, group="dual", name="run_dual", config_name="dual_model_config.json")

    bare = tmp_path / "backbone" / "run_bare"
    bare.mkdir(parents=True)
    (bare / "best_model.pt").write_bytes(b"x")

    _loadable_run(tmp_path, group="profile_ae", name="run_ae", config_name="profile_autoencoder_config.json")

    out = ModelProbe(SilentLogger()).runs(str(tmp_path))

    assert out["ok"] is True
    assert [run["id"] for run in out["runs"]] == sorted([str(backbone), str(dual)], reverse=True)


def test_runs_reports_root_errors(tmp_path):
    out = ModelProbe(SilentLogger()).runs(str(tmp_path / "nowhere"))

    assert out["ok"] is False
    assert out["runs"] == []


def test_load_refuses_a_directory_that_is_not_a_run(tmp_path):
    probe = ModelProbe(SilentLogger())

    missing = probe.start_load(str(tmp_path / "nowhere"))
    assert missing["ok"] is False
    assert "not a directory" in missing["error"]

    (tmp_path / "runs_root").mkdir()
    rootish = probe.start_load(str(tmp_path / "runs_root"))
    assert rootish["ok"] is False
    assert "best_model.pt" in rootish["error"]


def test_load_refuses_a_run_without_the_persisted_config(tmp_path):
    run_dir = tmp_path / "run_bare"
    run_dir.mkdir()
    (run_dir / "best_model.pt").write_bytes(b"x")

    out = ModelProbe(SilentLogger()).start_load(str(run_dir))

    assert out["ok"] is False
    assert "meta/model_config.json" in out["error"]
    assert "meta/dual_model_config.json" in out["error"]
    assert "backbone and dual" in out["error"]


def test_loader_class_picks_the_dual_loader_for_dual_runs():
    from pipelines.backbone.inference.loader import RunLoader
    from pipelines.dual.inference.loader     import DualRunLoader

    probe = ModelProbe(SilentLogger())

    assert probe._loader_class(False) is RunLoader
    assert probe._loader_class(True)  is DualRunLoader


def test_model_label_names_the_dual_trunks():
    probe  = ModelProbe(SilentLogger())
    config = SimpleNamespace(params_backbone="unet_skip", existence_backbone="resunet")
    run    = SimpleNamespace(backbone_name="dual_resunet", model=SimpleNamespace(module=SimpleNamespace(config=config)))

    assert probe._model_label(run, False) == "dual_resunet"
    assert probe._model_label(run, True)  == "dual_resunet (unet_skip + resunet)"
