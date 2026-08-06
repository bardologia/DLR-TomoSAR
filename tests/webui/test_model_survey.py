from __future__ import annotations

import numpy as np
import pytest

from model_probe  import ModelProbe
from model_survey import ModelSurvey

from pipelines.backbone.inference.probes import PredictionCurves

from tests.conftest           import SilentLogger
from tests.webui._probe_fakes import N_AZ, N_RG, PH, PW, fake_run


def _survey(n_slots: int = 1, with_gt: bool = True) -> ModelSurvey:
    survey = ModelSurvey(SilentLogger())
    run    = fake_run(n_slots, with_gt)

    survey.loaded = {
        "run"      : run,
        "path"     : "fake",
        "split"    : "test",
        "device"   : "cpu",
        "labels"   : ["primary", "sec PS04"],
        "renderer" : PredictionCurves(n_slots, run.x_axis),
        "patch"    : (PH, PW),
        "backbone" : "bare",
    }
    survey.status = {"state": "running", "path": "fake", "progress": 0.0, "stage": "", "error": ""}

    return survey


def test_survey_covers_the_region_and_finishes():
    survey = _survey()
    survey._survey()

    assert survey.status["state"]    == "done"
    assert survey.status["progress"] == pytest.approx(1.0)

    result = survey.result
    assert result["region"]   == [N_AZ, N_RG]
    assert result["coverage"] == {"tiles": 4, "total_tiles": 4, "pixels": 256}
    assert result["channels"] == ["primary", "sec PS04"]
    assert result["seconds"]  > 0.0


def test_survey_fit_detection_and_matches():
    survey = _survey()
    survey._survey()

    result = survey.result

    fit = result["fit"]
    assert fit["curve_mse_gt"]["mean"] > 0.0
    assert fit["curve_mse_gt"]["p90"] >= fit["curve_mse_gt"]["median"]
    assert fit["curve_mse_raw"]["mean"] > 0.0
    assert fit["base_power"] > 0.0

    detection = result["detection"]
    assert detection["mean_pred_active"] == pytest.approx(1.0)
    assert detection["mean_gt_active"]   == pytest.approx(1.0)
    assert detection["count_match_frac"] == pytest.approx(1.0)

    matched = result["matched"]
    assert matched["n_matched"] == 256
    assert matched["mu"] > 1.0
    assert matched["sigma"] == pytest.approx(1.0, abs=1e-5)


def test_survey_ranks_ablation_and_attribution_on_the_used_channel():
    survey = _survey()
    survey._survey()

    result = survey.result

    channels = result["ablation"]["channels"]
    assert channels[0]["label"]      == "primary"
    assert channels[0]["delta_mse"]  > 1e-8
    assert channels[1]["delta_mse"]  == pytest.approx(0.0, abs=1e-12)
    assert channels[1]["flips_frac"] == pytest.approx(0.0)

    mu = {f["family"]: f for f in result["attribution"]["families"]}["mu"]
    assert mu["dead"] is False
    assert mu["shares"][0] == pytest.approx(1.0)

    sigma = {f["family"]: f for f in result["attribution"]["families"]}["sigma"]
    assert sigma["dead"] is True


def test_survey_flips_noise_occlusion_and_erf_profiles():
    survey = _survey()
    survey._survey()

    result = survey.result

    flips = {entry["flip"]: entry["delta_mse"] for entry in result["flips"]}
    assert flips["none"] == pytest.approx(0.0, abs=1e-12)
    assert flips["azimuth"] == pytest.approx(0.0, abs=1e-10)
    assert flips["range"]   == pytest.approx(0.0, abs=1e-10)

    noise = result["noise"]
    assert noise[0]["value"]     == pytest.approx(0.0)
    assert noise[0]["delta_mse"] == pytest.approx(0.0, abs=1e-12)
    assert noise[-1]["delta_mse"] > noise[1]["delta_mse"]

    occlusion = result["occlusion"]
    assert occlusion["occluder"] == [2, 2]
    assert occlusion["delta"][0] > 0.0
    assert occlusion["delta"][-1] == pytest.approx(0.0, abs=1e-12)

    erf = {f["family"]: f for f in result["erf"]["families"]}
    assert erf["mu"]["dead"] is False
    assert erf["mu"]["r50"] == 0
    assert erf["mu"]["samples"] == ModelSurvey.ERF_SAMPLES
    assert erf["sigma"]["dead"] is True

    vitals = result["vitals"]
    assert vitals["summary"]["n_layers"] == 1
    assert vitals["entries"][0]["type"]  == "Identity"


def test_survey_without_gt_omits_fit_gt_sections():
    survey = _survey(with_gt=False)
    survey._survey()

    result = survey.result
    assert result["fit"]["curve_mse_gt"] is None
    assert result["detection"] is None
    assert result["matched"]   is None


def test_survey_tile_cap_strides_evenly_and_reports_totals():
    survey          = _survey()
    survey.tile_cap = 2
    survey._survey()

    coverage = survey.result["coverage"]
    assert coverage["tiles"]       == 2
    assert coverage["total_tiles"] == 4
    assert coverage["pixels"]      == 128


def test_survey_cancel_stops_the_run():
    survey = _survey()
    survey.cancel_flag = True
    survey._survey()

    assert survey.status["state"] == "cancelled"
    assert survey.result is None


def test_survey_start_refuses_non_runs(tmp_path):
    survey = ModelSurvey(SilentLogger())

    out = survey.start({"path": str(tmp_path / "nowhere")})
    assert out["ok"] is False
    assert "not a directory" in out["error"]


def test_survey_result_before_any_run_fails():
    assert ModelSurvey(SilentLogger()).survey_result()["ok"] is False
    assert ModelSurvey(SilentLogger()).cancel()["ok"] is False
