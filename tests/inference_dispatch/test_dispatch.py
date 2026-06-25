from __future__ import annotations

import pytest

from configuration.inference            import InferenceEntryConfig
from pipelines.shared.inference_dispatch import InferenceDispatcher, RunType


def _make_run(tmp_path, *filenames):
    meta = tmp_path / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (meta / name).write_text("{}")
    return tmp_path


def _dispatcher():
    return InferenceDispatcher(InferenceEntryConfig())


def test_classify_backbone(tmp_path):
    run = _make_run(tmp_path, "model_config.json")
    assert _dispatcher().classify(run) == RunType.BACKBONE


def test_classify_profile_autoencoder(tmp_path):
    run = _make_run(tmp_path, "profile_autoencoder_config.json")
    assert _dispatcher().classify(run) == RunType.PROFILE_AE


def test_classify_image_autoencoder(tmp_path):
    run = _make_run(tmp_path, "image_autoencoder_config.json")
    assert _dispatcher().classify(run) == RunType.IMAGE_AE


def test_classify_jepa_run_routes_to_backbone(tmp_path):
    run = _make_run(tmp_path, "model_config.json", "profile_autoencoder_config.json")
    assert _dispatcher().classify(run) == RunType.BACKBONE


def test_classify_unrecognized_run_raises(tmp_path):
    run = _make_run(tmp_path)
    with pytest.raises(ValueError):
        _dispatcher().classify(run)
