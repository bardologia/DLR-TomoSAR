from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from configuration.param_extraction import (
    FitMode,
    FitConfig,
    FitSettings,
    ExtractionConfig,
    ExtractParamsEntryConfig,
)

from pipelines.processing.param_extraction.pipeline import ExtractionPathResolver


def test_fit_config_alias_points_to_sigma_only():
    assert FitConfig is FitMode.SigmaOnly


def test_sigma_only_defaults_sane():
    cfg = FitMode.SigmaOnly()
    assert cfg.k_max > 0
    assert cfg.sigma_init_divisor > 0
    assert cfg.lambda_k > 0
    assert 0.0 < cfg.threshold_factor < 1.0


def test_sigma_only_asdict_round_trips():
    cfg     = FitMode.SigmaOnly()
    payload = dataclasses.asdict(cfg)
    assert FitMode.SigmaOnly(**payload) == cfg


def test_fit_settings_parameters_per_profile():
    cfg = FitSettings()
    assert cfg.parameters_per_profile == 3 * cfg.fit_config.k_max
    assert cfg.fitting_method == "sigma_only_adam"
    assert cfg.max_fit_iterations > 0


def test_fit_settings_default_factory_independent():
    a = FitSettings()
    b = FitSettings()
    assert a.fit_config is not b.fit_config


def test_extraction_config_paths_derived():
    cfg = ExtractionConfig(processed_data_path=Path("/tmp/run"))
    assert cfg.data_directory     == Path("/tmp/run/data")
    assert cfg.metadata_directory == Path("/tmp/run/meta")
    assert cfg.parameters_npy_path.name   == "parameters.npy"
    assert cfg.diagnostics_npz_path.name  == "fit_diagnostics.npz"


def test_extraction_config_default_suffix_encodes_fit():
    cfg = ExtractionConfig(processed_data_path=Path("/tmp/run"))
    suffix = cfg.output_suffix_value
    assert suffix.startswith("sigmaonly_k")
    assert f"k{cfg.fit_settings.fit_config.k_max}" in suffix
    assert "sig" in suffix
    assert "lam" in suffix


def test_extraction_config_explicit_suffix_used():
    cfg = ExtractionConfig(processed_data_path=Path("/tmp/run"), output_suffix="custom")
    assert cfg.output_suffix_value == "custom"
    assert cfg.output_subdir_name == "params_custom"


def test_extraction_config_output_directory_layout():
    cfg = ExtractionConfig(processed_data_path=Path("/tmp/run"))
    assert cfg.output_directory.parent.name == "params"
    assert cfg.output_directory.name == cfg.output_subdir_name


def test_extraction_config_matches_param_extraction_meta_suffix(param_extraction_meta):
    cfg = ExtractionConfig(
        processed_data_path = Path("/tmp/run"),
        fit_settings        = FitSettings(fit_config=FitMode.SigmaOnly(k_max=5, sigma_init_divisor=4.0, lambda_k=1e-2)),
    )
    assert cfg.output_subdir_name == "params_sigmaonly_k5_sig4_lam0p01"


def test_extraction_discover_height_range_from_state(tmp_path):
    import json

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "config_state.json").write_text(json.dumps({"tomogram_config": {"height_range": [-20.0, 80.0]}}))

    cfg = ExtractionConfig(processed_data_path=tmp_path)
    assert ExtractionPathResolver(cfg).discover_height_range() == (-20.0, 80.0)


def test_extraction_discover_height_range_override():
    cfg = ExtractionConfig(processed_data_path=Path("/tmp/run"), height_range=(0.0, 50.0))
    assert ExtractionPathResolver(cfg).discover_height_range() == (0.0, 50.0)


def test_extraction_discover_height_range_missing_raises(tmp_path):
    cfg = ExtractionConfig(processed_data_path=tmp_path)
    with pytest.raises(FileNotFoundError):
        ExtractionPathResolver(cfg).discover_height_range()


def test_extraction_discover_tomogram_missing_raises(tmp_path):
    (tmp_path / "data").mkdir()
    cfg = ExtractionConfig(processed_data_path=tmp_path)
    with pytest.raises(FileNotFoundError):
        ExtractionPathResolver(cfg).discover_tomogram_path()


def test_extract_params_entry_defaults():
    cfg = ExtractParamsEntryConfig()
    assert cfg.fit_k_max > 0
    assert cfg.parameter_workers > 0
    assert cfg.range_batch_size > 0
    assert isinstance(cfg.gpu_device_ids, list)
    assert isinstance(cfg.dataset_filter, list)


def test_extract_params_entry_default_lists_independent():
    a = ExtractParamsEntryConfig()
    b = ExtractParamsEntryConfig()
    a.gpu_device_ids.append(99)
    assert b.gpu_device_ids == [0, 1, 2, 3]
