from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

from tools.runtime.config_cli import ConfigCli


_MAIN_DIR = Path(__file__).resolve().parents[2] / "main"

THREAD_KEYS = ("MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS")

DEFER_HEAVY_IMPORTS = (
    "train_backbone",
    "train_jepa",
    "train_profile_autoencoder",
    "train_image_autoencoder",
    "infer_backbone",
    "infer_profile_autoencoder",
    "infer_image_autoencoder",
    "generate_interferograms",
    "generate_tomogram",
    "tune",
    "benchmark",
    "cross_validate",
    "tune_dataloader",
    "pre_process",
    "extract_params",
)

CLI_MODULES = (
    "infer_backbone",
    "infer_profile_autoencoder",
    "infer_image_autoencoder",
    "pre_process",
    "extract_params",
    "tune",
    "tune_dataloader",
    "benchmark",
    "cross_validate",
    "compare_runs",
)

ENTRY_CONFIGS = {
    "infer_backbone"            : ("configuration.inference",          "InferenceEntryConfig"),
    "infer_profile_autoencoder" : ("configuration.inference",          "InferenceEntryConfig"),
    "infer_image_autoencoder"   : ("configuration.inference",          "InferenceEntryConfig"),
    "pre_process"    : ("configuration.sar.processing_config",        "PreProcessEntryConfig"),
    "extract_params" : ("configuration.param_extraction", "ExtractParamsEntryConfig"),
    "tune"           : ("configuration.tuning",                       "TuningEntryConfig"),
    "tune_dataloader": ("configuration.benchmark.dataloader_tuning",  "DataLoaderTuningEntryConfig"),
    "benchmark"      : ("configuration.benchmark",                    "BenchmarkConfig"),
    "cross_validate" : ("configuration.cross_validation",             "CrossValidationConfig"),
    "compare_runs"   : ("configuration.benchmark",                    "BenchmarkConfig"),
}


@pytest.fixture
def main_on_path(monkeypatch):
    if str(_MAIN_DIR) not in sys.path:
        monkeypatch.syspath_prepend(str(_MAIN_DIR))
    return _MAIN_DIR


@pytest.fixture
def frozen_env(monkeypatch):
    snapshot = dict(os.environ)
    yield
    for key in set(os.environ) - set(snapshot):
        monkeypatch.delenv(key, raising=False)
    for key, value in snapshot.items():
        monkeypatch.setenv(key, value)


def _import_main(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.mark.parametrize("name", DEFER_HEAVY_IMPORTS)
def test_module_imports_without_error(name, main_on_path, frozen_env):
    module = _import_main(name)

    assert module is not None


@pytest.mark.parametrize("name", DEFER_HEAVY_IMPORTS)
def test_module_exposes_main_callable(name, main_on_path, frozen_env):
    module = _import_main(name)

    assert callable(module.main)


@pytest.mark.parametrize("name", DEFER_HEAVY_IMPORTS)
def test_module_guards_execution_behind_name_main(name, main_on_path, frozen_env):
    source = (_MAIN_DIR / f"{name}.py").read_text()

    assert 'if __name__ == "__main__":' in source


def test_import_does_not_set_cuda_visible_devices(main_on_path, frozen_env, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    for name in DEFER_HEAVY_IMPORTS:
        _import_main(name)

    assert "CUDA_VISIBLE_DEVICES" not in os.environ


@pytest.mark.parametrize("name", ("train_backbone", "train_jepa", "train_profile_autoencoder", "train_image_autoencoder"))
def test_train_main_defers_heavy_imports(name, main_on_path, frozen_env):
    source = (_MAIN_DIR / f"{name}.py").read_text()

    assert "from pipelines" not in source.split("def main")[0]
    assert "from pipelines" in source.split("def main", 1)[1]


@pytest.mark.parametrize("name", DEFER_HEAVY_IMPORTS)
def test_compare_runs_is_the_only_module_pinning_threads_at_import(name, main_on_path, frozen_env, monkeypatch):
    for key in THREAD_KEYS:
        monkeypatch.delenv(key, raising=False)

    _import_main(name)

    pinned = all(os.environ.get(key) == "4" for key in THREAD_KEYS)
    if name == "compare_runs":
        return
    assert not pinned or name in {"compare_runs"}


def test_compare_runs_has_no_module_level_side_effects(main_on_path, monkeypatch):
    for key in THREAD_KEYS:
        monkeypatch.delenv(key, raising=False)

    sys.modules.pop("compare_runs", None)
    importlib.import_module("compare_runs")

    assert all(key not in os.environ for key in THREAD_KEYS)


def test_compare_runs_exposes_main_callable(main_on_path, frozen_env):
    module = _import_main("compare_runs")

    assert callable(module.main)


@pytest.mark.parametrize("name", CLI_MODULES)
def test_entry_config_constructs_from_defaults(name):
    module_path, class_name = ENTRY_CONFIGS[name]
    config_class = getattr(importlib.import_module(module_path), class_name)

    config = config_class()

    assert config is not None


@pytest.mark.parametrize("name", CLI_MODULES)
def test_config_cli_builds_parser_for_entry_config(name):
    module_path, class_name = ENTRY_CONFIGS[name]
    config_class = getattr(importlib.import_module(module_path), class_name)

    cli = ConfigCli(config_class(), description=f"{name} cli")

    assert cli.parser is not None


@pytest.mark.parametrize("name", CLI_MODULES)
def test_config_cli_help_config_exits_zero(name):
    module_path, class_name = ENTRY_CONFIGS[name]
    config_class = getattr(importlib.import_module(module_path), class_name)

    cli = ConfigCli(config_class())

    with pytest.raises(SystemExit) as excinfo:
        cli.apply(["--help-config"])

    assert excinfo.value.code == 0


@pytest.mark.parametrize("name", CLI_MODULES)
def test_config_cli_apply_no_args_returns_unmodified_config(name):
    module_path, class_name = ENTRY_CONFIGS[name]
    config_class = getattr(importlib.import_module(module_path), class_name)

    config = config_class()
    result = ConfigCli(config).apply([])

    assert result is config


def test_config_cli_rejects_unknown_override():
    from configuration.inference import InferenceEntryConfig

    cli = ConfigCli(InferenceEntryConfig())

    with pytest.raises(ValueError):
        cli.apply(["--not-a-real-flag", "1"])


def test_infer_worker_requires_run_dir_and_config(main_on_path, frozen_env, monkeypatch):
    module = _import_main("infer_backbone")
    monkeypatch.setattr(sys, "argv", ["infer_backbone.py", "--worker", "--run-dir", "x"])

    with pytest.raises(SystemExit):
        module.main()


def test_tune_worker_requires_model(main_on_path, frozen_env, monkeypatch):
    module = _import_main("tune")
    monkeypatch.setattr(sys, "argv", ["tune.py", "--worker"])

    with pytest.raises(SystemExit):
        module.main()


def test_benchmark_worker_requires_model_and_run_tag(main_on_path, frozen_env, monkeypatch):
    module = _import_main("benchmark")
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--worker", "train"])

    with pytest.raises(SystemExit):
        module.main()


def test_cross_validate_worker_requires_fold(main_on_path, frozen_env, monkeypatch):
    module = _import_main("cross_validate")
    monkeypatch.setattr(sys, "argv", ["cross_validate.py", "--worker", "train"])

    with pytest.raises(SystemExit):
        module.main()


def test_cross_validate_infer_worker_requires_split(main_on_path, frozen_env, monkeypatch):
    module = _import_main("cross_validate")
    monkeypatch.setattr(sys, "argv", ["cross_validate.py", "--worker", "infer", "--fold", "0", "--run-tag", "t"])

    with pytest.raises(SystemExit):
        module.main()
