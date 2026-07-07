from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from launch_layout            import LaunchLayout
from project_paths            import ProjectPaths
from script_catalog           import ScriptCatalog
from tools.runtime.config_cli import ConfigCli
from configuration.benchmark.general        import BenchmarkConfig
from configuration.comparison               import ComparisonEntryConfig
from configuration.cross_validation.general import CrossValidationConfig
from configuration.diagnostics              import TensorboardExportEntryConfig
from configuration.patch_sweep.general      import PatchSweepConfig
from configuration.training                 import BackboneEntryConfig, JepaEntryConfig, ProfileAeEntryConfig, ImageAeEntryConfig, UnrolledEntryConfig
from configuration.tuning.general           import TuningEntryConfig
from pipelines.backbone.training.loss_terms import LossComponentCatalog

_DISPATCH_ONLY = {"generate_tomogram", "generate_interferograms"}

_TRAINING_PAGES = [
    ("train_backbone",            BackboneEntryConfig),
    ("train_jepa",                JepaEntryConfig),
    ("train_profile_autoencoder", ProfileAeEntryConfig),
    ("train_image_autoencoder",   ImageAeEntryConfig),
    ("train_unrolled",            UnrolledEntryConfig),
    ("benchmark",                 BenchmarkConfig),
    ("cross_validate",            CrossValidationConfig),
    ("sweep_patches",             PatchSweepConfig),
    ("tune",                      TuningEntryConfig),
]


@pytest.mark.parametrize("key, flow_config", _TRAINING_PAGES)
def test_training_layout_claims_every_config_field_exactly_once(key, flow_config):
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]

    LaunchLayout().build(key, leaves)


def test_sweep_loss_choices_match_the_component_catalog():
    choices = {choice["value"] for choice in LaunchLayout.MULTI_SWEEP_LOSSES["choices"]}

    assert choices == set(LossComponentCatalog.names())


def test_compare_runs_layout_claims_every_config_field_exactly_once():
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(ComparisonEntryConfig())]

    LaunchLayout().build("compare_runs", leaves)


def test_export_tensorboard_plots_layout_claims_every_config_field_exactly_once():
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(TensorboardExportEntryConfig())]

    LaunchLayout().build("export_tensorboard_plots", leaves)


def test_every_registered_script_is_reachable_from_the_catalog():
    members = {member for group in ScriptCatalog.GROUPS.values() for member, _label in group["members"]}
    pages   = set(ScriptCatalog.META) | members

    assert set(ProjectPaths.SCRIPT_DIRS) - _DISPATCH_ONLY == pages
    assert pages <= set(LaunchLayout.LAYOUTS)


@pytest.mark.parametrize("key, flow_config", _TRAINING_PAGES)
def test_vram_reservation_gate_present_on_training_pages(key, flow_config):
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]
    layout = LaunchLayout().build(key, leaves)

    claims = LaunchLayout()._claims(layout)

    assert "training.reserve_vram"      in claims
    assert "training.vram_keep_free_gb" in claims
