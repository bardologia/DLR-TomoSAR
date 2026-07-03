from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from launch_layout            import LaunchLayout
from tools.runtime.config_cli import ConfigCli
from configuration.benchmark.general        import BenchmarkConfig
from configuration.cross_validation.general import CrossValidationConfig
from configuration.training                 import BackboneEntryConfig, JepaEntryConfig, ProfileAeEntryConfig, ImageAeEntryConfig
from configuration.tuning.general           import TuningEntryConfig

_TRAINING_PAGES = [
    ("train_backbone",            BackboneEntryConfig),
    ("train_jepa",                JepaEntryConfig),
    ("train_profile_autoencoder", ProfileAeEntryConfig),
    ("train_image_autoencoder",   ImageAeEntryConfig),
    ("benchmark",                 BenchmarkConfig),
    ("cross_validate",            CrossValidationConfig),
    ("tune",                      TuningEntryConfig),
]


@pytest.mark.parametrize("key, flow_config", _TRAINING_PAGES)
def test_training_layout_claims_every_config_field_exactly_once(key, flow_config):
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]

    LaunchLayout().build(key, leaves)


@pytest.mark.parametrize("key, flow_config", _TRAINING_PAGES)
def test_vram_reservation_gate_present_on_training_pages(key, flow_config):
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(flow_config())]
    layout = LaunchLayout().build(key, leaves)

    claims = LaunchLayout()._claims(layout)

    assert "training.reserve_vram"      in claims
    assert "training.vram_keep_free_gb" in claims
