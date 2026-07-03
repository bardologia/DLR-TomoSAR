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
from configuration.training  import BackboneEntryConfig, JepaEntryConfig, ProfileAeEntryConfig, ImageAeEntryConfig

_TRAINING_PAGES = [
    ("train_backbone",            BackboneEntryConfig),
    ("train_jepa",                JepaEntryConfig),
    ("train_profile_autoencoder", ProfileAeEntryConfig),
    ("train_image_autoencoder",   ImageAeEntryConfig),
]


@pytest.mark.parametrize("key, entry_config", _TRAINING_PAGES)
def test_training_layout_claims_every_entry_field_exactly_once(key, entry_config):
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(entry_config())]

    LaunchLayout().build(key, leaves)


@pytest.mark.parametrize("key, entry_config", _TRAINING_PAGES)
def test_vram_reservation_gate_present_on_training_pages(key, entry_config):
    leaves = [{"path": path} for path, _value in ConfigCli._leaves(entry_config())]
    layout = LaunchLayout().build(key, leaves)

    claims = LaunchLayout()._claims(layout)

    assert "pretrain.reserve_vram"      in claims
    assert "pretrain.vram_keep_free_gb" in claims
