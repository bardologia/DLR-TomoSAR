from __future__ import annotations

import pytest

from pipelines.shared.model.model_builder import ModelBuilder


def test_config_from_registry_applies_known_overrides():
    config = ModelBuilder.config_from_registry("resunet", {"dropout": 0.25})

    assert config.dropout == 0.25


def test_config_from_registry_rejects_unknown_override():
    with pytest.raises(AttributeError, match="encodr_lr"):
        ModelBuilder.config_from_registry("resunet", {"encodr_lr": 1e-4})


def test_image_size_override_empty_for_convolutional_backbone():
    assert ModelBuilder.image_size_override("resunet", (64, 128)) == {}


def test_image_size_override_square_patch_yields_scalar():
    assert ModelBuilder.image_size_override("swin_unet", (64, 64)) == {"image_size": 64}


def test_image_size_override_rejects_rectangular_patch():
    with pytest.raises(ValueError, match="rectangular"):
        ModelBuilder.image_size_override("unetr", (64, 128))
