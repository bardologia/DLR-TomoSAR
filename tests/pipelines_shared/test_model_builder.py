from __future__ import annotations

import pytest

from pipelines.shared.model.model_builder import ModelBuilder


def test_config_from_registry_applies_known_overrides():
    config = ModelBuilder.config_from_registry("resunet", {"dropout": 0.25})

    assert config.dropout == 0.25


def test_config_from_registry_rejects_unknown_override():
    with pytest.raises(AttributeError, match="encodr_lr"):
        ModelBuilder.config_from_registry("resunet", {"encodr_lr": 1e-4})
