from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from models import (
    BACKBONE_MODEL_REGISTRY,
    IMAGE_AE_MODEL_REGISTRY,
    PROFILE_AE_MODEL_REGISTRY,
    UNROLLED_MODEL_REGISTRY,
    get_backbone,
    get_image_autoencoder,
    get_profile_autoencoder,
    get_unrolled,
)

BASELINE_PATH = Path(__file__).resolve().parents[1] / "state_dict_baseline.json"

MODEL_FAMILIES = {
    "backbone"            : (BACKBONE_MODEL_REGISTRY,   get_backbone),
    "profile_autoencoder" : (PROFILE_AE_MODEL_REGISTRY, get_profile_autoencoder),
    "image_autoencoder"   : (IMAGE_AE_MODEL_REGISTRY,   get_image_autoencoder),
    "unrolled"            : (UNROLLED_MODEL_REGISTRY,   get_unrolled),
}

FAMILY_MODEL_CASES = [(family, name) for family, (registry, _factory) in MODEL_FAMILIES.items() for name in sorted(registry)]


class StateDictSignature:
    @staticmethod
    def compute(model: torch.nn.Module) -> dict:
        state_dict = model.state_dict()
        lines      = [f"{key} {tuple(tensor.shape)}" for key, tensor in state_dict.items()]
        digest     = hashlib.sha256("\n".join(lines).encode()).hexdigest()

        return {
            "signature"      : digest,
            "num_keys"       : len(state_dict),
            "num_parameters" : sum(p.numel() for p in model.parameters()),
        }


class TestStateDictCompatibility:
    @pytest.fixture(scope="class")
    def baseline(self) -> dict:
        if not BASELINE_PATH.exists():
            pytest.fail(f"Missing baseline {BASELINE_PATH}. Generate it with scripts/generate_state_dict_baseline.py")
        return json.loads(BASELINE_PATH.read_text())

    @pytest.mark.parametrize("family, name", FAMILY_MODEL_CASES)
    def test_state_dict_matches_baseline(self, family: str, name: str, baseline: dict):
        if name not in baseline.get(family, {}):
            pytest.fail(f"Model '{family}/{name}' missing from baseline. Regenerate with scripts/generate_state_dict_baseline.py")

        _registry, factory = MODEL_FAMILIES[family]
        model, _config     = factory(name)
        signature          = StateDictSignature.compute(model)
        expected           = baseline[family][name]

        assert signature["num_keys"] == expected["num_keys"], f"state_dict key count changed for '{family}/{name}'"
        assert signature["num_parameters"] == expected["num_parameters"], f"parameter count changed for '{family}/{name}'"
        assert signature["signature"] == expected["signature"], f"state_dict keys or shapes changed for '{family}/{name}': checkpoints trained before this change will not load"

    def test_baseline_covers_full_registries(self, baseline: dict):
        expected = {family: set(registry) for family, (registry, _factory) in MODEL_FAMILIES.items()}
        actual   = {family: set(models) for family, models in baseline.items()}
        assert actual == expected
