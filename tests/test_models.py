from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from models import MODEL_REGISTRY, get_model

BASELINE_PATH = Path(__file__).resolve().parent / "state_dict_baseline.json"


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

    @pytest.mark.parametrize("name", sorted(MODEL_REGISTRY.keys()))
    def test_state_dict_matches_baseline(self, name: str, baseline: dict):
        if name not in baseline:
            pytest.fail(f"Model '{name}' missing from baseline. Regenerate with scripts/generate_state_dict_baseline.py")

        model, _  = get_model(name)
        signature = StateDictSignature.compute(model)

        assert signature["num_keys"] == baseline[name]["num_keys"], f"state_dict key count changed for '{name}'"
        assert signature["num_parameters"] == baseline[name]["num_parameters"], f"parameter count changed for '{name}'"
        assert signature["signature"] == baseline[name]["signature"], f"state_dict keys or shapes changed for '{name}': checkpoints trained before this change will not load"

    def test_baseline_covers_full_registry(self, baseline: dict):
        assert set(baseline.keys()) == set(MODEL_REGISTRY.keys())


class TestForwardPass:
    @pytest.mark.parametrize("name", sorted(MODEL_REGISTRY.keys()))
    def test_forward_is_finite(self, name: str):
        torch.manual_seed(0)

        model, config = get_model(name)
        model.eval()

        spatial = getattr(config, "image_size", 64)
        x       = torch.randn(1, config.in_channels, spatial, spatial)

        with torch.no_grad():
            output = model(x)

        tensors = output if isinstance(output, (tuple, list)) else [output]
        for tensor in tensors:
            assert torch.all(torch.isfinite(tensor)), f"non-finite forward output for '{name}'"
