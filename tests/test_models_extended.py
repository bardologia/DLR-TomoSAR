from __future__ import annotations

import numpy as np
import pytest
import torch

from models import (
    CONFIG_REGISTRY,
    MODEL_REGISTRY,
    UNetPlusPlusConfig,
    get_model,
)

ALL_NAMES = sorted(MODEL_REGISTRY.keys())

_MODEL_CACHE: dict = {}


def cached_model(name: str):
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = get_model(name)
    return _MODEL_CACHE[name]


def seed_everything(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_input(config, spatial: int, batch: int = 1) -> torch.Tensor:
    return torch.randn(batch, config.in_channels, spatial, spatial, device="cpu")


def as_tensor_list(output):
    if isinstance(output, (tuple, list)):
        return list(output)
    return [output]


class TestRegistryIntegrity:
    def test_model_and_config_registries_have_identical_keys(self):
        assert set(MODEL_REGISTRY.keys()) == set(CONFIG_REGISTRY.keys())

    def test_registries_are_not_empty(self):
        assert len(MODEL_REGISTRY) > 0
        assert len(CONFIG_REGISTRY) == len(MODEL_REGISTRY)

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_registry_values_are_classes(self, name: str):
        assert isinstance(MODEL_REGISTRY[name], type)
        assert isinstance(CONFIG_REGISTRY[name], type)

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_get_model_returns_instance_of_registered_class(self, name: str):
        model, config = cached_model(name)

        assert isinstance(model, MODEL_REGISTRY[name])
        assert isinstance(config, CONFIG_REGISTRY[name])
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_get_model_uses_provided_config(self, name: str):
        config = CONFIG_REGISTRY[name]()
        model, returned_config = get_model(name, config=config)

        assert returned_config is config

    def test_get_model_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError):
            get_model("definitely_not_a_real_model")

    @pytest.mark.parametrize(
        "alias, canonical",
        [
            ("UNet", "unet"),
            ("attention-unet", "attention_unet"),
            ("Swin Unet", "swin_unet"),
        ],
    )
    def test_get_model_normalizes_name_casing_and_separators(self, alias: str, canonical: str):
        model, _ = get_model(alias)

        assert isinstance(model, MODEL_REGISTRY[canonical])

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_overrides_apply_to_generated_config(self, name: str):
        model, config = get_model(name, init_mode="kaiming")

        assert config.init_mode == "kaiming"

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_overrides_apply_to_existing_config(self, name: str):
        base = CONFIG_REGISTRY[name]()
        model, config = get_model(name, config=base, init_mode="kaiming")

        assert config.init_mode == "kaiming"
        assert config is base

    def test_unknown_override_on_existing_config_is_ignored(self):
        base = CONFIG_REGISTRY["unet"]()
        model, config = get_model("unet", config=base, not_a_real_field=123)

        assert not hasattr(config, "not_a_real_field")


class TestOutputShapeContracts:
    @pytest.mark.parametrize("name", ALL_NAMES)
    @pytest.mark.parametrize("spatial", [32, 64])
    def test_output_spatial_shape_matches_input(self, name: str, spatial: int):
        seed_everything()

        model, config = cached_model(name)
        model.eval()

        x = make_input(config, spatial)
        with torch.no_grad():
            output = model(x)

        tensors = as_tensor_list(output)
        for tensor in tensors:
            assert tensor.shape[0] == 1
            assert tensor.shape[1] == config.out_channels
            assert tensor.shape[2] == spatial
            assert tensor.shape[3] == spatial
            assert torch.all(torch.isfinite(tensor))

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_output_dtype_is_float32(self, name: str):
        seed_everything()

        model, config = cached_model(name)
        model.eval()

        x = make_input(config, 32)
        with torch.no_grad():
            output = model(x)

        for tensor in as_tensor_list(output):
            assert tensor.dtype == torch.float32

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_batch_dimension_is_preserved(self, name: str):
        seed_everything()

        model, config = cached_model(name)
        model.eval()

        x = make_input(config, 32, batch=2)
        with torch.no_grad():
            output = model(x)

        for tensor in as_tensor_list(output):
            assert tensor.shape[0] == 2

class TestDeepSupervision:
    def test_unetplusplus_deep_supervision_returns_list(self):
        seed_everything()

        config = UNetPlusPlusConfig(deep_supervision=True)
        model, _ = get_model("unetplusplus", config=config)
        model.eval()

        x = make_input(config, 32)
        with torch.no_grad():
            output = model(x)

        assert isinstance(output, list)
        assert len(output) > 1
        for tensor in output:
            assert tensor.shape[1] == config.out_channels
            assert tensor.shape[2] == 32
            assert tensor.shape[3] == 32

    def test_unetplusplus_single_output_when_supervision_disabled(self):
        seed_everything()

        config = UNetPlusPlusConfig(deep_supervision=False)
        model, _ = get_model("unetplusplus", config=config)
        model.eval()

        x = make_input(config, 32)
        with torch.no_grad():
            output = model(x)

        assert torch.is_tensor(output)


class TestGradientFlow:
    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_backward_produces_finite_grads_with_nonzero_early_and_late(self, name: str):
        seed_everything()

        model, config = get_model(name)
        model.train()

        x = make_input(config, 32, batch=2)
        output = model(x)
        loss = sum(tensor.pow(2).mean() for tensor in as_tensor_list(output))
        loss.backward()

        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        assert len(named) >= 2

        missing = [n for n, p in named if p.grad is None]
        assert missing == [], f"parameters without grad in '{name}': {missing}"

        early_name, early_param = named[0]
        late_name, late_param   = named[-1]

        for label, param in ((early_name, early_param), (late_name, late_param)):
            assert torch.all(torch.isfinite(param.grad)), f"non-finite grad for '{label}' in '{name}'"
            assert float(param.grad.abs().sum()) > 0.0, f"all-zero grad for '{label}' in '{name}'"


class TestTrainEvalModeSwitching:
    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_train_then_eval_sets_training_flag(self, name: str):
        model, _ = cached_model(name)

        model.train()
        assert model.training is True
        assert all(m.training for m in model.modules())

        model.eval()
        assert model.training is False
        assert not any(m.training for m in model.modules())

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_eval_mode_is_deterministic(self, name: str):
        seed_everything()

        model, config = cached_model(name)
        model.eval()

        x = make_input(config, 32)
        with torch.no_grad():
            first  = as_tensor_list(model(x))
            second = as_tensor_list(model(x))

        for a, b in zip(first, second):
            assert torch.allclose(a, b)

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_dropout_modules_track_mode(self, name: str):
        model, _ = cached_model(name)

        dropouts = [m for m in model.modules() if isinstance(m, torch.nn.modules.dropout._DropoutNd)]
        if not dropouts:
            pytest.skip(f"'{name}' has no dropout modules")

        model.train()
        assert all(d.training for d in dropouts)

        model.eval()
        assert not any(d.training for d in dropouts)
