from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from configuration.inference import InferenceConfig
from pipelines.backbone.inference.loader import RunLoader
from pipelines.backbone.inference.model_wrapper import ModelWrapper
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata


def _write_fake_checkpoint(path: Path, x_axis: np.ndarray) -> dict:
    state = {
        "weight": torch.zeros(2, 2),
        "bias":   torch.zeros(2),
    }
    ckpt = {
        "params":        state,
        "x_axis":        x_axis,
        "epoch":         7,
        "best_val_loss": 0.125,
        "best_epoch":    5,
    }
    torch.save(ckpt, str(path))
    return ckpt


def test_load_checkpoint_numpy_x_axis(tmp_path):
    x_axis    = np.linspace(-20.0, 80.0, 32).astype(np.float32)
    ckpt_path = tmp_path / "best_model.pt"
    _write_fake_checkpoint(ckpt_path, x_axis)

    loader = RunLoader.__new__(RunLoader)
    ckpt, loaded_axis, meta = loader._load_checkpoint(ckpt_path, "cpu")

    assert isinstance(loaded_axis, np.ndarray)
    assert loaded_axis.dtype == np.float32
    assert np.allclose(loaded_axis, x_axis)
    assert meta["epoch"]         == 7
    assert meta["best_epoch"]    == 5
    assert meta["best_val_loss"] == pytest.approx(0.125)
    assert "params" in ckpt


def test_load_checkpoint_torch_tensor_x_axis(tmp_path):
    x_axis    = torch.linspace(-20.0, 80.0, 16)
    ckpt_path = tmp_path / "ckpt.pt"
    _write_fake_checkpoint(ckpt_path, x_axis)

    loader = RunLoader.__new__(RunLoader)
    _, loaded_axis, meta = loader._load_checkpoint(ckpt_path, "cpu")

    assert isinstance(loaded_axis, np.ndarray)
    assert loaded_axis.dtype == np.float32
    assert loaded_axis.size  == 16
    assert np.allclose(loaded_axis, x_axis.numpy())


def test_load_checkpoint_uses_weights_only_false(tmp_path):
    x_axis    = np.arange(8, dtype=np.float32)
    ckpt_path = tmp_path / "obj.pt"
    ckpt      = {
        "params":        {"w": torch.ones(1)},
        "x_axis":        x_axis,
        "epoch":         0,
        "best_val_loss": 1.0,
        "best_epoch":    0,
        "extra":         np.array([1, 2, 3]),
    }
    torch.save(ckpt, str(ckpt_path))

    loader = RunLoader.__new__(RunLoader)
    loaded, axis, _ = loader._load_checkpoint(ckpt_path, "cpu")

    assert np.array_equal(loaded["extra"], np.array([1, 2, 3]))
    assert axis.size == 8


def test_model_wrapper_identity_no_normalizer():
    class _Echo(torch.nn.Module):
        def forward(self, x):
            return x * 2.0

    wrapper = ModelWrapper(_Echo(), "cpu")
    out     = wrapper(np.ones((1, 3, 4, 4), dtype=np.float32))

    assert isinstance(out, np.ndarray)
    assert np.allclose(out, 2.0)


def test_model_wrapper_denormalize_passthrough_without_clamp():
    class _Echo(torch.nn.Module):
        def forward(self, x):
            return x

    wrapper = ModelWrapper(_Echo(), "cpu", x_axis=None, amp_max=None, normalizer=None)
    t       = torch.arange(6.0).reshape(1, 6, 1, 1)

    assert torch.allclose(wrapper.denormalize_output(t), t)


def test_inference_metadata_paths_with_output_subdir(tmp_path):
    cfg  = InferenceConfig(run_directory=tmp_path, output_subdir="my_run", device="cpu")
    meta = InferenceMetadata(cfg)

    assert meta.output_dir == tmp_path / "inference" / "my_run"
    assert meta.figures_dir.parent    == meta.output_dir
    assert meta.figure_path("x", "png") == meta.figures_dir / "x.png"


def test_inference_metadata_create_dirs(tmp_path):
    cfg  = InferenceConfig(run_directory=tmp_path, output_subdir="run0", device="cpu")
    meta = InferenceMetadata(cfg)
    meta.create_dirs()

    assert meta.output_dir.is_dir()
    assert meta.figures_dir.is_dir()
    assert meta.animations_dir.is_dir()
    assert meta.logs_dir.is_dir()
    assert meta.cube_dir.is_dir()


def test_full_run_load_skips_without_run_dir():
    candidate = Path("/ste/rnd/User/vice_vi")
    if candidate.is_dir():
        pytest.skip("server run directory present; full RunLoader.load covered elsewhere")
    pytest.skip("no real trained run directory available locally; RunLoader.load requires meta/ + checkpoint + preprocessing run")
