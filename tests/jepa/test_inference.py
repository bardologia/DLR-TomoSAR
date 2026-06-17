from __future__ import annotations

import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from pipelines.backbone.dataset.spatial    import GridInfo
from pipelines.backbone.inference.loader    import ModelWrapper, RunLoader
from pipelines.jepa.inference.loader         import JepaInferenceModel, JepaRunLoader
from pipelines.jepa.inference.pipeline       import JEPA_INFERENCE_COMPONENTS, JEPA_PARAM_INFERENCE_COMPONENTS
from pipelines.jepa.inference.predictor      import JepaCurvePredictor
from pipelines.jepa.training.trainer         import JepaModule
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats

from tests.jepa.conftest import EMBEDDING_DIM, PROFILE_LENGTH, SPATIAL, make_autoencoder


def build_inference_module():
    autoencoder = make_autoencoder("l2")
    backbone    = nn.Conv2d(2, EMBEDDING_DIM, kernel_size=1)
    return JepaModule(backbone, profile_autoencoder=autoencoder, image_autoencoder=None)


def test_inference_components_wiring():
    assert JEPA_INFERENCE_COMPONENTS.loader_cls       is JepaRunLoader
    assert JEPA_INFERENCE_COMPONENTS.predictor_cls     is JepaCurvePredictor
    assert JEPA_INFERENCE_COMPONENTS.param_space       is False
    assert JEPA_PARAM_INFERENCE_COMPONENTS.param_space is True


def test_jepa_inference_model_outputs_denormalized_curve():
    module     = build_inference_module()
    normalizer = ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0))
    adapter    = JepaInferenceModel(module, normalizer).eval()
    images     = torch.randn(2, 2, SPATIAL, SPATIAL)

    with torch.no_grad():
        out = adapter(images)

    assert out.shape == (2, PROFILE_LENGTH, SPATIAL, SPATIAL)
    assert torch.isfinite(out).all()


def test_jepa_inference_model_curve_is_nonnegative():
    module     = build_inference_module()
    normalizer = ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0))
    adapter    = JepaInferenceModel(module, normalizer).eval()
    images     = torch.randn(2, 2, SPATIAL, SPATIAL)

    with torch.no_grad():
        out = adapter(images)

    assert (out >= 0.0).all()


def test_wrap_model_returns_model_wrapper_with_curve_output():
    module     = build_inference_module()
    normalizer = ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0))

    loader                    = JepaRunLoader.__new__(JepaRunLoader)
    loader.profile_normalizer = normalizer

    wrapper = loader._wrap_model(module, device="cpu", norm_stats=None, x_axis=None, amp_max=None)

    assert isinstance(wrapper, ModelWrapper)

    images = np.random.randn(2, 2, SPATIAL, SPATIAL).astype(np.float32)
    out    = wrapper(images)

    assert out.shape == (2, PROFILE_LENGTH, SPATIAL, SPATIAL)
    assert out.dtype == np.float32


def test_load_checkpoint_round_trip_numpy_x_axis(tmp_path):
    module = build_inference_module()
    x_axis = np.linspace(-4.0, 4.0, PROFILE_LENGTH).astype(np.float32)

    ckpt = {
        "epoch"         : 7,
        "best_val_loss" : 0.123,
        "best_epoch"    : 5,
        "x_axis"        : x_axis,
        "params"        : module.state_dict(),
    }
    path = tmp_path / "best_model.pt"
    torch.save(ckpt, path)

    loader              = JepaRunLoader.__new__(JepaRunLoader)
    loaded, axis, meta  = RunLoader._load_checkpoint(loader, path, "cpu")

    assert np.allclose(axis, x_axis)
    assert axis.dtype       == np.float32
    assert meta["epoch"]    == 7
    assert meta["best_epoch"] == 5
    assert meta["best_val_loss"] == pytest.approx(0.123)


def test_load_checkpoint_accepts_tensor_x_axis(tmp_path):
    module = build_inference_module()
    x_axis = torch.linspace(-4.0, 4.0, PROFILE_LENGTH)

    ckpt = {
        "epoch"         : 1,
        "best_val_loss" : 1.0,
        "best_epoch"    : 1,
        "x_axis"        : x_axis,
        "params"        : module.state_dict(),
    }
    path = tmp_path / "best_model.pt"
    torch.save(ckpt, path)

    loader             = JepaRunLoader.__new__(JepaRunLoader)
    _, axis, _         = RunLoader._load_checkpoint(loader, path, "cpu")

    assert axis.dtype == np.float32
    assert np.allclose(axis, x_axis.numpy())


class FakeProgress:
    def add_task(self, *args, **kwargs):
        return 0

    def advance(self, *args, **kwargs):
        pass


class FakeLogger:
    def __init__(self):
        import contextlib
        self._cm = contextlib.contextmanager(self._track)

    def _track(self, transient=False):
        yield FakeProgress()

    def track(self, transient=False):
        return self._cm(transient)

    def section(self, *args, **kwargs):
        pass

    def kv_table(self, *args, **kwargs):
        pass


def build_fake_run(model_fn, n_gaussians, n_elev, spatial):
    grid = GridInfo(
        n_v          = 1,
        n_h          = 1,
        pad_top      = 0,
        pad_bot      = 0,
        pad_left     = 0,
        pad_right    = 0,
        patch_size   = (spatial, spatial),
        stride       = spatial,
        spatial_size = (spatial, spatial),
    )

    gt     = torch.rand(1, n_gaussians * 3, spatial, spatial)
    loader = [(torch.rand(1, 2, spatial, spatial), gt)]
    x_axis = np.linspace(-4.0, 4.0, n_elev).astype(np.float32)

    class FakeNormalizer:
        def denormalize_output(self, tensor):
            return tensor

    dataset = types.SimpleNamespace(normalizer=FakeNormalizer())
    region  = types.SimpleNamespace(azimuth_start=0, range_start=0)

    return types.SimpleNamespace(
        n_gaussians   = n_gaussians,
        x_axis        = x_axis,
        x_axis_length = n_elev,
        loader        = loader,
        model         = model_fn,
        dataset       = dataset,
        grid          = grid,
        split_region  = region,
    )


def test_jepa_curve_predictor_produces_curve_cubes(tmp_path):
    n_gaussians = 2
    n_elev      = 6
    spatial     = 4

    def model_fn(images):
        batch = images.shape[0]
        return np.random.rand(batch, n_elev, spatial, spatial).astype(np.float32)

    run  = build_fake_run(model_fn, n_gaussians, n_elev, spatial)
    meta = types.SimpleNamespace(cube_dir=tmp_path)

    predictor = JepaCurvePredictor(
        run,
        FakeLogger(),
        window_kind = "uniform",
        cube_dtype  = "float32",
        save_cubes  = False,
        meta        = meta,
    )

    result = predictor.run_inference()

    assert result.pred_curves.shape == (n_elev, spatial, spatial)
    assert result.gt_curves.shape   == (n_elev, spatial, spatial)
    assert result.pixel_mse.shape   == (spatial, spatial)
    assert np.isfinite(result.pixel_mse).all()


def test_jepa_curve_predictor_zero_error_on_identical_output(tmp_path):
    n_gaussians = 2
    n_elev      = 6
    spatial     = 4

    x = np.linspace(-4.0, 4.0, n_elev).astype(np.float32).reshape(1, 1, -1, 1, 1)

    captured = {}

    def model_fn(images):
        from tools.data.gaussians import GaussianReconstructor
        gt_params = run["gt"][:, : n_gaussians * 3].cpu().numpy().astype(np.float32)
        batch     = gt_params.shape[0]
        gt_gauss  = gt_params.reshape(batch, n_gaussians, 3, spatial, spatial)
        curves    = GaussianReconstructor.reconstruct_batch(gt_gauss, x)
        captured["curves"] = curves
        return curves.astype(np.float32)

    gt     = torch.rand(1, n_gaussians * 3, spatial, spatial)
    loader = [(torch.rand(1, 2, spatial, spatial), gt)]
    run    = {"gt": gt}

    grid = GridInfo(
        n_v=1, n_h=1, pad_top=0, pad_bot=0, pad_left=0, pad_right=0,
        patch_size=(spatial, spatial), stride=spatial, spatial_size=(spatial, spatial),
    )

    class FakeNormalizer:
        def denormalize_output(self, tensor):
            return tensor

    fake_run = types.SimpleNamespace(
        n_gaussians   = n_gaussians,
        x_axis        = np.linspace(-4.0, 4.0, n_elev).astype(np.float32),
        x_axis_length = n_elev,
        loader        = loader,
        model         = model_fn,
        dataset       = types.SimpleNamespace(normalizer=FakeNormalizer()),
        grid          = grid,
        split_region  = types.SimpleNamespace(azimuth_start=0, range_start=0),
    )

    predictor = JepaCurvePredictor(
        fake_run,
        FakeLogger(),
        window_kind = "uniform",
        cube_dtype  = "float32",
        save_cubes  = False,
        meta        = types.SimpleNamespace(cube_dir=tmp_path),
    )

    result = predictor.run_inference()

    assert result.pixel_mse.max() == pytest.approx(0.0, abs=1e-8)
