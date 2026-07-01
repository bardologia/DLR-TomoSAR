from __future__ import annotations

from types import SimpleNamespace

import torch

from pipelines.autoencoder_common.inference.predictor import AeReconstructionPredictor, AeResult
from pipelines.image_autoencoder.inference.predictor import ImageAePredictor, ImageAeResult
from pipelines.profile_autoencoder.inference.predictor import ProfileAePredictor, ProfileAeResult


class _Ident:
    def denormalize_input(self, x):
        return x

    def denormalize(self, x):
        return x


class _ImageModel:
    def reconstruct(self, x):
        return x * 0.9, torch.zeros(x.shape[0], 8)


class _ProfileModel:
    def reconstruct(self, x):
        return x * 0.9, torch.zeros(x.shape[0], 8, 1, 1)


def _logger():
    return SimpleNamespace(section=lambda *a, **k: None, kv_table=lambda *a, **k: None)


def test_predictors_share_base():
    assert issubclass(ImageAePredictor, AeReconstructionPredictor)
    assert issubclass(ProfileAePredictor, AeReconstructionPredictor)
    assert issubclass(ImageAeResult, AeResult)
    assert issubclass(ProfileAeResult, AeResult)


def test_image_predictor_run_inference():
    x   = torch.rand(2, 1, 4, 4)
    run = SimpleNamespace(model=_ImageModel(), normalizer=_Ident(), loader=[(x,)])

    res = ImageAePredictor(run, "cpu", _logger()).run_inference()

    assert res.gt.shape == (2, 1, 4, 4)
    assert res.embeddings.shape == (2, 8)
    assert isinstance(res, ImageAeResult)


def test_profile_predictor_run_inference():
    x   = torch.rand(2, 20)
    run = SimpleNamespace(model=_ProfileModel(), normalizer=_Ident(), loader=[x])

    res = ProfileAePredictor(run, "cpu", _logger()).run_inference()

    assert res.gt.shape == (2, 20)
    assert res.embeddings.shape == (2, 8)
    assert isinstance(res, ProfileAeResult)
