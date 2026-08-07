from __future__ import annotations

from types import SimpleNamespace

import pytest

from configuration.architectures                import Conv2dImageAutoencoderConfig, GruAutoencoderConfig
from configuration.training                     import JepaEntryConfig
from pipelines.jepa.training                    import pipeline as jepa_pipeline
from pipelines.jepa.training.pipeline           import SingleTrainRunner
from pipelines.shared.config.config_persistence import ImageAutoencoderConfigIO, ProfileAutoencoderConfigIO
from pipelines.shared.training.run_naming       import JepaNamingSpec


def _entry(tmp_path, **fields) -> JepaEntryConfig:
    config = JepaEntryConfig()

    config.profile_autoencoder_logdir = tmp_path / "profile_runs"
    config.image_autoencoder_logdir   = tmp_path / "image_runs"

    for name, value in fields.items():
        setattr(config, name, value)

    return config


def _save_profile_ae(tmp_path) -> None:
    ProfileAutoencoderConfigIO.save(GruAutoencoderConfig(), "gru_ae", tmp_path / "profile_runs" / "gru_run" / "meta")


def _save_image_ae(tmp_path) -> None:
    ImageAutoencoderConfigIO.save(Conv2dImageAutoencoderConfig(), "conv2d_ae", tmp_path / "image_runs" / "conv_run" / "meta")


def _runner(monkeypatch, config: JepaEntryConfig) -> SingleTrainRunner:
    monkeypatch.setattr(jepa_pipeline, "ConfigFactory", lambda entry: SimpleNamespace(gaussian_config=lambda: SimpleNamespace(n_default_gaussians=5)))
    return SingleTrainRunner(config)


def test_label_profile_coupled_carries_target_ae_mode_and_embedding_loss(monkeypatch, tmp_path):
    _save_profile_ae(tmp_path)
    runner = _runner(monkeypatch, _entry(tmp_path, profile_autoencoder_run="gru_run"))

    assert runner.label == "resunet-conv-stopgrad-K_5-hv-none-pae_gru.frozen-emb_mse_1-curve_mse_1"


def test_label_image_only_carries_the_param_loss_and_the_image_ae(monkeypatch, tmp_path):
    _save_image_ae(tmp_path)
    runner = _runner(monkeypatch, _entry(tmp_path, image_autoencoder_run="conv_run"))

    assert runner.label == "resunet-conv-sorted_gt-K_5-hv-A-iae_conv2d.frozen-param_l1_1"


def test_label_names_both_autoencoders_with_their_modes(monkeypatch, tmp_path):
    _save_profile_ae(tmp_path)
    _save_image_ae(tmp_path)
    fields = dict(profile_autoencoder_run="gru_run", profile_autoencoder_mode="finetune", target_provider="live", image_autoencoder_run="conv_run", image_autoencoder_mode="finetune")
    runner = _runner(monkeypatch, _entry(tmp_path, **fields))

    assert runner.label == "resunet-conv-live-K_5-hv-none-pae_gru.finetune-iae_conv2d.finetune-emb_mse_1-curve_mse_1"


def test_resolve_rejects_a_config_without_any_autoencoder(tmp_path):
    with pytest.raises(ValueError):
        JepaNamingSpec.resolve(_entry(tmp_path))


def test_resolve_fails_loudly_when_the_ae_run_lacks_its_config(tmp_path):
    (tmp_path / "profile_runs" / "gru_run" / "meta").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        JepaNamingSpec.resolve(_entry(tmp_path, profile_autoencoder_run="gru_run"))
