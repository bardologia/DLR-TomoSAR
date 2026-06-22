from __future__ import annotations

import dataclasses

import pytest

from configuration.training.general.optimization import (
    OptimizerConfig,
    SchedulerConfig,
    WarmupConfig,
    EarlyStoppingConfig,
    GradientClipperConfig,
)
from configuration.training.general.runtime import (
    IOConfig,
    OverfitConfig,
    TrainingLoopConfig,
    MemoryConfig,
    ResourceConfig,
)
from configuration.training.general.loss import (
    LossConfig,
    LossCurriculumConfig,
)
from configuration.training.general.trainer import SharedSubConfigInheritance
from configuration.training.general.entry   import TrainEntryConfig

from configuration.training.backbone import (
    PatchTrialsConfig,
    SecondaryTrialsConfig,
    BackboneTrainerConfig,
    BackboneEntryConfig,
)
from configuration.training.jepa import (
    EmbeddingLossConfig,
    JepaTrainerConfig,
    JepaDefaults,
    JepaEntryConfig,
)
from configuration.training.image_autoencoder import (
    ImageAeLossConfig,
    ImageAeTrainerConfig,
    ImageAeEntryConfig,
)
from configuration.training.profile_autoencoder import (
    ProfileAeLossConfig,
    ProfileAeTrainerConfig,
    ProfileAeEntryConfig,
)
from configuration.inference.general import InferenceConfig

from tests.configuration._helpers import make_gaussian


SIMPLE_DEFAULT_CONFIGS = [
    OptimizerConfig,
    SchedulerConfig,
    WarmupConfig,
    EarlyStoppingConfig,
    GradientClipperConfig,
    IOConfig,
    OverfitConfig,
    TrainingLoopConfig,
    MemoryConfig,
    ResourceConfig,
    LossConfig,
    LossCurriculumConfig,
    EmbeddingLossConfig,
    ImageAeLossConfig,
    ProfileAeLossConfig,
    PatchTrialsConfig,
    SecondaryTrialsConfig,
    TrainEntryConfig,
    BackboneEntryConfig,
    JepaEntryConfig,
    ImageAeEntryConfig,
    ProfileAeEntryConfig,
]

SIMPLE_IDS = [c.__name__ for c in SIMPLE_DEFAULT_CONFIGS]

TRAINER_CONFIGS = [
    BackboneTrainerConfig,
    JepaTrainerConfig,
    ImageAeTrainerConfig,
    ProfileAeTrainerConfig,
]


@pytest.mark.parametrize("config_cls", SIMPLE_DEFAULT_CONFIGS, ids=SIMPLE_IDS)
def test_default_config_instantiates(config_cls):
    instance = config_cls()
    assert dataclasses.is_dataclass(instance)


@pytest.mark.parametrize("config_cls", TRAINER_CONFIGS, ids=[c.__name__ for c in TRAINER_CONFIGS])
def test_trainer_config_requires_gaussian(config_cls):
    with pytest.raises(TypeError):
        config_cls()

    instance = config_cls(gaussian=make_gaussian())
    assert dataclasses.is_dataclass(instance)


def test_optimizer_config_values_sane():
    cfg = OptimizerConfig()
    assert cfg.lr > 0
    assert cfg.eps > 0
    assert len(cfg.betas) == 2
    assert all(0 < b < 1 for b in cfg.betas)


def test_scheduler_config_positive_epochs():
    cfg = SchedulerConfig()
    assert cfg.epochs > 0
    assert cfg.eta_min >= 0


def test_warmup_config_factor_in_range():
    cfg = WarmupConfig()
    assert cfg.warmup_steps >= 0
    assert 0.0 <= cfg.warmup_start_factor <= 1.0


def test_early_stopping_patience_positive():
    cfg = EarlyStoppingConfig()
    assert cfg.patience > 0


def test_loss_config_weight_used_directly():
    cfg = LossConfig(weight_mse_curve=2.0, weight_param_l1=3.0)
    assert cfg.weight_mse_curve == pytest.approx(2.0)
    assert cfg.weight_param_l1  == pytest.approx(3.0)


def test_loss_config_param_weights_tuple():
    cfg = LossConfig()
    assert isinstance(cfg.param_weights, tuple)
    assert len(cfg.param_weights) == 3


def test_loss_curriculum_holds_two_loss_configs():
    cfg = LossCurriculumConfig()
    assert isinstance(cfg.warmup, LossConfig)
    assert isinstance(cfg.complete, LossConfig)
    assert cfg.warmup is not cfg.complete


def test_train_entry_config_holds_all_modes():
    cfg = TrainEntryConfig()
    assert isinstance(cfg.backbone, BackboneEntryConfig)
    assert isinstance(cfg.jepa, JepaEntryConfig)
    assert isinstance(cfg.profile_autoencoder, ProfileAeEntryConfig)
    assert isinstance(cfg.image_autoencoder, ImageAeEntryConfig)


def test_backbone_entry_default_subconfigs():
    cfg = BackboneEntryConfig()
    assert cfg.backbone_name == "resunet"
    assert cfg.n_gaussians > 0
    assert isinstance(cfg.curriculum, LossCurriculumConfig)
    assert isinstance(cfg.inference, InferenceConfig)
    assert isinstance(cfg.warmup_losses, dict)
    assert isinstance(cfg.complete_losses, dict)


def test_backbone_entry_default_complete_losses_nonempty():
    cfg = BackboneEntryConfig()
    assert len(cfg.complete_losses) > 0
    for spec in cfg.complete_losses.values():
        assert spec["use_param_l1"] is True


def test_backbone_trainer_subconfig_factories():
    cfg = BackboneTrainerConfig(gaussian=make_gaussian())
    assert isinstance(cfg.optimizer, OptimizerConfig)
    assert isinstance(cfg.scheduler, SchedulerConfig)
    assert isinstance(cfg.gradient_clipper, GradientClipperConfig)


def test_jepa_defaults_inference_is_inference_config():
    inf = JepaDefaults.inference()
    assert isinstance(inf, InferenceConfig)
    assert inf.save_cubes is True


def test_jepa_trainer_param_loss_default():
    cfg = JepaTrainerConfig(gaussian=make_gaussian())
    assert cfg.param_loss.use_param_l1 is True
    assert isinstance(cfg.embedding_loss, EmbeddingLossConfig)


def test_shared_subconfig_inheritance_copies_fields():
    class Base:
        pass

    base = Base()
    for name in ("geometry", "early_stopping", "warmup", "scheduler", "io",
                 "optimizer", "training", "resources", "gradient_clipper"):
        setattr(base, name, object())

    target = JepaTrainerConfig(gaussian=make_gaussian())
    target.inherit_shared_from(base)
    assert target.optimizer is base.optimizer
    assert target.scheduler is base.scheduler


def test_shared_subconfig_class_is_mixin():
    assert issubclass(JepaTrainerConfig, SharedSubConfigInheritance)
    assert issubclass(ImageAeTrainerConfig, SharedSubConfigInheritance)
    assert issubclass(ProfileAeTrainerConfig, SharedSubConfigInheritance)


def test_image_ae_entry_training_overrides():
    cfg = ImageAeEntryConfig()
    assert cfg.training.batch_size == 512
    assert cfg.ae_model_name == "conv2d_ae"


def test_profile_ae_entry_training_overrides():
    cfg = ProfileAeEntryConfig()
    assert cfg.training.batch_size == 1024
    assert cfg.ae_model_name == "mlp_ae"
