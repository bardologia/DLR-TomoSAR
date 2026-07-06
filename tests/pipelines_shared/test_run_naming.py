from __future__ import annotations

import pytest

from configuration.training              import LossConfig, LossCurriculumConfig
from configuration.training.general.loss import ParamMatching
from pipelines.shared.training.run_naming import RunNaming


def _loss(**flags) -> LossConfig:
    return LossConfig(**flags)


def test_loss_tag_lists_enabled_terms_param_first():
    loss = _loss(use_param_l1=True, use_cosine_curve=True, use_covariance_match=True)

    assert RunNaming.loss_tag(loss) == "param_l1-covariance_match-cosine_curve"


def test_loss_tag_rejects_empty_loss():
    with pytest.raises(ValueError):
        RunNaming.loss_tag(_loss())


def test_matching_tag_reads_the_enum_value():
    loss = _loss(use_param_l1=True)
    loss.param_matching = ParamMatching.SORTED_GT

    assert RunNaming.matching_tag(loss) == "sorted_gt"


def test_tag_orders_model_head_matching_loss():
    loss = _loss(use_param_l1=True)

    assert RunNaming.tag("resunet", "set_pred", loss) == "resunet_set_pred_hungarian_param_l1"


def test_training_tag_names_the_final_stage():
    warmup   = _loss(use_param_l1=True)
    complete = _loss(use_param_l1=True, use_covariance_match=True)

    enabled  = LossCurriculumConfig(enabled=True,  warmup=warmup, complete=complete)
    disabled = LossCurriculumConfig(enabled=False, warmup=warmup, complete=complete)

    assert RunNaming.training_tag("unet", "conv", enabled)  == "unet_conv_hungarian_param_l1-covariance_match"
    assert RunNaming.training_tag("unet", "conv", disabled) == "unet_conv_hungarian_param_l1-covariance_match"


def test_benchmark_unit_keeps_the_component_separator():
    loss = _loss(use_param_l1=True)

    assert RunNaming.benchmark_unit("resunet-multihead", "mse_curve", loss) == "resunet_multihead_hungarian__mse_curve"
    assert RunNaming.benchmark_unit("unet", "param_l1", loss)               == "unet_conv_hungarian__param_l1"


def test_benchmark_unit_without_component_uses_the_full_tag():
    loss = _loss(use_param_l1=True, use_cosine_curve=True)

    assert RunNaming.benchmark_unit("resunet-set_pred", None, loss) == "resunet_set_pred_hungarian_param_l1-cosine_curve"


def test_compose_appends_the_suffix():
    assert RunNaming.compose("unet_conv_hungarian_param_l1", "p-64")  == "unet_conv_hungarian_param_l1_p-64"
    assert RunNaming.compose("unet_conv_hungarian_param_l1", None)    == "unet_conv_hungarian_param_l1"
