from __future__ import annotations

from configuration.training import LossConfig, LossCurriculumConfig
from pipelines.backbone.training.loss_terms import LOSS_TERMS
from pipelines.shared.model.model_builder   import ModelBuilder


class RunNaming:

    NAMING_ORDER = tuple(reversed(LOSS_TERMS))

    @classmethod
    def loss_tag(cls, loss: LossConfig) -> str:
        names = [term.name for term in cls.NAMING_ORDER if getattr(loss, term.use_flag)]
        if not names:
            raise ValueError("Cannot name a run from a loss config with no enabled loss terms")

        return "-".join(names)

    @staticmethod
    def matching_tag(loss: LossConfig) -> str:
        return loss.param_matching.value

    @classmethod
    def tag(cls, model_name: str, head: str, loss: LossConfig) -> str:
        return f"{model_name}_{head}_{cls.matching_tag(loss)}_{cls.loss_tag(loss)}"

    @classmethod
    def training_tag(cls, model_name: str, head: str, curriculum: LossCurriculumConfig) -> str:
        return cls.tag(model_name, head, curriculum.active_stages()[-1])

    @classmethod
    def benchmark_unit(cls, model_key: str, component: str | None, loss: LossConfig) -> str:
        name, head = ModelBuilder.split_key(model_key)
        if component is None:
            return cls.tag(name, head, loss)

        return f"{name}_{head}_{cls.matching_tag(loss)}__{component}"

    @staticmethod
    def compose(tag: str, suffix: str | None) -> str:
        return tag if not suffix else f"{tag}_{suffix}"
