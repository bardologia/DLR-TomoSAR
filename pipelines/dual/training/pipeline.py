from __future__ import annotations

from configuration.architectures import DualResUNetConfig
from models.dual import get_dual
from pipelines.backbone.training.pipeline       import TrainingPipeline
from pipelines.shared.config.config_persistence import DualModelConfigIO


class TrunkChannelMap:

    MAX_TRACKS = 64
    GROUPS     = ("pass", "ifg", "dem")

    @classmethod
    def resolve(cls, input_config, in_channels: int, groups: tuple) -> tuple[int, ...]:
        unknown = [group for group in groups if group not in cls.GROUPS]
        if unknown:
            raise ValueError(f"Unknown input groups {unknown}. Available: {list(cls.GROUPS)}")

        keys    = cls._channel_keys(input_config, in_channels)
        indices = tuple(index for index, key in enumerate(keys) if key.split("/")[0] in groups)

        if not indices:
            raise ValueError(f"Input groups {tuple(groups)} select no channels under the active input configuration")

        return indices

    @classmethod
    def _channel_keys(cls, input_config, in_channels: int) -> list[str]:
        counts = [n for n in range(cls.MAX_TRACKS + 1) if input_config.total_channels(n, n) == in_channels]
        if len(counts) != 1:
            raise ValueError(f"Could not infer a unique track count from in_channels={in_channels} under the active input configuration")

        return input_config.channel_group_keys(counts[0], counts[0])


class DualTrainingPipeline(TrainingPipeline):

    MODEL_CONFIG_IO = DualModelConfigIO

    def _model_overrides(self, in_channels: int, out_channels: int) -> dict:
        model_config = self.model_config if self.model_config is not None else DualResUNetConfig()
        input_config = self.dataset_config.input_config

        overrides                       = super()._model_overrides(in_channels, out_channels)
        overrides["params_channels"]    = TrunkChannelMap.resolve(input_config, in_channels, model_config.params_input)
        overrides["existence_channels"] = TrunkChannelMap.resolve(input_config, in_channels, model_config.existence_input)

        return overrides

    def _model_factory(self):
        return get_dual
