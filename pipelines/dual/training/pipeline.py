from __future__ import annotations

from models.dual import get_dual
from pipelines.backbone.training.pipeline       import TrainingPipeline
from pipelines.shared.config.config_persistence import DualModelConfigIO


class IfgChannelMap:

    MAX_TRACKS = 64

    @classmethod
    def resolve(cls, input_config, in_channels: int) -> tuple[int, ...]:
        if not input_config.use_interferograms:
            raise ValueError("The dual family requires use_interferograms=True; without interferogram channels the existence trunk has no input")

        counts = [n for n in range(cls.MAX_TRACKS + 1) if input_config.total_channels(n, n) == in_channels]
        if len(counts) != 1:
            raise ValueError(f"Could not infer a unique track count from in_channels={in_channels} under the active input configuration")

        keys = input_config.channel_group_keys(counts[0], counts[0])

        return tuple(index for index, key in enumerate(keys) if key.startswith("ifg/"))


class DualTrainingPipeline(TrainingPipeline):

    MODEL_CONFIG_IO = DualModelConfigIO

    def _model_overrides(self, in_channels: int, out_channels: int) -> dict:
        overrides                 = super()._model_overrides(in_channels, out_channels)
        overrides["ifg_channels"] = IfgChannelMap.resolve(self.dataset_config.input_config, in_channels)

        return overrides

    def _model_factory(self):
        return get_dual
