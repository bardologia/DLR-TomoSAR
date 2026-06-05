from __future__ import annotations

import gc
from dataclasses import dataclass, field

from configuration.benchmark_config import BenchmarkConfig
from models import CONFIG_REGISTRY, get_model
from pipelines.benchmark_pipeline.width_scaler import WidthScaler
from tools.logger import Logger

_IMAGE_SIZE_MODELS = {"swin_unet", "transunet", "unetr"}


@dataclass
class SizeMatchResult:
    model         : str
    scale         : float
    overrides     : dict
    parameters    : int
    target        : int
    deviation_pct : float
    iterations    : int
    history       : list[dict] = field(default_factory=list)


class SizeMatcher:
    def __init__(self, config: BenchmarkConfig, logger: Logger) -> None:
        self.config       = config
        self.logger       = logger
        self.scaler       = WidthScaler()
        self.image_size   = config.training.patch_size[0]
        self.in_channels  = config.size_match.in_channels
        self.out_channels = config.n_gaussians * 3

    def reference_count(self) -> int:
        reference = self.config.size_match.reference_model
        return self._count(reference, CONFIG_REGISTRY[reference]())

    def match(self, model_name: str, target: int) -> SizeMatchResult:
        size_match = self.config.size_match

        low  = size_match.scale_low
        high = size_match.scale_high
        while self._count_at(model_name, high) < target and high < 64.0:
            high *= 2.0

        best    : tuple | None = None
        history : list[dict]   = []

        for iteration in range(1, size_match.max_iterations + 1):
            scale      = (low * high) ** 0.5
            parameters = self._count_at(model_name, scale)
            deviation  = (parameters - target) / max(target, 1)

            history.append({"iteration": iteration, "scale": scale, "parameters": parameters, "deviation_pct": 100.0 * deviation})

            if best is None or abs(deviation) < abs(best[1]):
                best = (scale, deviation, parameters)

            if abs(deviation) <= size_match.tolerance:
                break

            if parameters < target:
                low = scale
            else:
                high = scale

        scale, deviation, parameters = best

        return SizeMatchResult(
            model         = model_name,
            scale         = scale,
            overrides     = self.scaler.overrides(model_name, scale),
            parameters    = parameters,
            target        = target,
            deviation_pct = 100.0 * deviation,
            iterations    = len(history),
            history       = history,
        )

    def _count_at(self, model_name: str, scale: float) -> int:
        return self._count(model_name, self.scaler.scaled_config(model_name, scale))

    def _count(self, model_name: str, model_config) -> int:
        overrides = {"in_channels": self.in_channels, "out_channels": self.out_channels}
        if model_name in _IMAGE_SIZE_MODELS:
            overrides["image_size"] = self.image_size

        model, _   = get_model(model_name, config=model_config, **overrides)
        parameters = sum(p.numel() for p in model.parameters())

        del model
        gc.collect()

        return parameters
