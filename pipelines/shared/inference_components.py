from __future__ import annotations

from pathlib import Path

from pipelines.backbone.inference.pipeline import InferenceComponents
from pipelines.jepa.inference.pipeline     import JEPA_INFERENCE_COMPONENTS, JEPA_PARAM_INFERENCE_COMPONENTS
from pipelines.shared.config_persistence   import BackboneModelConfigIO, ImageAutoencoderConfigIO, ProfileAutoencoderConfigIO


class InferenceComponentsResolver:
    @staticmethod
    def for_run(run_directory: Path) -> InferenceComponents:
        meta         = Path(run_directory) / "meta"
        has_backbone = BackboneModelConfigIO.exists(meta)

        if ProfileAutoencoderConfigIO.exists(meta):
            if not has_backbone:
                raise ValueError(f"Run '{run_directory}' is a standalone profile-autoencoder run, not a JEPA run; this resolver only selects backbone/JEPA spatial components. The unified 'main/infer.py' detects and scores standalone autoencoder runs on its own.")
            return JEPA_INFERENCE_COMPONENTS

        if ImageAutoencoderConfigIO.exists(meta):
            if not has_backbone:
                raise ValueError(f"Run '{run_directory}' is a standalone image-autoencoder run, not a JEPA run; this resolver only selects backbone/JEPA spatial components. The unified 'main/infer.py' detects and scores standalone autoencoder runs on its own.")
            return JEPA_PARAM_INFERENCE_COMPONENTS

        return InferenceComponents()
