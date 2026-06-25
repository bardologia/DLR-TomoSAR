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
                raise ValueError(f"Run '{run_directory}' is a standalone profile-autoencoder run, not a JEPA run; the backbone 'infer' has no spatial cube to reconstruct. Run 'main/infer_profile_autoencoder.py' to score its reconstructions.")
            return JEPA_INFERENCE_COMPONENTS

        if ImageAutoencoderConfigIO.exists(meta):
            if not has_backbone:
                raise ValueError(f"Run '{run_directory}' is a standalone image-autoencoder run, not a JEPA run; the backbone 'infer' cannot reconstruct it. Image autoencoders are evaluated only as a JEPA front-end.")
            return JEPA_PARAM_INFERENCE_COMPONENTS

        return InferenceComponents()
