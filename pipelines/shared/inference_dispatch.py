from __future__ import annotations

from dataclasses import replace
from pathlib     import Path

from pipelines.shared.config_persistence import BackboneModelConfigIO, ImageAutoencoderConfigIO, ProfileAutoencoderConfigIO


class RunType:
    BACKBONE   = "backbone"
    PROFILE_AE = "profile_ae"
    IMAGE_AE   = "image_ae"


class InferenceDispatcher:
    def __init__(self, entry_config) -> None:
        self.entry = entry_config

    def classify(self, run_directory: Path) -> str:
        meta         = Path(run_directory) / "meta"
        has_backbone = BackboneModelConfigIO.exists(meta)
        has_profile  = ProfileAutoencoderConfigIO.exists(meta)
        has_image    = ImageAutoencoderConfigIO.exists(meta)

        if has_backbone:
            return RunType.BACKBONE
        if has_profile:
            return RunType.PROFILE_AE
        if has_image:
            return RunType.IMAGE_AE

        raise ValueError(f"Run '{run_directory}' has no recognized model config under meta/ (no backbone, profile-autoencoder, or image-autoencoder config); cannot infer.")

    def _run_backbone(self, run_directory: Path) -> None:
        from pipelines.backbone.inference.pipeline import InferencePipeline
        from pipelines.shared.inference_components import InferenceComponentsResolver

        config     = replace(self.entry.inference, run_directory=Path(run_directory), output_subdir=None)
        components = InferenceComponentsResolver.for_run(Path(run_directory))

        InferencePipeline(config, components=components).run()

    def _run_profile_ae(self, run_directory: Path) -> None:
        from pipelines.profile_autoencoder.inference.pipeline import ProfileAeInferencePipeline

        config = replace(self.entry.profile_inference, run_directory=Path(run_directory), output_subdir=None)

        ProfileAeInferencePipeline(config).run()

    def _run_image_ae(self, run_directory: Path) -> None:
        from pipelines.image_autoencoder.inference.pipeline import ImageAeInferencePipeline

        config = replace(self.entry.image_inference, run_directory=Path(run_directory), output_subdir=None)

        ImageAeInferencePipeline(config).run()

    def run(self, run_directory: Path) -> None:
        handlers = {
            RunType.BACKBONE   : self._run_backbone,
            RunType.PROFILE_AE : self._run_profile_ae,
            RunType.IMAGE_AE   : self._run_image_ae,
        }

        handlers[self.classify(run_directory)](run_directory)
