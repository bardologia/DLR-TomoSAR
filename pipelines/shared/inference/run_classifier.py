from __future__ import annotations

from pathlib import Path


class RunType:
    BACKBONE   = "backbone"
    PROFILE_AE = "profile_ae"
    IMAGE_AE   = "image_ae"
    UNROLLED   = "unrolled"


class RunArtifacts:
    BACKBONE_CONFIG   = "model_config.json"
    PROFILE_AE_CONFIG = "profile_autoencoder_config.json"
    IMAGE_AE_CONFIG   = "image_autoencoder_config.json"
    UNROLLED_CONFIG   = "unrolled_model_config.json"


class RunClassifier:

    TYPE_ARTIFACTS = (
        (RunType.BACKBONE,   RunArtifacts.BACKBONE_CONFIG),
        (RunType.PROFILE_AE, RunArtifacts.PROFILE_AE_CONFIG),
        (RunType.IMAGE_AE,   RunArtifacts.IMAGE_AE_CONFIG),
        (RunType.UNROLLED,   RunArtifacts.UNROLLED_CONFIG),
    )

    @classmethod
    def classify(cls, run_directory: Path) -> str:
        meta = Path(run_directory) / "meta"

        for run_type, filename in cls.TYPE_ARTIFACTS:
            if (meta / filename).is_file():
                return run_type

        raise ValueError(f"Run '{run_directory}' has no recognized model config under meta/ (no backbone, profile-autoencoder, image-autoencoder, or unrolled config); cannot infer.")

    @classmethod
    def is_type(cls, run_directory: Path, run_type: str) -> bool:
        try:
            return cls.classify(run_directory) == run_type
        except ValueError:
            return False
