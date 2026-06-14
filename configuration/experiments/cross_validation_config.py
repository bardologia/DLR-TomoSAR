from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from configuration.experiments.benchmark_config import (
    BenchmarkPathsConfig,
    ComparisonReportConfig,
    InferenceQueueConfig,
    TrainingQueueConfig,
)
from configuration.training.autoencoder_config  import AutoencoderLossConfig
from configuration.sar.geometry_config     import GeometryConfig
from configuration.training.jepa_config         import EmbeddingLossConfig
from configuration.model.autoencoder_models_config import AutoencoderBaseConfig, MlpAutoencoderConfig
from configuration.training.runtime_config      import OverfitConfig


@dataclass
class FoldConfig:
    n_folds       : int = 10
    azimuth_start : int = 1000
    azimuth_end   : int = 16000


@dataclass
class JepaCvConfig:
    stage_a_run     : Path | None         = None
    stage_a_mode    : str                 = "frozen"
    target_provider : str                 = "stopgrad"
    embedding_loss  : EmbeddingLossConfig = field(default_factory=EmbeddingLossConfig)


@dataclass
class AeCvConfig:
    ae_model_name   : str                   = "mlp_ae"
    autoencoder     : AutoencoderBaseConfig = field(default_factory=MlpAutoencoderConfig)
    ae_loss         : AutoencoderLossConfig = field(default_factory=AutoencoderLossConfig)
    pixel_subsample : float                 = 1.0
    keep_empty_frac : float                 = 0.05


@dataclass
class CrossValidationConfig:
    training_type   : str  = "backbone"

    model_name      : str  = "resunet"
    model_overrides : dict = field(default_factory=dict)

    paths      : BenchmarkPathsConfig   = field(default_factory=lambda: BenchmarkPathsConfig(log_base_dir=Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/cross_validation")))
    folds      : FoldConfig             = field(default_factory=FoldConfig)
    training   : TrainingQueueConfig    = field(default_factory=TrainingQueueConfig)
    inference  : InferenceQueueConfig   = field(default_factory=InferenceQueueConfig)
    comparison : ComparisonReportConfig = field(default_factory=ComparisonReportConfig)

    geometry   : GeometryConfig         = field(default_factory=GeometryConfig)
    overfit    : OverfitConfig          = field(default_factory=OverfitConfig)
    jepa       : JepaCvConfig           = field(default_factory=JepaCvConfig)
    autoencoder: AeCvConfig             = field(default_factory=AeCvConfig)

    inference_splits : list[str] = field(default_factory=lambda: ["val", "test"])

    gpus            : list[int]  = field(default_factory=lambda: [2, 3])
    run_tag         : str | None = None
    resume          : bool       = True
    seed            : int        = 0
    n_gaussians     : int        = 5
    poll_interval_s : float      = 5.0

    def runs_inference(self) -> bool:
        return self.training_type != "autoencoder"
