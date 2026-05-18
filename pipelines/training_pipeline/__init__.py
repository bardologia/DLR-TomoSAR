from tools                                    import Logger, ModelSummary, ShapeLogger, Tracker
from pipelines.training_pipeline.metadata import TrainingRunMetadata


def __getattr__(name):
    if name == "Trainer":
        from pipelines.training_pipeline.trainer import Trainer
        return Trainer
    if name == "TrainingPipeline":
        from pipelines.training_pipeline.pipeline import TrainingPipeline
        return TrainingPipeline
    raise AttributeError(f"module 'training_pipeline' has no attribute '{name}'")


__all__ = [
    "Logger",
    "ModelSummary",
    "ShapeLogger",
    "Tracker",
    "Trainer",
    "TrainingPipeline",
    "TrainingRunMetadata",
]
