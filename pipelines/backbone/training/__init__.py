def __getattr__(name):
    if name == "Trainer":
        from pipelines.backbone.training.trainer import Trainer
        return Trainer
    if name == "TrainingPipeline":
        from pipelines.backbone.training.pipeline import TrainingPipeline
        return TrainingPipeline
    if name == "TrainingRunMetadata":
        from pipelines.shared.run_metadata import TrainingRunMetadata
        return TrainingRunMetadata
    if name == "SingleTrainRunner":
        from pipelines.backbone.training.pipeline import SingleTrainRunner
        return SingleTrainRunner
    raise AttributeError(f"module 'backbone.training' has no attribute '{name}'")


__all__ = [
    "Trainer",
    "TrainingPipeline",
    "TrainingRunMetadata",
    "SingleTrainRunner",
]
