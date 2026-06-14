def __getattr__(name):
    if name == "Trainer":
        from pipelines.training.backbone.trainer import Trainer
        return Trainer
    if name == "TrainingPipeline":
        from pipelines.training.backbone.pipeline import TrainingPipeline
        return TrainingPipeline
    if name == "TrainingRunMetadata":
        from pipelines.shared.run_metadata import TrainingRunMetadata
        return TrainingRunMetadata
    if name == "SingleTrainRunner":
        from pipelines.training.backbone.pipeline import SingleTrainRunner
        return SingleTrainRunner
    raise AttributeError(f"module 'training.backbone' has no attribute '{name}'")


__all__ = [
    "Trainer",
    "TrainingPipeline",
    "TrainingRunMetadata",
    "SingleTrainRunner",
]
