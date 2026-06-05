def __getattr__(name):
    if name == "Trainer":
        from pipelines.training_pipeline.trainer import Trainer
        return Trainer
    if name == "TrainingPipeline":
        from pipelines.training_pipeline.pipeline import TrainingPipeline
        return TrainingPipeline
    if name == "TrainingRunMetadata":
        from pipelines.training_pipeline.pipeline import TrainingRunMetadata
        return TrainingRunMetadata
    if name == "SingleTrainRunner":
        from pipelines.training_pipeline.pipeline import SingleTrainRunner
        return SingleTrainRunner
    raise AttributeError(f"module 'training_pipeline' has no attribute '{name}'")


__all__ = [
    "Trainer",
    "TrainingPipeline",
    "TrainingRunMetadata",
    "SingleTrainRunner",
]
