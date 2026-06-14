def __getattr__(name):
    if name == "Trainer":
        from pipelines.backbone_pipeline.trainer import Trainer
        return Trainer
    if name == "TrainingPipeline":
        from pipelines.backbone_pipeline.pipeline import TrainingPipeline
        return TrainingPipeline
    if name == "TrainingRunMetadata":
        from pipelines.backbone_pipeline.pipeline import TrainingRunMetadata
        return TrainingRunMetadata
    if name == "SingleTrainRunner":
        from pipelines.backbone_pipeline.pipeline import SingleTrainRunner
        return SingleTrainRunner
    raise AttributeError(f"module 'backbone_pipeline' has no attribute '{name}'")


__all__ = [
    "Trainer",
    "TrainingPipeline",
    "TrainingRunMetadata",
    "SingleTrainRunner",
]
