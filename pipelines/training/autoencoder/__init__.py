def __getattr__(name):
    if name == "Trainer":
        from pipelines.training.autoencoder.trainer import Trainer
        return Trainer
    if name == "TrainingPipeline":
        from pipelines.training.autoencoder.pipeline import TrainingPipeline
        return TrainingPipeline
    if name == "SingleTrainRunner":
        from pipelines.training.autoencoder.pipeline import SingleTrainRunner
        return SingleTrainRunner
    raise AttributeError(f"module 'training.autoencoder' has no attribute '{name}'")


__all__ = [
    "Trainer",
    "TrainingPipeline",
    "SingleTrainRunner",
]
