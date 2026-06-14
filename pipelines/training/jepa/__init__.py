def __getattr__(name):
    if name == "Trainer":
        from pipelines.training.jepa.trainer import Trainer
        return Trainer
    if name == "JepaModule":
        from pipelines.training.jepa.trainer import JepaModule
        return JepaModule
    if name == "TrainingPipeline":
        from pipelines.training.jepa.pipeline import TrainingPipeline
        return TrainingPipeline
    if name == "SingleTrainRunner":
        from pipelines.training.jepa.pipeline import SingleTrainRunner
        return SingleTrainRunner
    raise AttributeError(f"module 'training.jepa' has no attribute '{name}'")


__all__ = [
    "Trainer",
    "JepaModule",
    "TrainingPipeline",
    "SingleTrainRunner",
]
