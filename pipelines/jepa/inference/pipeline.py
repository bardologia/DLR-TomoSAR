from __future__ import annotations

from pipelines.backbone.inference.pipeline import InferenceComponents
from pipelines.jepa.inference.loader       import JepaRunLoader
from pipelines.jepa.inference.predictor    import JepaCurvePredictor

JEPA_INFERENCE_COMPONENTS = InferenceComponents(
    loader_cls    = JepaRunLoader,
    predictor_cls = JepaCurvePredictor,
    param_space   = False,
)
