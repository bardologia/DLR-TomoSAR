"""Inference & evaluation package for DLR-TomoSAR.

Submodules are intentionally not auto-imported to keep the package light and
to avoid pulling in ``training.trainer`` (which depends on optional packages
such as ``psutil``) when only the configuration dataclass is needed.

Typical usage::

    from inference.config   import InferenceConfig
    from inference.pipeline import InferencePipeline

    InferencePipeline(InferenceConfig(run_directory=...)).run()
"""
