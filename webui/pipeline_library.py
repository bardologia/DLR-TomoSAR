from __future__ import annotations


class PipelineLibrary:

    def collect(self) -> list[dict]:
        return [
            {
                "key"    : "processing",
                "name"   : "Processing",
                "script" : "pre_process",
                "blurb"  : "Ingest raw F-SAR products via PyRat and produce the beamformed tomogram and interferograms.",
                "stages" : ["Tomogram construction", "Interferogram formation", "Artifact registry"],
            },
            {
                "key"    : "param",
                "name"   : "Parameter Extraction",
                "script" : "extract_params",
                "blurb"  : "Fit a K-Gaussian mixture per pixel to build the supervised parameter targets.",
                "stages" : ["Prominence init (CPU)", "Adam sigma fit (JAX GPU)", "Mu-sorted parameter array"],
            },
            {
                "key"    : "dataset",
                "name"   : "Dataset",
                "script" : None,
                "blurb"  : "Turn processed artifacts into PyTorch DataLoaders. Consumed directly by training.",
                "stages" : ["Crop and split", "Patch extraction", "Augmentation", "Normalization"],
            },
            {
                "key"    : "training",
                "name"   : "Training",
                "script" : "train",
                "blurb"  : "Supervised loop with AdamW, EMA, warmup, cosine annealing, and curriculum loss.",
                "stages" : ["Curve reconstruction", "Loss", "Optimiser step", "Eval and checkpoint"],
            },
            {
                "key"    : "inference",
                "name"   : "Inference",
                "script" : "infer",
                "blurb"  : "Sliding-window prediction, overlap-add stitching, metrics, and report generation.",
                "stages" : ["Windowed predict", "Cube stitch", "Metrics", "Report and figures"],
            },
            {
                "key"    : "tuning",
                "name"   : "Tuning",
                "script" : "tune",
                "blurb"  : "Optuna two-phase hyperparameter search distributed across GPUs.",
                "stages" : ["Phase 1 search", "Phase 2 search", "Best-trial export"],
            },
        ]
