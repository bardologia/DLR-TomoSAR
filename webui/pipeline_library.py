from __future__ import annotations


class PipelineLibrary:

    def collect(self) -> list[dict]:
        return [
            {
                "key"    : "processing",
                "name"   : "Processing (Tomogram + Interferograms)",
                "script" : "pre_process",
                "blurb"  : "Ingest raw F-SAR passes via PyRat into the Capon reference tomogram, the amplitude-weighted interferometric stack, and the per-pixel geometry field for the physics loss.",
                "stages" : ["Capon tomogram (PyRat)", "Interferogram formation", "Track baselines & geometry field", "Artifact registry & layout"],
            },
            {
                            "key"    : "param",
                            "name"   : "Parameter Extraction",
                            "script" : "extract_params",
                            "blurb"  : "Fit a K-Gaussian mixture per elevation profile to build the supervised parameter targets, with penalised model-order selection.",
                            "stages" : ["Profile conditioning", "Prominence init (CPU)", "Adam mixture fit (JAX GPU)", "Penalised best-K", "Mean-sorted target and diagnostics"],
                        },
            {
                "key"    : "dataset",
                "name"   : "Dataset (Loaders)",
                "script" : None,
                "blurb"  : "Turn processed artifacts into PyTorch DataLoaders, consumed directly by training.",
                "stages" : ["Crop and split", "Patch extraction", "Augmentation", "Normalization"],
            },
            {
                            "key"    : "training",
                            "name"   : "Training (Supervised backbone)",
                            "script" : "train_backbone",
                            "blurb"  : "Supervised backbone loop: one forward pass to per-pixel Gaussian parameters, a weight-normalised curve, parameter and optional physics loss, AdamW with warmup, cosine annealing and a two-phase loss curriculum, and best-epoch checkpointing.",
                            "stages" : ["Curve reconstruction", "Composite loss", "AdamW step", "Schedule and curriculum", "Eval and checkpoint"],
                        },
            {
                "key"    : "inference",
                "name"   : "Inference",
                "script" : "infer_backbone",
                "blurb"  : "Sliding-window prediction, overlap-add curve stitching and centrality-selected parameter stitching, then the full metric suite, reduced-Capon and interferometric-consistency baselines, and report generation.",
                "stages" : ["Windowed predict", "Cube stitch", "Metrics", "Reduced & consistency", "Report and figures"],
            },
            {
                "key"    : "tuning",
                "name"   : "Tuning",
                "script" : "tune",
                "blurb"  : "Optuna joint hyperparameter search: TPE density-ratio proposals with constant-liar parallelism and median pruning, distributed across GPUs and resumable in chunks.",
                "stages" : ["Search space", "Joint search (TPE)", "Trial and pruning", "Best-config export"],
            },
        ]
