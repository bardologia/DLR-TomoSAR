# DLR-TomoSAR — Pipeline Reports

Detailed walk-throughs of every pipeline shipped under [pipelines/](../pipelines).
Each report covers the configuration surface, the step-by-step internal flow, the equations involved, and the outputs / metadata produced.

| # | Pipeline                          | Report                                                          | What it does                                                                 |
|---|-----------------------------------|------------------------------------------------------------------|------------------------------------------------------------------------------|
| 1 | `pre_processing_pipeline`         | [pre_processing_pipeline.md](pre_processing_pipeline.md)         | Builds the full / input tomograms (PyRat) and the complex SAR pass stack.    |
| 2 | `param_extraction_pipeline`       | [param_extraction_pipeline.md](param_extraction_pipeline.md)     | Fits a multi-Gaussian elevation model to every pixel (Curve-Fit / MLE).      |
| 3 | `dataset_creation_pipeline`       | [dataset_creation_pipeline.md](dataset_creation_pipeline.md)     | Crops, patches, normalizes, and exposes train/val/test PyTorch DataLoaders.  |
| 4 | `training_pipeline`               | [training_pipeline.md](training_pipeline.md)                     | Trains a model with a 11-term composite loss, AMP, EMA, warmup, scheduler.   |
| 5 | `autoencoder_pipeline`            | [autoencoder_pipeline.md](autoencoder_pipeline.md)               | Self-supervised profile representation learning (VICReg + NT-Xent).          |
| 6 | `inference_pipeline`              | [inference_pipeline.md](inference_pipeline.md)                   | Patch-stitched inference, per-pixel metrics, plots, GIFs, Markdown report.   |

## Typical end-to-end run

```
pre_processing_pipeline   ─► data/{tomofull,inputs}.npy + meta/
param_extraction_pipeline ─► data/<prefix>_<suffix>.npy  (3K, A, R)
dataset_creation_pipeline ─► train/val/test DataLoaders + meta/normalization_stats.json
training_pipeline         ─► run_<model>_<ts>/best_model.pt + tensorboard + docs
inference_pipeline        ─► <run>/inference/<ts>/report.md + figures + cubes
autoencoder_pipeline      ─► run_autoencoder_<ts>/report.md (independent self-supervised path)
```

Each report uses workspace-relative links to the source files so you can jump from a description directly into the implementation.
