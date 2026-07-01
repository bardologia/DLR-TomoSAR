from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.data.io import FileIO


class AeMetricsBase:
    EPS = 1e-8

    def __init__(self, result, normalizer) -> None:
        self.gt         = result.gt.astype(np.float64)
        self.pred       = result.pred.astype(np.float64)
        self.emb        = result.embeddings.astype(np.float64)
        self.normalizer = normalizer

    def _embedding_stats(self) -> dict:
        dim_std = np.std(self.emb, axis=0)

        return {
            "embedding_norm_mean"           : float(np.mean(np.linalg.norm(self.emb, axis=1))),
            "embedding_dim_std_mean"        : float(np.mean(dim_std)),
            "embedding_active_dim_fraction" : float(np.mean(dim_std > 1e-4)),
        }

    @staticmethod
    def write_json(metrics: dict, path: Path) -> Path:
        return FileIO.save_json(metrics, Path(path), indent=4)
