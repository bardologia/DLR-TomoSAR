from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.logger import Logger


class ParameterIO:
    def __init__(self, logger : Logger) -> None:
        self.logger = logger

    def save_params(self, parameters_array : np.ndarray, npy_path : Path) -> Path:
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.subsection(f"Saving parameter stack of shape {parameters_array.shape} to disk")
        np.save(str(npy_path), np.ascontiguousarray(parameters_array), allow_pickle=False)

        return npy_path

    def load_params(self, npy_path : Path) -> np.ndarray:
        self.logger.subsection("Loading saved parameters for metrics and plots")
        return np.load(str(npy_path)).astype(np.float32, copy=False)

    def load_metadata(self, meta_path : Path) -> dict:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
