from __future__ import annotations

import re
from dataclasses import replace
from pathlib     import Path
from typing      import Callable

import numpy as np

from tools.baselines.containers import TrackBaselines


class TrackReader:
    MIN_ROWS = 4

    def __init__(self, reader: Callable | None = None):
        self._reader = reader

    def read(self, path: str | Path) -> np.ndarray:
        reader = self._reader if self._reader is not None else self._default_reader()
        data   = np.asarray(reader(str(path)))

        if data.ndim != 2 or data.shape[0] < self.MIN_ROWS:
            raise ValueError(f"Track file {path} has shape {data.shape}, expected (>= {self.MIN_ROWS}, n_azimuth)")

        return data

    @staticmethod
    def _default_reader() -> Callable:
        from STEtools.ste_io import rrat
        return rrat


class PassProductResolver:
    PRODUCT_SUBDIR    = Path(".")
    PRODUCT_PATTERNS  = ()
    TRACK_DIR_PATTERN = re.compile(r"[Tt]\d+\w*")

    def label(self, pass_directory: str | Path) -> str:
        node   = self._pass_node(pass_directory)
        flight = node.parent.name
        return f"{flight}_{node.name}" if flight else node.name

    def _pass_node(self, pass_directory: str | Path) -> Path:
        path = Path(pass_directory)
        return path.parent if self.TRACK_DIR_PATTERN.fullmatch(path.name) else path

    def resolve(self, pass_directory: str | Path) -> Path:
        directory = Path(pass_directory) / self.PRODUCT_SUBDIR

        for pattern in self.PRODUCT_PATTERNS:
            candidates = sorted(directory.glob(pattern))
            if candidates:
                return candidates[0]

        raise FileNotFoundError(f"No file matching {self.PRODUCT_PATTERNS} under {directory}")

    def resolve_passes(self, pass_directories: list) -> dict:
        mapping = {}

        for directory in pass_directories:
            label = self.label(directory)
            path  = self.resolve(directory)
            mapping[label] = path

        return mapping


class TrackFileResolver(PassProductResolver):
    PRODUCT_SUBDIR   = Path("INF") / "INF-TRACK"
    PRODUCT_PATTERNS = ("track_sar_resa_*.rat", "track_*.rat")


class BaselinesResolver:
    @staticmethod
    def baselines_file(dataset_dir: str | Path) -> Path:
        return Path(dataset_dir) / "meta" / TrackBaselines.FILENAME

    def resolved(self, geometry, dataset_dir: str | Path, secondary_labels=None):
        if geometry.baselines_source not in ("auto", "dataset", "manual"):
            raise ValueError(f"Unknown baselines_source '{geometry.baselines_source}', expected 'auto', 'dataset' or 'manual'")

        if geometry.baselines_source == "manual" or len(geometry.kz_values) > 0:
            return geometry

        path = self.baselines_file(dataset_dir)
        if not path.exists():
            if geometry.baselines_source == "dataset":
                raise FileNotFoundError(f"baselines_source='dataset' but {path} does not exist")
            return geometry

        table = TrackBaselines.load(path).subset(secondary_labels)
        return replace(geometry, baselines=table.baselines(geometry.baseline_component, look_angle_deg=geometry.look_angle_deg), baselines_origin=str(path))
