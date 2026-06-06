from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar

import numpy as np


@dataclass
class TrackBaselines:
    labels         : list
    vertical       : list
    horizontal     : list
    vertical_std   : list
    horizontal_std : list
    track_files    : list = field(default_factory=list)
    azimuth_window : tuple | None = None

    FILENAME : ClassVar[str] = "baselines.json"

    @property
    def reference(self) -> str:
        return self.labels[0]

    @property
    def n_tracks(self) -> int:
        return len(self.labels)

    def baselines(self, component: str = "vertical", look_angle_deg: float | None = None) -> tuple:
        if component == "vertical":
            return tuple(self.vertical)
        if component == "horizontal":
            return tuple(self.horizontal)
        if component == "magnitude":
            return tuple(float(np.hypot(v, h)) for v, h in zip(self.vertical, self.horizontal))
        if component == "perpendicular":
            if look_angle_deg is None:
                raise ValueError("Baseline component 'perpendicular' requires look_angle_deg")
            theta = float(np.deg2rad(look_angle_deg))
            return tuple(float(h * np.cos(theta) + v * np.sin(theta)) for v, h in zip(self.vertical, self.horizontal))
        raise ValueError(f"Unknown baseline component '{component}', expected 'vertical', 'horizontal', 'magnitude' or 'perpendicular'")

    def describe(self) -> dict:
        window = "full track" if self.azimuth_window is None else f"[{self.azimuth_window[0]}, {self.azimuth_window[1]})"
        return {
            "Tracks"             : self.n_tracks,
            "Reference"          : self.reference,
            "Azimuth window"     : window,
            "Vertical [m]"       : ", ".join(f"{v:.2f}" for v in self.vertical),
            "Horizontal [m]"     : ", ".join(f"{h:.2f}" for h in self.horizontal),
            "Vertical std [m]"   : ", ".join(f"{s:.2f}" for s in self.vertical_std),
            "Horizontal std [m]" : ", ".join(f"{s:.2f}" for s in self.horizontal_std),
        }

    def to_payload(self) -> dict:
        return {
            "labels"         : list(self.labels),
            "reference"      : self.reference,
            "vertical"       : [float(v) for v in self.vertical],
            "horizontal"     : [float(h) for h in self.horizontal],
            "vertical_std"   : [float(s) for s in self.vertical_std],
            "horizontal_std" : [float(s) for s in self.horizontal_std],
            "track_files"    : [str(f) for f in self.track_files],
            "azimuth_window" : list(self.azimuth_window) if self.azimuth_window is not None else None,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "TrackBaselines":
        window = payload.get("azimuth_window")
        return cls(
            labels         = list(payload["labels"]),
            vertical       = [float(v) for v in payload["vertical"]],
            horizontal     = [float(h) for h in payload["horizontal"]],
            vertical_std   = [float(s) for s in payload.get("vertical_std", [])],
            horizontal_std = [float(s) for s in payload.get("horizontal_std", [])],
            track_files    = list(payload.get("track_files", [])),
            azimuth_window = tuple(window) if window is not None else None,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(), indent=4), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrackBaselines":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


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


class TrackFileResolver:
    TRACK_SUBDIR      = Path("INF") / "INF-TRACK"
    TRACK_PATTERNS    = ("track_sar_resa_*.rat", "track_*.rat")
    TRACK_DIR_PATTERN = re.compile(r"[Tt]\d+\w*")

    def resolve(self, pass_directory: str | Path) -> Path:
        directory = Path(pass_directory) / self.TRACK_SUBDIR

        for pattern in self.TRACK_PATTERNS:
            candidates = sorted(directory.glob(pattern))
            if candidates:
                return candidates[0]

        raise FileNotFoundError(f"No track file matching {self.TRACK_PATTERNS} under {directory}")

    def label(self, pass_directory: str | Path) -> str:
        path = Path(pass_directory)
        if self.TRACK_DIR_PATTERN.fullmatch(path.name):
            return path.parent.name
        return path.name

    def resolve_passes(self, pass_directories: list) -> dict:
        return {self.label(directory): self.resolve(directory) for directory in pass_directories}


class BaselineValidator:
    PRODUCT_PATTERN = re.compile(r"^track_[a-z]+(?:_[a-z]+)*")

    def __init__(self, std_threshold: float = 5.0):
        self.std_threshold = std_threshold

    def _product(self, filename: str) -> str | None:
        match = self.PRODUCT_PATTERN.match(filename)
        return match.group(0) if match is not None else None

    def _check_products(self, table: TrackBaselines) -> None:
        products = {self._product(Path(f).name) for f in table.track_files}
        products.discard(None)

        if len(products) > 1:
            raise ValueError(f"Mixed track file products {sorted(products)}; all tracks must come from the same product, prefer track_sar_resa")

    def _check_stds(self, table: TrackBaselines) -> None:
        for label, v_std, h_std in zip(table.labels, table.vertical_std, table.horizontal_std):
            worst = max(float(v_std), float(h_std))
            if worst > self.std_threshold:
                raise ValueError(f"Track {label} position std {worst:.1f} m exceeds threshold {self.std_threshold:.1f} m; the track file is likely a different product or coordinate frame")

    def validate(self, table: TrackBaselines) -> None:
        self._check_products(table)
        self._check_stds(table)


class BaselineExtractor:
    HORIZONTAL_ROW = 2
    VERTICAL_ROW   = 3

    def __init__(self, track_paths: dict, azimuth_window: tuple | None = None, reader: Callable | None = None, validator: BaselineValidator | None = None):
        self.track_paths    = {label: Path(path) for label, path in track_paths.items()}
        self.azimuth_window = azimuth_window
        self.reader         = TrackReader(reader)
        self.validator      = validator if validator is not None else BaselineValidator()

    @classmethod
    def from_pass_directories(cls, pass_directories: list, azimuth_window: tuple | None = None, reader: Callable | None = None, validator: BaselineValidator | None = None) -> "BaselineExtractor":
        track_paths = TrackFileResolver().resolve_passes(pass_directories)
        return cls(track_paths, azimuth_window=azimuth_window, reader=reader, validator=validator)

    def _window_slice(self, n_samples: int) -> slice:
        if self.azimuth_window is None:
            return slice(0, n_samples)

        start, end = int(self.azimuth_window[0]), int(self.azimuth_window[1])
        if start >= n_samples:
            raise ValueError(f"Azimuth window start {start} exceeds track length {n_samples}")

        return slice(max(0, start), min(n_samples, end))

    def _track_statistics(self, raw: np.ndarray) -> tuple:
        window     = self._window_slice(raw.shape[1])
        horizontal = raw[self.HORIZONTAL_ROW, window]
        vertical   = raw[self.VERTICAL_ROW,   window]

        return (
            float(np.nanmean(horizontal)), float(np.nanstd(horizontal)),
            float(np.nanmean(vertical)),   float(np.nanstd(vertical)),
        )

    def extract(self) -> TrackBaselines:
        labels     = list(self.track_paths.keys())
        files      = [self.track_paths[label] for label in labels]
        statistics = [self._track_statistics(self.reader.read(path)) for path in files]

        h_mean = [s[0] for s in statistics]
        h_std  = [s[1] for s in statistics]
        v_mean = [s[2] for s in statistics]
        v_std  = [s[3] for s in statistics]

        reference_h = h_mean[0]
        reference_v = v_mean[0]

        table = TrackBaselines(
            labels         = labels,
            vertical       = [v - reference_v for v in v_mean],
            horizontal     = [h - reference_h for h in h_mean],
            vertical_std   = v_std,
            horizontal_std = h_std,
            track_files    = [str(f) for f in files],
            azimuth_window = tuple(self.azimuth_window) if self.azimuth_window is not None else None,
        )

        self.validator.validate(table)

        return table
