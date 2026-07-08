from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuration.training.general.run import TrainingPathsConfig
from pipelines.patch_sweep.planner import SecondarySpread
from tools.monitoring.logger import Logger
from tools.sar.geometry_field import GeometryField

DEFAULT_FIELD  = REPO_ROOT / "test_data/meta/geometry_field.npz"
BOXCAR_SIDE    = 20.0
SWEEP_COUNTS   = (5, 9, 29)
HARD_THRESHOLD = 4.0


class KzApertureMeasurement:
    def __init__(self, field_path: Path = DEFAULT_FIELD):
        self.logger   = Logger(log_dir="logs", name="kz_aperture")
        self.field    = GeometryField.load(field_path)
        self.track_kz = dict(zip(self.field.labels, self.field.kz("height").mean(axis=(1, 2))))

    def subsets(self) -> dict:
        primary    = self.field.reference
        candidates = sorted((label for label in self.field.labels if label != primary), key=self.track_kz.get)
        selections = {"production": [primary, *TrainingPathsConfig().secondary_labels]}

        for count in SWEEP_COUNTS:
            selections[f"sweep even-spread n={count}"] = [primary, *SecondarySpread.even(candidates, count - 1)]

        selections["clustered 4 shortest"] = [primary, *candidates[:4]]

        return selections

    def report(self, name: str, labels: list) -> None:
        full   = self.moment(self.field.labels)
        subset = self.moment(labels)
        ratio  = full / subset
        naive  = BOXCAR_SIDE * np.sqrt(len(self.field.labels) / len(labels))
        floor  = BOXCAR_SIDE * np.sqrt(ratio)
        kappa  = np.sqrt(subset / len(labels))
        regime = "hard" if ratio >= HARD_THRESHOLD else "soft"

        self.logger.info(f"{name:24s} n={len(labels):2d}  S={subset:7.3f}  kappa={kappa:.3f}  dz={self.resolution(labels):5.1f} m  track-count s*={naive:5.1f}  kz-aware s*={floor:6.1f}  regime={regime}")

    def moment(self, labels: list) -> float:
        values = np.array([self.track_kz[label] for label in labels])
        return float(((values - values.mean()) ** 2).sum())

    def resolution(self, labels: list) -> float:
        values = np.array([self.track_kz[label] for label in labels])
        return float(2.0 * np.pi / (values.max() - values.min()))

    def run(self) -> None:
        labels = self.field.labels
        kappa  = np.sqrt(self.moment(labels) / len(labels))

        self.logger.info(f"{'full stack':24s} n={len(labels):2d}  S={self.moment(labels):7.3f}  kappa={kappa:.3f}  dz={self.resolution(labels):5.1f} m")

        for name, selection in self.subsets().items():
            self.report(name, selection)


if __name__ == "__main__":
    KzApertureMeasurement().run()
