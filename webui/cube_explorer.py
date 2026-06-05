from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import ProjectPaths
from web_logger import WebLogger


class CubeExplorer:

    SOURCES = ("pred", "gt")

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:
        self.paths     = paths
        self.logger    = logger
        self.logs_root = (paths.repo_root / "logs").resolve()
        self.cache     = {}
        self.lock      = threading.Lock()

    def list_cubes(self) -> list[dict]:
        cubes = []
        for cube_file in sorted(self.logs_root.rglob("inference/*/cubes/pred_curves.npy")):
            stamp_dir = cube_file.parent.parent
            run_dir   = stamp_dir.parent.parent
            shape     = self._shape(cube_file)
            if shape is None:
                continue

            cubes.append({
                "id"      : str(stamp_dir.relative_to(self.logs_root)),
                "run"     : run_dir.name,
                "group"   : str(run_dir.relative_to(self.logs_root).parent),
                "stamp"   : stamp_dir.name,
                "sources" : [s for s in self.SOURCES if (stamp_dir / "cubes" / f"{s}_curves.npy").is_file()],
                "n_elev"  : shape[0],
                "n_az"    : shape[1],
                "n_rg"    : shape[2],
            })

        cubes.sort(key=lambda c: c["id"], reverse=True)
        return cubes

    def topdown_png(self, cube_id: str, source: str) -> bytes | None:
        entry = self._entry(cube_id, source)
        if entry is None:
            return None

        mean       = entry["mean"]
        vmin, vmax = np.percentile(mean, [1.0, 99.0])

        buf = io.BytesIO()
        plt.imsave(buf, mean, cmap="jet", vmin=float(vmin), vmax=float(vmax), format="png")
        return buf.getvalue()

    def slice_png(self, cube_id: str, source: str, axis: str, az: int, rg: int) -> bytes | None:
        entry = self._entry(cube_id, source)
        if entry is None or axis not in ("range", "azimuth"):
            return None

        cube               = entry["cube"]
        n_elev, n_az, n_rg = cube.shape
        az                 = int(np.clip(az, 0, n_az - 1))
        rg                 = int(np.clip(rg, 0, n_rg - 1))

        if axis == "range":
            data = np.asarray(cube[:, :, rg], dtype=np.float32)
        else:
            data = np.asarray(cube[:, az, :], dtype=np.float32)

        sort_idx = np.argsort(entry["x_axis"])
        data     = np.flipud(data[sort_idx])

        buf = io.BytesIO()
        plt.imsave(buf, data, cmap="jet", vmin=entry["vmin"], vmax=entry["vmax"], format="png")
        return buf.getvalue()

    def _entry(self, cube_id: str, source: str) -> dict | None:
        if source not in self.SOURCES:
            return None

        stamp_dir = (self.logs_root / cube_id).resolve()
        if not str(stamp_dir).startswith(str(self.logs_root)):
            return None

        cube_path = stamp_dir / "cubes" / f"{source}_curves.npy"
        if not cube_path.is_file():
            return None

        key   = str(cube_path)
        mtime = cube_path.stat().st_mtime

        with self.lock:
            cached = self.cache.get(key)
            if cached is not None and cached["mtime"] == mtime:
                return cached

        cube = np.load(cube_path, mmap_mode="r")
        if cube.ndim != 3:
            return None

        mean    = np.zeros(cube.shape[1:], dtype=np.float32)
        samples = []
        step    = max(1, cube.shape[0] // 8)
        for i in range(cube.shape[0]):
            plane = np.asarray(cube[i], dtype=np.float32)
            mean += plane
            if i % step == 0:
                samples.append(plane[:: max(1, plane.shape[0] // 200), :: max(1, plane.shape[1] // 200)].ravel())
        mean /= cube.shape[0]

        sampled    = np.concatenate(samples)
        sampled    = sampled[np.isfinite(sampled)]
        vmin, vmax = (np.percentile(sampled, [1.0, 99.0]) if sampled.size else (0.0, 1.0))

        x_axis, has_axis = self._elevation_axis(stamp_dir, cube.shape[0])

        entry = {
            "cube"     : cube,
            "mean"     : mean,
            "x_axis"   : x_axis,
            "has_axis" : has_axis,
            "vmin"     : float(vmin),
            "vmax"     : float(vmax),
            "mtime"    : mtime,
        }
        with self.lock:
            self.cache[key] = entry

        self.logger.muted(f"cube loaded: {cube_path} shape={cube.shape}")
        return entry

    def _elevation_axis(self, stamp_dir: Path, n_elev: int) -> tuple[np.ndarray, bool]:
        metrics_path = stamp_dir / "metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            lo      = float(metrics["x_axis_min"])
            hi      = float(metrics["x_axis_max"])
            return np.linspace(lo, hi, n_elev), True
        except (OSError, ValueError, KeyError):
            return np.arange(n_elev, dtype=np.float64), False

    @staticmethod
    def _shape(cube_path: Path) -> tuple | None:
        try:
            shape = np.load(cube_path, mmap_mode="r").shape
        except (OSError, ValueError):
            return None
        return shape if len(shape) == 3 else None
