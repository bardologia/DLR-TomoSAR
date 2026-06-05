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

    PLOT_RC = {
        "font.family"     : "serif",
        "font.serif"      : ["Times New Roman", "DejaVu Serif"],
        "font.size"       : 11,
        "axes.linewidth"  : 0.8,
        "xtick.direction" : "in",
        "ytick.direction" : "in",
        "savefig.bbox"    : "tight",
        "savefig.dpi"     : 130,
    }

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

        cube           = entry["cube"]
        n_elev, n_az, n_rg = cube.shape
        az             = int(np.clip(az, 0, n_az - 1))
        rg             = int(np.clip(rg, 0, n_rg - 1))
        x_axis         = entry["x_axis"]

        if axis == "range":
            data    = np.asarray(cube[:, :, rg], dtype=np.float32)
            x_label = "azimuth index"
            marker  = az
            title   = f"{source} — range cut @ rg={rg}"
            x_lo, x_hi = 0, n_az
        else:
            data    = np.asarray(cube[:, az, :], dtype=np.float32)
            x_label = "range index"
            marker  = rg
            title   = f"{source} — azimuth cut @ az={az}"
            x_lo, x_hi = 0, n_rg

        sort_idx = np.argsort(x_axis)
        data     = data[sort_idx]
        x_sorted = x_axis[sort_idx]

        vmin, vmax = np.percentile(data[np.isfinite(data)], [1.0, 99.0])
        extent     = [x_lo, x_hi, float(x_sorted[0]), float(x_sorted[-1])]

        plt.rcParams.update(self.PLOT_RC)
        fig, ax = plt.subplots(figsize=(7.6, 3.4))
        im      = ax.imshow(data, cmap="jet", vmin=float(vmin), vmax=float(vmax), extent=extent, aspect="auto", origin="lower", interpolation="nearest")
        ax.axvline(marker, color="white", linewidth=0.9, linestyle="--", alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(x_label)
        ax.set_ylabel("elevation [m]" if entry["has_axis"] else "elevation bin")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label("intensity")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
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

        mean              = np.zeros(cube.shape[1:], dtype=np.float32)
        for i in range(cube.shape[0]):
            mean += np.asarray(cube[i], dtype=np.float32)
        mean             /= cube.shape[0]

        x_axis, has_axis = self._elevation_axis(stamp_dir, cube.shape[0])

        entry = {"cube": cube, "mean": mean, "x_axis": x_axis, "has_axis": has_axis, "mtime": mtime}
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
