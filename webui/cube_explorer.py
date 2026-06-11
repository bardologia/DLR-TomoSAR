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

    SOURCES = ("pred", "gt", "reduced", "full")

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:
        self.paths     = paths
        self.logger    = logger
        self.logs_root = (paths.repo_root / "logs").resolve()
        self.lock      = threading.Lock()
        self.loaded    = None
        self.status    = {"state": "idle", "id": None, "progress": 0.0, "stage": "", "error": ""}

    def list_cubes(self) -> list[dict]:
        cubes = []
        for cube_file in sorted(self.logs_root.rglob("inference/*/cubes/pred_curves.npy")):
            stamp_dir = cube_file.parent.parent
            run_dir   = stamp_dir.parent.parent

            cubes.append({
                "id"    : str(stamp_dir.relative_to(self.logs_root)),
                "run"   : run_dir.name,
                "group" : str(run_dir.relative_to(self.logs_root).parent),
                "stamp" : stamp_dir.name,
            })

        cubes.sort(key=lambda c: c["id"], reverse=True)
        return cubes

    def start_load(self, cube_id: str) -> dict:
        stamp_dir = self._stamp_dir(cube_id)
        if stamp_dir is None:
            return {"ok": False, "error": f"unknown cube id: {cube_id}"}

        with self.lock:
            if self.status["state"] == "loading":
                return {"ok": False, "error": f"a load is already running for {self.status['id']}"}
            if self.status["state"] == "ready" and self.status["id"] == cube_id and self.loaded is not None:
                return {"ok": True}

            self.loaded = None
            self.status = {"state": "loading", "id": cube_id, "progress": 0.0, "stage": "scanning sources", "error": ""}

        threading.Thread(target=self._load_worker, args=(cube_id, stamp_dir), daemon=True).start()
        return {"ok": True}

    def load_status(self) -> dict:
        with self.lock:
            payload = dict(self.status)
            if payload["state"] == "ready" and self.loaded is not None:
                payload["cube"] = self.loaded["meta"]
        return payload

    def primary_png(self, cube_id: str) -> bytes | None:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return None
            primary = self.loaded["primary"]

        vmin, vmax = np.percentile(primary, [1.0, 99.0])

        buf = io.BytesIO()
        plt.imsave(buf, primary, cmap="gray", vmin=float(vmin), vmax=float(vmax), format="png")
        return buf.getvalue()

    def profiles(self, cube_id: str, az: int, rg: int) -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}
            entries = self.loaded["entries"]
            meta    = self.loaded["meta"]

        az = int(np.clip(az, 0, meta["n_az"] - 1))
        rg = int(np.clip(rg, 0, meta["n_rg"] - 1))

        sources = {}
        for source, entry in entries.items():
            order            = np.argsort(entry["x_axis"])
            heights          = np.asarray(entry["x_axis"])[order]
            values           = entry["cube"][:, az, rg][order]
            sources[source]  = {"heights": heights.tolist(), "values": values.astype(float).tolist()}

        return {"ok": True, "az": az, "rg": rg, "sources": sources}

    def slice_png(self, cube_id: str, source: str, axis: str, az: int, rg: int, space: str = "physical") -> bytes | None:
        entry = self._entry(cube_id, source)
        if entry is None or axis not in ("range", "azimuth"):
            return None

        cube               = entry["cube"]
        n_elev, n_az, n_rg = cube.shape
        az                 = int(np.clip(az, 0, n_az - 1))
        rg                 = int(np.clip(rg, 0, n_rg - 1))

        if axis == "range":
            data = cube[:, :, rg]
        else:
            data = cube[:, az, :]

        if space == "normalized":
            peak       = data.max(axis=0, keepdims=True)
            safe       = np.where(peak > 1e-12, peak, 1.0)
            data       = (data / safe).astype(np.float32)
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = entry["vmin"], entry["vmax"]

        sort_idx = np.argsort(entry["x_axis"])
        data     = np.flipud(data[sort_idx])

        buf = io.BytesIO()
        plt.imsave(buf, data, cmap="jet", vmin=vmin, vmax=vmax, format="png")
        return buf.getvalue()

    def _entry(self, cube_id: str, source: str) -> dict | None:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return None
            return self.loaded["entries"].get(source)

    def _stamp_dir(self, cube_id: str) -> Path | None:
        stamp_dir = (self.logs_root / cube_id).resolve()
        if not str(stamp_dir).startswith(str(self.logs_root)):
            return None
        if not (stamp_dir / "cubes" / "pred_curves.npy").is_file():
            return None
        return stamp_dir

    def _load_worker(self, cube_id: str, stamp_dir: Path) -> None:
        try:
            entries, meta, primary = self._load_all(stamp_dir)

            with self.lock:
                self.loaded = {"id": cube_id, "entries": entries, "meta": meta, "primary": primary}
                self.status = {"state": "ready", "id": cube_id, "progress": 1.0, "stage": "ready", "error": ""}

            self.logger.muted(f"cube ready: {cube_id} sources={meta['sources']}")
        except Exception as exc:
            with self.lock:
                self.loaded = None
                self.status = {"state": "error", "id": cube_id, "progress": 0.0, "stage": "", "error": str(exc)}

            self.logger.error(f"cube load failed: {cube_id}: {exc}")

    def _load_all(self, stamp_dir: Path) -> tuple[dict, dict, np.ndarray]:
        cubes_dir = stamp_dir / "cubes"
        pred_raw  = np.load(cubes_dir / "pred_curves.npy", mmap_mode="r")
        if pred_raw.ndim != 3:
            raise ValueError(f"pred_curves.npy is not a 3D cube: shape={pred_raw.shape}")

        n_elev, n_az, n_rg = pred_raw.shape
        curve_axis         = self._curve_axis(stamp_dir, n_elev)

        plan = [("pred", pred_raw, curve_axis)]
        for source in ("gt", "reduced"):
            path = cubes_dir / f"{source}_curves.npy"
            if path.is_file():
                plan.append((source, np.load(path, mmap_mode="r"), curve_axis))

        full_raw = self._full_raw(stamp_dir, n_az, n_rg)
        if full_raw is not None:
            plan.append(("full", full_raw, np.arange(full_raw.shape[0], dtype=np.float64)))

        total = sum(raw.shape[0] for _, raw, _ in plan)
        done  = [0]

        def advance(source: str) -> None:
            done[0] += 1
            with self.lock:
                self.status["progress"] = done[0] / total
                self.status["stage"]    = source

        entries = {}
        for source, raw, x_axis in plan:
            if raw.shape[1:] != (n_az, n_rg):
                raise ValueError(f"source '{source}' spatial shape {raw.shape[1:]} does not match pred {(n_az, n_rg)}")
            entries[source] = self._ingest(raw, x_axis, lambda s=source: advance(s))

        primary = self._primary_db(stamp_dir, n_az, n_rg)

        meta = {
            "sources" : [s for s in self.SOURCES if s in entries],
            "n_az"    : n_az,
            "n_rg"    : n_rg,
            "n_elev"  : {s: int(entries[s]["cube"].shape[0]) for s in entries},
        }
        return entries, meta, primary

    def _ingest(self, raw: np.ndarray, x_axis: np.ndarray, advance) -> dict:
        cube = np.empty(raw.shape, dtype=np.float32)

        for i in range(raw.shape[0]):
            plane   = np.asarray(raw[i])
            cube[i] = np.abs(plane) if np.iscomplexobj(plane) else plane
            advance()

        sample     = cube[:, :: max(1, cube.shape[1] // 256), :: max(1, cube.shape[2] // 256)]
        sample     = sample[np.isfinite(sample)]
        vmin, vmax = (np.percentile(sample, [1.0, 99.0]) if sample.size else (0.0, 1.0))

        return {
            "cube"   : cube,
            "x_axis" : x_axis,
            "vmin"   : float(vmin),
            "vmax"   : float(vmax),
        }

    def _metrics(self, stamp_dir: Path) -> dict:
        metrics_path = stamp_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"metrics.json missing in {stamp_dir}; rerun inference to regenerate it")

        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def _curve_axis(self, stamp_dir: Path, n_elev: int) -> np.ndarray:
        metrics = self._metrics(stamp_dir)
        return np.linspace(float(metrics["x_axis_min"]), float(metrics["x_axis_max"]), n_elev)

    def _preproc_layout(self, stamp_dir: Path) -> tuple[Path, dict] | None:
        meta_path = stamp_dir.parent.parent / "meta" / "dataset_creation_config.json"
        if not meta_path.is_file():
            return None

        payload     = json.loads(meta_path.read_text(encoding="utf-8"))
        preproc_dir = Path(payload["preprocessing_run_directory"])
        layout_path = preproc_dir / "data" / "dataset.json"
        if not layout_path.is_file():
            return None

        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        return preproc_dir, layout

    def _crop_bounds(self, stamp_dir: Path, layout: dict, n_az: int, n_rg: int) -> tuple[int, int, int, int]:
        metrics = self._metrics(stamp_dir)
        if "split_region" not in metrics:
            raise KeyError(f"metrics.json in {stamp_dir} has no split_region; rerun inference to regenerate it")

        az_start, az_end, rg_start, rg_end = (int(v) for v in metrics["split_region"])
        if (az_end - az_start, rg_end - rg_start) != (n_az, n_rg):
            raise ValueError(f"split_region {metrics['split_region']} does not match the {n_az}x{n_rg} cube")

        global_crop = layout["global_crop"]

        az_lo = az_start - global_crop[0]
        az_hi = az_end   - global_crop[0]
        rg_lo = rg_start - global_crop[2]
        rg_hi = rg_end   - global_crop[2]
        return az_lo, az_hi, rg_lo, rg_hi

    def _full_raw(self, stamp_dir: Path, n_az: int, n_rg: int) -> np.ndarray | None:
        resolved = self._preproc_layout(stamp_dir)
        if resolved is None:
            return None

        preproc_dir, layout = resolved

        tomo_name = layout["artifacts"].get("tomogram_full")
        if not tomo_name:
            return None

        tomo_path = preproc_dir / "data" / tomo_name
        if not tomo_path.is_file():
            return None

        az_lo, az_hi, rg_lo, rg_hi = self._crop_bounds(stamp_dir, layout, n_az, n_rg)

        raw = np.load(tomo_path, mmap_mode="r")
        if raw.ndim != 3:
            raise ValueError(f"full tomogram is not a 3D cube: shape={raw.shape}")
        if az_lo < 0 or rg_lo < 0 or az_hi > raw.shape[1] or rg_hi > raw.shape[2]:
            raise ValueError(f"cube region az[{az_lo}:{az_hi}] rg[{rg_lo}:{rg_hi}] falls outside the full tomogram {raw.shape}")

        return raw[:, az_lo:az_hi, rg_lo:rg_hi]

    def _primary_db(self, stamp_dir: Path, n_az: int, n_rg: int) -> np.ndarray:
        resolved = self._preproc_layout(stamp_dir)
        if resolved is None:
            raise FileNotFoundError(f"cannot resolve the preprocessing run for {stamp_dir}; the primary SLC is required for the cube map")

        preproc_dir, layout = resolved

        primary_name = layout["artifacts"].get("primary")
        if not primary_name:
            raise FileNotFoundError(f"dataset.json in {preproc_dir} lists no primary artifact")

        primary_path = preproc_dir / "data" / primary_name
        if not primary_path.is_file():
            raise FileNotFoundError(f"primary SLC missing: {primary_path}")

        az_lo, az_hi, rg_lo, rg_hi = self._crop_bounds(stamp_dir, layout, n_az, n_rg)

        raw = np.load(primary_path, mmap_mode="r")
        if raw.ndim != 2:
            raise ValueError(f"primary SLC is not a 2D image: shape={raw.shape}")
        if az_lo < 0 or rg_lo < 0 or az_hi > raw.shape[0] or rg_hi > raw.shape[1]:
            raise ValueError(f"cube region az[{az_lo}:{az_hi}] rg[{rg_lo}:{rg_hi}] falls outside the primary SLC {raw.shape}")

        amplitude = np.abs(np.asarray(raw[az_lo:az_hi, rg_lo:rg_hi])).astype(np.float32)
        return 20.0 * np.log10(np.maximum(amplitude, 1e-12))

