from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

from project_paths            import ProjectPaths
from tools.loss.param_loss    import ParamMatcher
from tools.reporting.plotting import PlotBase
from web_logger               import WebLogger


class SliceFigureArchiver(PlotBase):

    LABELS = {
        "pred"    : "Prediction",
        "gt"      : "GT (Gaussian)",
        "reduced" : "Capon reduced",
        "full"    : "Capon full (raw)",
    }

    def render(self, data: np.ndarray, heights: np.ndarray, vmin: float, vmax: float, source: str, axis: str, az: int, rg: int, space: str, path: Path) -> Path:
        x_label   = "azimuth index" if axis == "range" else "range index"
        title_pos = f"range = {rg}" if axis == "range" else f"azimuth = {az}"
        y_label   = "elevation bin" if source == "full" else "elevation [m]"
        cbar      = "intensity (per-column normalised)" if space == "normalized" else "intensity"
        extent    = [0, int(data.shape[1]), float(heights[0]), float(heights[-1])]

        previous = PlotBase.style
        PlotBase.use_style("paper")
        try:
            return self._imshow_figure(
                data,
                x_label        = x_label,
                y_label        = y_label,
                title          = f"{self.LABELS[source]} — {title_pos}",
                cmap           = "jet",
                vmin           = vmin,
                vmax           = vmax,
                extent         = extent,
                origin         = "lower",
                colorbar_label = cbar,
                figsize        = self.figsize(self.FULL_WIDTH),
                path           = path,
            )
        finally:
            PlotBase.use_style(previous)


class CubeExplorer:

    SOURCES       = ("pred", "gt", "reduced", "full")
    PARAM_SOURCES = ("pred", "gt")
    PARAM_FIELDS  = {"amp": 0, "mu": 1, "sigma": 2}
    PARAM_BAD     = "#10151a"

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:
        self.paths    = paths
        self.logger   = logger
        self.archiver = SliceFigureArchiver()
        self.roots    = set()
        self.lock     = threading.Lock()
        self.loaded   = None
        self.status   = {"state": "idle", "id": None, "progress": 0.0, "stage": "", "error": ""}

    def list_cubes(self, base: str) -> dict:
        root, error = self._catalog_root(base)
        if error:
            return {"ok": False, "error": error, "cubes": []}

        self.roots.add(str(root))

        cubes = []
        for cube_file in sorted(root.rglob("inference/*/cubes/pred_curves.npy")):
            stamp_dir = cube_file.parent.parent
            run_dir   = stamp_dir.parent.parent

            cubes.append({
                "id"    : str(stamp_dir),
                "run"   : run_dir.name,
                "group" : str(run_dir.relative_to(root).parent),
                "stamp" : stamp_dir.name,
            })

        cubes.sort(key=lambda c: c["id"], reverse=True)
        return {"ok": True, "root": str(root), "cubes": cubes}

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
            order   = np.argsort(entry["x_axis"])
            heights = np.asarray(entry["x_axis"])[order]
            values  = entry["cube"][:, az, rg][order]
            sources[source]  = {"heights": heights.tolist(), "values": values.astype(float).tolist()}

        return {"ok": True, "az": az, "rg": rg, "sources": sources}

    def slice_ssim(self, cube_id: str, az: int, rg: int, space: str = "physical") -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}
            entries = self.loaded["entries"]
            meta    = self.loaded["meta"]

        gt = entries.get("gt")
        if gt is None:
            return {"ok": True, "az": int(az), "rg": int(rg), "range": {}, "azimuth": {}}

        az = int(np.clip(az, 0, meta["n_az"] - 1))
        rg = int(np.clip(rg, 0, meta["n_rg"] - 1))

        gt_cube = gt["cube"]
        out     = {"range": {}, "azimuth": {}}

        for source in ("pred", "reduced", "full"):
            entry = entries.get(source)
            if entry is None or entry["cube"].shape != gt_cube.shape:
                continue

            cube = entry["cube"]
            out["range"][source]   = self._ssim_score(cube[:, :, rg], gt_cube[:, :, rg], space)
            out["azimuth"][source] = self._ssim_score(cube[:, az, :], gt_cube[:, az, :], space)

        return {"ok": True, "az": az, "rg": rg, "range": out["range"], "azimuth": out["azimuth"]}

    @staticmethod
    def _ssim_score(cur: np.ndarray, ref: np.ndarray, space: str) -> float | None:
        cur = np.nan_to_num(np.asarray(cur, dtype=np.float64))
        ref = np.nan_to_num(np.asarray(ref, dtype=np.float64))

        if space == "normalized":
            cur = cur / np.where((p := cur.max(axis=0, keepdims=True)) > 1e-12, p, 1.0)
            ref = ref / np.where((p := ref.max(axis=0, keepdims=True)) > 1e-12, p, 1.0)

        data_range = float(ref.max() - ref.min())
        if data_range <= 0.0:
            return None

        min_side = min(cur.shape)
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        if win_size < 3:
            return None

        value = float(ssim(ref, cur, data_range=data_range, win_size=win_size))
        return value if np.isfinite(value) else None

    def slice_png(self, cube_id: str, source: str, axis: str, az: int, rg: int, space: str = "physical") -> bytes | None:
        entry = self._entry(cube_id, source)
        if entry is None or axis not in ("range", "azimuth"):
            return None

        n_elev, n_az, n_rg = entry["cube"].shape
        az = int(np.clip(az, 0, n_az - 1))
        rg = int(np.clip(rg, 0, n_rg - 1))

        data, heights, vmin, vmax = self._cut(entry, axis, az, rg, space)

        buf = io.BytesIO()
        plt.imsave(buf, np.flipud(data), cmap="jet", vmin=vmin, vmax=vmax, format="png")
        return buf.getvalue()

    def plane_png(self, cube_id: str, source: str, frac: float, space: str = "physical") -> bytes | None:
        entry = self._entry(cube_id, source)
        if entry is None:
            return None

        cube   = entry["cube"]
        n_elev = cube.shape[0]
        order  = np.argsort(entry["x_axis"])
        pos    = int(round(float(np.clip(frac, 0.0, 1.0)) * (n_elev - 1)))
        elev   = int(order[pos])

        data = np.asarray(cube[elev], dtype=np.float32)

        if space == "normalized":
            peak = float(np.nanmax(data)) if np.isfinite(data).any() else 0.0
            data = data / (peak if peak > 1e-12 else 1.0)
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = entry["vmin"], entry["vmax"]

        buf = io.BytesIO()
        plt.imsave(buf, np.nan_to_num(data, nan=vmin), cmap="jet", vmin=vmin, vmax=vmax, format="png")
        return buf.getvalue()

    def param_map_png(self, cube_id: str, source: str, field: str, slot: int) -> bytes | None:
        resolved = self._param_state(cube_id)
        if resolved is None or field not in (*self.PARAM_FIELDS, "count"):
            return None

        params, meta = resolved

        if source in self.PARAM_SOURCES:
            block = params.get(source)
            if block is None:
                return None
            data, vmin, vmax, cmap = self._param_source_map(block, meta, field, slot)
        elif source == "error":
            if not meta["error"]:
                return None
            data, vmin, vmax, cmap = self._param_error_map(params, meta, field, slot)
        else:
            return None

        palette = plt.get_cmap(cmap).copy()
        palette.set_bad(color=self.PARAM_BAD)

        buf = io.BytesIO()
        plt.imsave(buf, data, cmap=palette, vmin=vmin, vmax=vmax, format="png")
        return buf.getvalue()

    def param_cbar_png(self, cube_id: str, source: str, field: str) -> bytes | None:
        resolved = self._param_state(cube_id)
        if resolved is None or field not in (*self.PARAM_FIELDS, "count"):
            return None

        _, meta = resolved
        cmap    = self._param_cmap(source, field, meta)
        if cmap is None:
            return None

        ramp = np.tile(np.linspace(0.0, 1.0, 256), (12, 1))

        buf = io.BytesIO()
        plt.imsave(buf, ramp, cmap=cmap, vmin=0.0, vmax=1.0, format="png")
        return buf.getvalue()

    def params_at(self, cube_id: str, az: int, rg: int) -> dict:
        resolved = self._param_state(cube_id)
        if resolved is None:
            return {"ok": False, "error": "no parameter cubes are loaded"}

        params, meta = resolved
        n_slots      = meta["n_slots"]
        threshold    = meta["threshold"]

        az = int(np.clip(az, 0, next(iter(params.values())).shape[1] - 1))
        rg = int(np.clip(rg, 0, next(iter(params.values())).shape[2] - 1))

        sources = {}
        for source, block in params.items():
            slots = []
            for k in range(n_slots):
                amp   = float(block[3 * k,     az, rg])
                mu    = float(block[3 * k + 1, az, rg])
                sigma = float(block[3 * k + 2, az, rg])
                slots.append({"amp": amp, "mu": mu, "sigma": sigma, "active": bool(amp >= threshold)})
            sources[source] = slots

        return {"ok": True, "az": az, "rg": rg, "n_slots": n_slots, "threshold": threshold, "sources": sources}

    def _param_state(self, cube_id: str) -> tuple[dict, dict] | None:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return None
            params = self.loaded["params"]
            meta   = self.loaded["meta"]["params"]

        if meta is None or not params:
            return None
        return params, meta

    def _param_source_map(self, block: np.ndarray, meta: dict, field: str, slot: int) -> tuple[np.ndarray, float, float, str]:
        threshold = meta["threshold"]
        n_slots   = meta["n_slots"]

        if field == "count":
            amps = block[0::3]
            data = (amps >= threshold).sum(axis=0).astype(np.float32)
            vmin, vmax = meta["ranges"]["count"]
            return data, vmin, vmax, "viridis"

        slot    = int(np.clip(slot, 0, n_slots - 1))
        channel = np.asarray(block[3 * slot + self.PARAM_FIELDS[field]], dtype=np.float32)

        if field in ("mu", "sigma"):
            active  = block[3 * slot] >= threshold
            channel = np.where(active, channel, np.nan)

        vmin, vmax = meta["ranges"][field]
        return channel, vmin, vmax, "viridis"

    def _param_error_map(self, params: dict, meta: dict, field: str, slot: int) -> tuple[np.ndarray, float, float, str]:
        threshold = meta["threshold"]
        n_slots   = meta["n_slots"]
        pred      = params["pred"]
        gt        = params["gt"]

        if field == "count":
            count_pred = (pred[0::3] >= threshold).sum(axis=0).astype(np.float32)
            count_gt   = (gt[0::3]   >= threshold).sum(axis=0).astype(np.float32)
            vmin, vmax = meta["ranges"]["error_count"]
            return count_pred - count_gt, vmin, vmax, "coolwarm"

        slot   = int(np.clip(slot, 0, n_slots - 1))
        offset = 3 * slot + self.PARAM_FIELDS[field]
        diff   = np.abs(np.asarray(pred[offset], dtype=np.float32) - np.asarray(gt[offset], dtype=np.float32))

        if field in ("mu", "sigma"):
            active = (pred[3 * slot] >= threshold) & (gt[3 * slot] >= threshold)
            diff   = np.where(active, diff, np.nan)

        vmin, vmax = meta["ranges"][f"error_{field}"]
        return diff, vmin, vmax, "inferno"

    def _param_cmap(self, source: str, field: str, meta: dict):
        if source in self.PARAM_SOURCES:
            return "viridis"
        if source == "error" and meta["error"]:
            return "coolwarm" if field == "count" else "inferno"
        return None

    def save_slices(self, cube_id: str, az: int, rg: int, space: str = "physical") -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}
            entries = self.loaded["entries"]
            meta    = self.loaded["meta"]

        stamp_dir = self._stamp_dir(cube_id)
        if stamp_dir is None:
            return {"ok": False, "error": f"unknown cube id: {cube_id}"}
        if space not in ("physical", "normalized"):
            return {"ok": False, "error": f"unknown space: {space}"}

        az = int(np.clip(az, 0, meta["n_az"] - 1))
        rg = int(np.clip(rg, 0, meta["n_rg"] - 1))

        run_dir = stamp_dir.parent.parent
        rel     = Path("figures") / "cube_slices" / f"az{az:04d}_rg{rg:04d}"
        out_dir = run_dir / rel

        files = []
        for source in meta["sources"]:
            for axis in ("range", "azimuth"):
                data, heights, vmin, vmax = self._cut(entries[source], axis, az, rg, space)
                saved = self.archiver.render(data, heights, vmin, vmax, source, axis, az, rg, space, out_dir / f"{axis}_{source}_{space}.png")
                files.append(saved.name)

        self.logger.ok(f"saved {len(files)} slice figures to {out_dir}")
        return {"ok": True, "dir": str(out_dir), "rel": str(rel), "az": az, "rg": rg, "files": files}

    @staticmethod
    def _cut(entry: dict, axis: str, az: int, rg: int, space: str) -> tuple[np.ndarray, np.ndarray, float, float]:
        cube = entry["cube"]
        data = cube[:, :, rg] if axis == "range" else cube[:, az, :]

        if space == "normalized":
            peak = data.max(axis=0, keepdims=True)
            safe = np.where(peak > 1e-12, peak, 1.0)
            data = (data / safe).astype(np.float32)
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = entry["vmin"], entry["vmax"]

        order   = np.argsort(entry["x_axis"])
        heights = np.asarray(entry["x_axis"], dtype=np.float64)[order]
        return data[order], heights, float(vmin), float(vmax)

    def _entry(self, cube_id: str, source: str) -> dict | None:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return None
            return self.loaded["entries"].get(source)

    def _stamp_dir(self, cube_id: str) -> Path | None:
        if not cube_id:
            return None

        stamp_dir = Path(cube_id).resolve()
        if not any(stamp_dir.is_relative_to(root) for root in self.roots):
            return None
        if not (stamp_dir / "cubes" / "pred_curves.npy").is_file():
            return None
        return stamp_dir

    def _catalog_root(self, raw: str) -> tuple[Path | None, str]:
        raw = (raw or "").strip()
        if not raw:
            return None, "set the runs directory in the Results tab first"

        root = Path(raw).expanduser()
        if not root.is_absolute():
            return None, "an absolute path is required"

        root = root.resolve()
        if not root.is_dir():
            return None, f"not a directory: {root}"

        return root, ""

    def _load_worker(self, cube_id: str, stamp_dir: Path) -> None:
        try:
            entries, meta, primary, params = self._load_all(stamp_dir)

            with self.lock:
                self.loaded = {"id": cube_id, "entries": entries, "meta": meta, "primary": primary, "params": params}
                self.status = {"state": "ready", "id": cube_id, "progress": 1.0, "stage": "ready", "error": ""}

            self.logger.muted(f"cube ready: {cube_id} sources={meta['sources']}")
        except Exception as exc:
            with self.lock:
                self.loaded = None
                self.status = {"state": "error", "id": cube_id, "progress": 0.0, "stage": "", "error": str(exc)}

            self.logger.error(f"cube load failed: {cube_id}: {exc}")

    def _load_all(self, stamp_dir: Path) -> tuple[dict, dict, np.ndarray, dict]:
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
        params  = self._load_params(cubes_dir, n_az, n_rg)

        meta = {
            "sources" : [s for s in self.SOURCES if s in entries],
            "n_az"    : n_az,
            "n_rg"    : n_rg,
            "n_elev"  : {s: int(entries[s]["cube"].shape[0]) for s in entries},
            "x_min"   : float(curve_axis[0]),
            "x_max"   : float(curve_axis[-1]),
            "params"  : self._params_meta(params, curve_axis),
        }
        return entries, meta, primary, params

    def _load_params(self, cubes_dir: Path, n_az: int, n_rg: int) -> dict:
        params = {}
        for source in self.PARAM_SOURCES:
            path = cubes_dir / f"params_{source}.npy"
            if not path.is_file():
                continue

            raw = np.asarray(np.load(path), dtype=np.float32)
            if raw.ndim != 3 or raw.shape[0] % 3 != 0 or raw.shape[0] == 0:
                raise ValueError(f"params_{source}.npy is not a (3K, az, rg) cube: shape={raw.shape}")
            if raw.shape[1:] != (n_az, n_rg):
                raise ValueError(f"params_{source}.npy spatial shape {raw.shape[1:]} does not match the {n_az}x{n_rg} cube")

            params[source] = raw

        return params

    def _params_meta(self, params: dict, curve_axis: np.ndarray) -> dict | None:
        if not params:
            return None

        n_slots   = next(iter(params.values())).shape[0] // 3
        threshold = ParamMatcher.ACTIVE_AMP_THR
        has_error = "pred" in params and "gt" in params

        ranges = {
            "amp"   : [0.0, self._field_ceiling(params, 0)],
            "mu"    : [float(curve_axis[0]), float(curve_axis[-1])],
            "sigma" : [0.0, self._field_ceiling(params, 2)],
            "count" : [0.0, float(n_slots)],
        }

        if has_error:
            for field, offset in self.PARAM_FIELDS.items():
                diff = np.abs(params["pred"][offset::3] - params["gt"][offset::3])
                high = float(np.nanpercentile(diff, 99.0)) if diff.size else 1.0
                ranges[f"error_{field}"] = [0.0, high if high > 0.0 else 1.0]
            ranges["error_count"] = [-float(n_slots), float(n_slots)]

        return {
            "sources"   : [s for s in self.PARAM_SOURCES if s in params],
            "n_slots"   : n_slots,
            "threshold" : threshold,
            "error"     : has_error,
            "ranges"    : ranges,
        }

    @staticmethod
    def _field_ceiling(params: dict, offset: int) -> float:
        high = 0.0
        for block in params.values():
            channels = block[offset::3]
            if channels.size:
                high = max(high, float(np.nanpercentile(channels, 99.0)))
        return high if high > 0.0 else 1.0

    def _ingest(self, raw: np.ndarray, x_axis: np.ndarray, advance) -> dict:
        cube = np.empty(raw.shape, dtype=np.float32)

        for i in range(raw.shape[0]):
            plane   = np.asarray(raw[i])
            cube[i] = np.abs(plane) if np.iscomplexobj(plane) else plane
            advance()

        sample = cube[:, :: max(1, cube.shape[1] // 256), :: max(1, cube.shape[2] // 256)]
        sample = sample[np.isfinite(sample)]
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

