from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

from web_logger import WebLogger

from tools.reporting.reporting import MetricSectionGrouper


class RunAxes:

    TIMESTAMP = re.compile(r"_(\d{8}_\d{6})")
    K_TAG     = re.compile(r"^K_(\d+)$")
    LOSS_TAG  = re.compile(r"([a-z0-9_]+?)_(\d+(?:\.\d+)?(?:e-?\d+)?)(?:-|$)")

    @classmethod
    def parse(cls, name: str) -> dict | None:
        stamp_match = cls.TIMESTAMP.search(name)
        timestamp   = stamp_match.group(1) if stamp_match else ""
        tag         = name[: stamp_match.start()] if stamp_match else name
        suffix      = name[stamp_match.end() :].lstrip("_") if stamp_match else ""

        parts = tag.split("-", 6)
        if len(parts) < 7:
            return None

        k_match = cls.K_TAG.match(parts[3])
        if k_match is None:
            return None

        loss_tag = parts[6]
        losses   = []
        position = 0
        while position < len(loss_tag):
            loss_match = cls.LOSS_TAG.match(loss_tag, position)
            if loss_match is None:
                return None
            losses.append({"name": loss_match.group(1), "weight": float(loss_match.group(2))})
            position = loss_match.end()

        if not losses:
            return None

        return {
            "model"     : parts[0],
            "head"      : parts[1],
            "matching"  : parts[2],
            "k"         : int(k_match.group(1)),
            "aug"       : parts[4],
            "presence"  : parts[5],
            "loss"      : loss_tag,
            "losses"    : losses,
            "timestamp" : timestamp,
            "suffix"    : suffix,
        }


class RunLeaderboard:

    COLUMNS = (
        {"key": "curve_mse_gt",                    "label": "curve MSE",      "direction": -1, "default": True},
        {"key": "curve_mae_gt",                    "label": "curve MAE",      "direction": -1, "default": False},
        {"key": "curve_rmse_gt",                   "label": "curve RMSE",     "direction": -1, "default": False},
        {"key": "overall_r2_gt",                   "label": "overall R2",     "direction": 1,  "default": True},
        {"key": "psnr_db_gt",                      "label": "PSNR [dB]",      "direction": 1,  "default": False},
        {"key": "pixel_r2_gt_median",              "label": "px R2 med",      "direction": 1,  "default": True},
        {"key": "pixel_cosine_gt_median",          "label": "px cos med",     "direction": 1,  "default": True},
        {"key": "pixel_mse_gt_median",             "label": "px MSE med",     "direction": -1, "default": False},
        {"key": "pixel_peak_err_units_median_gt",  "label": "peak err med",   "direction": -1, "default": True},
        {"key": "fraction_pred_beats_reduced",     "label": "beats capon",    "direction": 1,  "default": True},
        {"key": "relative_mse_reduction",          "label": "MSE reduction",  "direction": 1,  "default": False},
        {"key": "improvement_pixel_mse_mean",      "label": "improv mean",    "direction": 1,  "default": False},
        {"key": "physics_coherence_error_mean",    "label": "phys coh err",   "direction": -1, "default": False},
        {"key": "physics_covariance_error_mean",   "label": "phys cov err",   "direction": -1, "default": False},
        {"key": "physics_valid_fraction",          "label": "phys valid",     "direction": 1,  "default": False},
    )

    HIGHER_BETTER = ("r2", "cosine", "psnr", "ssim", "agreement", "valid_fraction", "beats", "reduction", "improvement")
    LOWER_BETTER  = ("mse", "mae", "rmse", "error", "err", "_d_")

    CONFIG_FILES = (
        ("summary", Path("meta") / "run_summary.json"),
        ("trainer", Path("docs") / "trainer_config.json"),
        ("model",   Path("meta") / "model_config.json"),
    )

    SEED_DIR = re.compile(r"^seed(\d+)$")

    def __init__(self, logger: WebLogger) -> None:
        self.logger = logger
        self.roots  = set()

    def table(self, base: str) -> dict:
        root, error = self._catalog_root(base)
        if error:
            return {"ok": False, "error": error, "rows": []}

        self.roots.add(str(root))

        rows, errors = [], []
        for metrics_path in sorted(root.rglob("inference/*/metrics.json")):
            stamp_dir = metrics_path.parent
            run_dir   = stamp_dir.parent.parent

            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append(f"{metrics_path}: {exc}")
                continue

            values = {c["key"]: metrics[c["key"]] for c in self.COLUMNS if self._is_number(metrics.get(c["key"]))}

            rows.append({
                "id"      : str(stamp_dir),
                "run"     : run_dir.name,
                "group"   : str(run_dir.relative_to(root).parent),
                "stamp"   : stamp_dir.name,
                "mtime"   : stamp_dir.stat().st_mtime,
                "axes"    : RunAxes.parse(run_dir.name),
                "metrics" : values,
            })

        rows.sort(key=lambda row: row["mtime"], reverse=True)
        self.logger.info(f"leaderboard: {len(rows)} inference results under {root}")

        return {"ok": True, "root": str(root), "columns": [dict(c) for c in self.COLUMNS], "rows": rows, "errors": errors}

    def trials(self, base: str) -> dict:
        root, error = self._catalog_root(base)
        if error:
            return {"ok": False, "error": error, "experiments": []}

        self.roots.add(str(root))

        latest = {}
        for metrics_path in sorted(root.rglob("inference/*/metrics.json")):
            stamp_dir = metrics_path.parent
            run_dir   = stamp_dir.parent.parent

            seed_match = self.SEED_DIR.match(run_dir.name)
            if seed_match is None:
                continue

            unit_dir = run_dir.parent
            key      = (str(unit_dir), int(seed_match.group(1)))
            mtime    = stamp_dir.stat().st_mtime
            if key not in latest or mtime > latest[key][1]:
                latest[key] = (metrics_path, mtime)

        units = {}
        for (unit_dir, seed), (metrics_path, _) in latest.items():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue

            values = {c["key"]: metrics[c["key"]] for c in self.COLUMNS if self._is_number(metrics.get(c["key"]))}
            units.setdefault(unit_dir, []).append((seed, values))

        experiments = {}
        for unit_dir, seed_rows in sorted(units.items()):
            unit_path  = Path(unit_dir)
            experiment = str(unit_path.parent.relative_to(root)) if unit_path.parent != root else "."

            aggregated = {}
            for column in self.COLUMNS:
                samples = [values[column["key"]] for _, values in seed_rows if column["key"] in values]
                if not samples:
                    continue
                aggregated[column["key"]] = {
                    "mean" : statistics.fmean(samples),
                    "std"  : statistics.stdev(samples) if len(samples) > 1 else 0.0,
                    "n"    : len(samples),
                }

            experiments.setdefault(experiment, []).append({
                "unit"    : unit_path.name,
                "path"    : unit_dir,
                "seeds"   : sorted(seed for seed, _ in seed_rows),
                "metrics" : aggregated,
            })

        payload = [{"key": name, "units": units_list} for name, units_list in sorted(experiments.items())]
        self.logger.info(f"leaderboard trials: {sum(len(e['units']) for e in payload)} units in {len(payload)} experiments under {root}")

        return {"ok": True, "root": str(root), "columns": [dict(c) for c in self.COLUMNS], "experiments": payload}

    MAX_DIFF_RUNS = 6

    def diff(self, runs: list[str]) -> dict:
        if len(runs) < 2:
            return {"ok": False, "error": "select at least two runs to compare"}
        if len(runs) > self.MAX_DIFF_RUNS:
            return {"ok": False, "error": f"comparison supports at most {self.MAX_DIFF_RUNS} runs"}

        sides = []
        for raw in runs:
            side = self._side(raw)
            if "error" in side:
                return {"ok": False, "error": side["error"]}
            sides.append(side)

        keys       = set().union(*(set(side["metrics"]) for side in sides))
        directions = {key: self._direction(key) for key in keys}
        sections   = [{"title": title, "keys": section_keys} for title, section_keys in MetricSectionGrouper().group(sorted(keys))]

        return {"ok": True, "sides": sides, "directions": directions, "sections": sections}

    def _side(self, raw: str) -> dict:
        stamp_dir = self._stamp_dir(raw)
        if stamp_dir is None:
            return {"error": f"unknown leaderboard entry: {raw}"}

        try:
            metrics = json.loads((stamp_dir / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"error": f"could not read metrics for {stamp_dir}: {exc}"}

        run_dir = stamp_dir.parent.parent
        numeric = {key: value for key, value in metrics.items() if self._is_number(value)}

        config = {}
        for label, rel in self.CONFIG_FILES:
            path = run_dir / rel
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                return {"error": f"could not read {path}: {exc}"}
            self._flatten(label, payload, config)

        return {"id": str(stamp_dir), "run": run_dir.name, "stamp": stamp_dir.name, "axes": RunAxes.parse(run_dir.name), "metrics": numeric, "config": config}

    def _flatten(self, prefix: str, value, out: dict) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                self._flatten(f"{prefix}.{key}", child, out)
            return

        out[prefix] = json.dumps(value) if isinstance(value, list) else value

    def _stamp_dir(self, raw: str) -> Path | None:
        if not raw:
            return None

        stamp_dir = Path(raw).resolve()
        if not any(stamp_dir.is_relative_to(root) for root in self.roots):
            return None
        if not (stamp_dir / "metrics.json").is_file():
            return None
        return stamp_dir

    @classmethod
    def _direction(cls, key: str) -> int:
        if any(token in key for token in cls.HIGHER_BETTER):
            return 1
        if any(token in key for token in cls.LOWER_BETTER):
            return -1
        return 0

    @staticmethod
    def _is_number(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    @staticmethod
    def _catalog_root(raw: str) -> tuple[Path | None, str]:
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
