from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from project_paths import ProjectPaths
from web_logger import WebLogger


class RunBrowser:

    MAX_DEPTH = 4

    SUBDIR_TITLES = {
        "profiles"            : "Profiles",
        "pixel_maps"          : "Pixel metric maps",
        "histograms"          : "Metric histograms",
        "slices"              : "Tomogram slices",
        "ssim"                : "SSIM",
        "elev_metrics"        : "Elevation metrics",
        "param_maps"          : "Parameter maps",
        "param_distributions" : "Parameter distributions",
        "param_scatter"       : "Parameter scatter",
        "param_error_maps"    : "Parameter error maps",
        "slots"               : "Slot diagnostics",
    }

    FIGURE_GROUPS = (
        ("profiles_",      "Profiles"),
        ("slice_range_",   "Range slices"),
        ("slice_azimuth_", "Azimuth slices"),
        ("slice_elev_",    "Elevation slices"),
        ("ssim_",          "SSIM"),
        ("param_",         "Parameters"),
        ("slot_",          "Slot diagnostics"),
        ("metric_",        "Metric maps"),
        ("elev_metric_",   "Metric maps"),
    )

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:
        self.paths     = paths
        self.logger    = logger
        self.logs_root = (paths.repo_root / "logs").resolve()

    def list_runs(self) -> list[dict]:
        runs = []
        if self.logs_root.is_dir():
            self._scan(self.logs_root, 0, runs)
        runs.sort(key=lambda r: r["modified"], reverse=True)
        return runs

    def detail(self, run_id: str) -> dict:
        run_dir = self._resolve(run_id)
        if run_dir is None or not (run_dir / "inference").is_dir():
            return {"ok": False, "error": "unknown run"}

        stamps  = sorted((d for d in (run_dir / "inference").iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
        outputs = [self._output_payload(stamp_dir) for stamp_dir in stamps]

        return {"ok": True, "id": run_id, "name": run_dir.name, "outputs": outputs}

    def media_path(self, relative: str) -> Path | None:
        return self._resolve(relative)

    def _resolve(self, relative: str) -> Path | None:
        target = (self.logs_root / relative).resolve()
        if not str(target).startswith(str(self.logs_root)) or not target.exists():
            return None
        return target

    def _scan(self, directory: Path, depth: int, runs: list[dict]) -> None:
        if depth > self.MAX_DEPTH:
            return

        try:
            entries = sorted(d for d in directory.iterdir() if d.is_dir())
        except OSError:
            return

        for entry in entries:
            inference_dir = entry / "inference"

            if inference_dir.is_dir():
                stamps = [d for d in inference_dir.iterdir() if d.is_dir()]
                if stamps:
                    rel    = entry.relative_to(self.logs_root)
                    group  = str(rel.parent) if str(rel.parent) != "." else ""
                    latest = max(d.stat().st_mtime for d in stamps)
                    runs.append({
                        "id"       : str(rel),
                        "name"     : entry.name,
                        "group"    : group,
                        "modified" : datetime.fromtimestamp(latest).isoformat(timespec="minutes"),
                        "outputs"  : len(stamps),
                    })
                continue

            self._scan(entry, depth + 1, runs)

    def _output_payload(self, stamp_dir: Path) -> dict:
        figures_dir    = stamp_dir / "figures"
        animations_dir = stamp_dir / "animations"
        metrics_path   = stamp_dir / "metrics.json"
        report_path    = stamp_dir / "report.md"

        grouped = {}
        if figures_dir.is_dir():
            for subdir in sorted(d for d in figures_dir.iterdir() if d.is_dir()):
                title = self.SUBDIR_TITLES.get(subdir.name, subdir.name.replace("_", " ").capitalize())
                items = [{"name": f.stem, "url": self._url(f)} for f in sorted(subdir.glob("*.png"))]
                if items:
                    grouped.setdefault(title, []).extend(items)

            for figure in sorted(figures_dir.glob("*.png")):
                title = self._group_title(figure.stem)
                grouped.setdefault(title, []).append({"name": figure.stem, "url": self._url(figure)})

        order  = list(self.SUBDIR_TITLES.values()) + ["Range slices", "Azimuth slices", "Elevation slices", "Parameters", "Metric maps", "Diagnostics"]
        groups = [{"title": title, "items": grouped[title]} for title in dict.fromkeys(order) if title in grouped]
        groups += [{"title": title, "items": items} for title, items in grouped.items() if title not in order]

        gifs = []
        if animations_dir.is_dir():
            gifs = [{"name": gif.stem, "url": self._url(gif)} for gif in sorted(animations_dir.glob("*.gif"))]

        metrics = None
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metrics = None

        return {
            "stamp"      : stamp_dir.name,
            "groups"     : groups,
            "gifs"       : gifs,
            "metrics"    : metrics,
            "report_url" : self._url(report_path) if report_path.is_file() else None,
        }

    def _group_title(self, stem: str) -> str:
        for prefix, title in self.FIGURE_GROUPS:
            if stem.startswith(prefix):
                return title
        return "Diagnostics"

    def _url(self, target: Path) -> str:
        return "/runmedia/" + quote(str(target.relative_to(self.logs_root)))
