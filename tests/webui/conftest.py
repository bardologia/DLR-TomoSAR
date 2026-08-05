from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tools.sar.geocoding import Wgs84

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from cube_explorer   import CubeExplorer
from notifier        import JobNotifier
from process_manager import ProcessManager
from system_monitor  import ActiveUsers, SystemHistory, SystemMonitor
from web_logger      import WebLogger

N_ELEV, N_AZ, N_RG, N_SLOTS = 5, 8, 6, 2

SLEEP_LONG = "import time\ntime.sleep(30)\n"
ARGS_DUMP  = "import pathlib, sys\npathlib.Path('argv.txt').write_text(' '.join(sys.argv[1:]))\n"


class StubPaths:

    def __init__(self, root: Path) -> None:
        self.repo_root      = root
        self.main_dir       = root / "main"
        self.logs_dir       = root / "logs"
        self.gpu_guard_dir  = root / "logs" / "gpu_guard"
        self.gpu_pools_dir  = root / "logs" / "gpu_pools"
        self.saved_runs_dir = root / "logs" / "saved_runs"

    def has_script(self, key: str) -> bool:
        return (self.main_dir / "analysis" / f"{key}.py").exists()

    def script_entry(self, key: str) -> dict:
        path = self.main_dir / "analysis" / f"{key}.py"
        return {"path": path, "rel": f"main/analysis/{key}.py"}


class StubDescriber:

    def describe(self, key: str, interpreter: str, overrides: dict | None) -> str:
        return f"stub description for {key}"


def job_record(manager: ProcessManager, job_id: str) -> dict:
    with manager.lock:
        return dict(manager.jobs[job_id])


def wait_for_status(manager: ProcessManager, job_id: str, status: str, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job_record(manager, job_id)["status"] == status:
            return True
        time.sleep(0.05)
    return False


def wait_until_finished(manager: ProcessManager, job_id: str, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job_record(manager, job_id)["status"] != "running":
            return True
        time.sleep(0.05)
    return False


def make_geo_fields() -> dict:
    heading, squint, h0, height = 43.0, -1.35, 3700.0, 680.0

    corner_rg = np.array([0.0, N_RG, N_RG, 0.0])
    corner_az = np.array([0.0, 0.0, N_AZ, N_AZ])

    slant  = 3300.0 + corner_rg * 0.6
    along  = corner_az * 0.4 + slant * math.sin(math.radians(squint))
    ground = np.sqrt(slant * slant - (h0 - height) ** 2)

    head_e = math.sin(math.radians(heading))
    head_n = math.cos(math.radians(heading))
    east   = along * head_e + ground * head_n
    north  = along * head_n - ground * head_e

    origin_x, origin_y, origin_z = Wgs84.geodetic_to_ecef(12.65, 47.85, 0.0)
    east_axis, north_axis, _     = Wgs84.enu_axes(12.65, 47.85)

    ecef        = np.array([float(origin_x), float(origin_y), float(origin_z)])[None, :] + np.outer(east, east_axis) + np.outer(north, north_axis)
    lon, lat, _ = Wgs84.ecef_to_geodetic(ecef[:, 0], ecef[:, 1], ecef[:, 2])

    return {
        "h0"       : h0,
        "heading"  : heading,
        "squint"   : squint,
        "antdir"   : 1,
        "r"        : [3300.0 + 0.6 * i for i in range(N_RG)],
        "geo_poly" : {
            "pixels" : np.stack([corner_rg, corner_az], axis=1).ravel().tolist(),
            "lonlat" : np.stack([lon, lat, np.full(4, height)], axis=1).ravel().tolist(),
        },
    }


def make_preproc(base: Path, rng: np.random.Generator | None = None, with_spacing: bool = False, with_geo: bool = False) -> Path:
    rng     = np.random.default_rng(0) if rng is None else rng
    preproc = base / "preproc"
    (preproc / "data").mkdir(parents=True)

    primary = rng.normal(size=(N_AZ, N_RG)) + 1j * rng.normal(size=(N_AZ, N_RG))
    np.save(preproc / "data" / "primary.npy", primary)

    artifacts = {"primary": "primary.npy"}

    if with_geo:
        dem       = np.full((N_AZ, N_RG), 680.0, dtype=np.float32)
        dem[2, 3] = np.nan
        np.save(preproc / "data" / "dem_full.npy", dem)
        artifacts["dem_full"] = "dem_full.npy"

    layout = {"global_crop": [0, N_AZ, 0, N_RG], "artifacts": artifacts}
    (preproc / "data" / "dataset.json").write_text(json.dumps(layout))

    if with_spacing or with_geo:
        per_track = {"T01": {"ps_az": 0.4}, "T02": {"ps_az": 0.41}}

        if with_geo:
            fields = make_geo_fields()
            per_track["T01"].update(fields)
            per_track["T02"].update(fields)

        payload = {
            "labels"      : ["T01", "T02"],
            "reference"   : "T01",
            "shared"      : {"ps_rg": 0.6},
            "per_track"   : per_track,
            "track_files" : [],
        }
        (preproc / "meta").mkdir()
        (preproc / "meta" / "track_parameters.json").write_text(json.dumps(payload))

    return preproc


def make_stamp(run: Path, preproc: Path, rng: np.random.Generator, sources: tuple = ("pred", "gt"), x_min: float = -10.0, shape: tuple = (N_ELEV, N_AZ, N_RG), name: str = "stamp_1") -> Path:
    stamp = run / "inference" / name
    (stamp / "cubes").mkdir(parents=True)
    (run / "meta").mkdir(parents=True)

    (run / "meta" / "dataset_creation_config.json").write_text(json.dumps({"preprocessing_run_directory": str(preproc)}))
    (stamp / "metrics.json").write_text(json.dumps({"x_axis_min": x_min, "x_axis_max": 30.0, "split_region": [0, N_AZ, 0, N_RG]}))

    for source in sources:
        np.save(stamp / "cubes" / f"{source}_curves.npy", rng.random(shape).astype(np.float32))

    return stamp


def make_cube_run(base: Path, sources: tuple = ("pred", "gt"), with_spacing: bool = False, with_reduced: bool = False, with_params: tuple = (), with_metrics: bool = False, with_geo: bool = False) -> Path:
    rng     = np.random.default_rng(0)
    preproc = make_preproc(base, rng, with_spacing, with_geo)
    stamp   = make_stamp(base / "group" / "run_a", preproc, rng, sources)

    if with_reduced:
        reduced          = np.zeros((N_ELEV, N_AZ, N_RG), dtype=np.float32)
        reduced[2, 3, 4] = 2.0
        reduced[4, 5, 1] = 1.0
        np.save(stamp / "cubes" / "reduced_curves.npy", reduced)

    for source in with_params:
        block    = rng.random((3 * N_SLOTS, N_AZ, N_RG)).astype(np.float32)
        block[3] = 0.0
        np.save(stamp / "cubes" / f"params_{source}.npy", block)

    if with_metrics:
        r2       = rng.random((N_AZ, N_RG)).astype(np.float32)
        r2[0, 0] = np.nan
        np.save(stamp / "cubes" / "pixel_r2.npy", r2)
        np.save(stamp / "cubes" / "physics_valid_mask.npy", rng.random((N_AZ, N_RG)) > 0.5)
        np.save(stamp / "cubes" / "misshaped.npy", rng.random((3, 3)).astype(np.float32))

    return stamp


def open_explorer(base: Path, expected: int | None = None) -> tuple[CubeExplorer, list[str]]:
    explorer = CubeExplorer(WebLogger())
    listing  = explorer.list_cubes(str(base))

    assert listing["ok"], listing
    if expected is not None:
        assert len(listing["cubes"]) == expected, listing

    return explorer, [cube["id"] for cube in listing["cubes"]]


def load_cube(explorer: CubeExplorer, cube_id: str, timeout: float = 30.0) -> dict:
    assert explorer.start_load(cube_id)["ok"]

    deadline = time.time() + timeout
    while explorer.load_status()["state"] == "loading" and time.time() < deadline:
        time.sleep(0.05)

    status = explorer.load_status()
    assert status["state"] == "ready", status
    return status


def loaded_cube(base: Path, **kwargs) -> tuple[CubeExplorer, str]:
    make_cube_run(base, **kwargs)
    explorer, cube_ids = open_explorer(base, expected=1)
    load_cube(explorer, cube_ids[0])

    return explorer, cube_ids[0]


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    monkeypatch.setattr(SystemMonitor, "_du_loop", lambda self: None)
    monkeypatch.setattr(SystemHistory, "sample_loop", lambda self: None)
    monkeypatch.setattr(ActiveUsers, "sample_loop", lambda self: None)
    return SystemMonitor(StubPaths(tmp_path), WebLogger())


@pytest.fixture
def make_manager(tmp_path):
    def build(scripts: dict) -> ProcessManager:
        directory = tmp_path / "main" / "analysis"
        directory.mkdir(parents=True)

        for name, body in scripts.items():
            (directory / f"{name}.py").write_text(body)

        paths  = StubPaths(tmp_path)
        logger = WebLogger()
        return ProcessManager(paths, logger, JobNotifier(paths, logger), StubDescriber())

    return build
