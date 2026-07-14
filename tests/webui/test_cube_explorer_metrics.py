from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from cube_explorer import CubeExplorer
from web_logger    import WebLogger


N_ELEV, N_AZ, N_RG = 5, 8, 6


def _make_cube_run(base: Path) -> Path:
    rng     = np.random.default_rng(0)
    preproc = base / "preproc"
    (preproc / "data").mkdir(parents=True)

    primary = rng.normal(size=(N_AZ, N_RG)) + 1j * rng.normal(size=(N_AZ, N_RG))
    np.save(preproc / "data" / "primary.npy", primary)

    layout = {"global_crop": [0, N_AZ, 0, N_RG], "artifacts": {"primary": "primary.npy"}}
    (preproc / "data" / "dataset.json").write_text(json.dumps(layout))

    run   = base / "group" / "run_a"
    stamp = run / "inference" / "stamp_1"
    (stamp / "cubes").mkdir(parents=True)
    (run / "meta").mkdir(parents=True)

    (run / "meta" / "dataset_creation_config.json").write_text(json.dumps({"preprocessing_run_directory": str(preproc)}))
    (stamp / "metrics.json").write_text(json.dumps({"x_axis_min": -10.0, "x_axis_max": 30.0, "split_region": [0, N_AZ, 0, N_RG]}))

    for source in ("pred", "gt"):
        np.save(stamp / "cubes" / f"{source}_curves.npy", rng.random((N_ELEV, N_AZ, N_RG)).astype(np.float32))

    r2 = rng.random((N_AZ, N_RG)).astype(np.float32)
    r2[0, 0] = np.nan
    np.save(stamp / "cubes" / "pixel_r2.npy", r2)
    np.save(stamp / "cubes" / "physics_valid_mask.npy", rng.random((N_AZ, N_RG)) > 0.5)
    np.save(stamp / "cubes" / "misshaped.npy", rng.random((3, 3)).astype(np.float32))

    return stamp


def _loaded_explorer(base: Path) -> tuple[CubeExplorer, str]:
    _make_cube_run(base)
    explorer = CubeExplorer(paths=None, logger=WebLogger())

    listing = explorer.list_cubes(str(base))
    cube_id = listing["cubes"][0]["id"]
    assert explorer.start_load(cube_id)["ok"]

    deadline = time.time() + 30.0
    while explorer.load_status()["state"] == "loading" and time.time() < deadline:
        time.sleep(0.05)

    status = explorer.load_status()
    assert status["state"] == "ready", status
    return explorer, cube_id


def test_meta_lists_metric_layers(tmp_path):
    explorer, _ = _loaded_explorer(tmp_path)

    layers = explorer.load_status()["cube"]["metric_maps"]
    keys   = {layer["key"] for layer in layers}

    assert keys == {"pixel_r2", "physics_valid_mask"}
    assert all(np.isfinite(layer["vmin"]) and np.isfinite(layer["vmax"]) and layer["vmax"] > layer["vmin"] for layer in layers)
    assert next(layer for layer in layers if layer["key"] == "pixel_r2")["label"] == "R2"


def test_metric_overlay_png(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    png = explorer.metric_overlay_png(cube_id, "pixel_r2", vmin=0.0, vmax=1.0, keep_min=float("-inf"), keep_max=float("inf"), alpha=0.75)
    assert png and png[:4] == b"\x89PNG"

    thresholded = explorer.metric_overlay_png(cube_id, "pixel_r2", vmin=0.0, vmax=1.0, keep_min=0.5, keep_max=float("inf"), alpha=1.0)
    assert thresholded and thresholded[:4] == b"\x89PNG"

    degenerate = explorer.metric_overlay_png(cube_id, "pixel_r2", vmin=2.0, vmax=2.0, keep_min=float("-inf"), keep_max=float("inf"), alpha=0.5)
    assert degenerate and degenerate[:4] == b"\x89PNG"


def test_metric_overlay_rejects_unknown_key(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.metric_overlay_png(cube_id, "banana", 0.0, 1.0, float("-inf"), float("inf"), 0.75) is None
    assert explorer.metric_overlay_png(cube_id, "misshaped", 0.0, 1.0, float("-inf"), float("inf"), 0.75) is None
    assert explorer.metric_overlay_png("wrong", "pixel_r2", 0.0, 1.0, float("-inf"), float("inf"), 0.75) is None


def test_metric_value_at(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    stamp = Path(cube_id)
    r2    = np.load(stamp / "cubes" / "pixel_r2.npy")

    result = explorer.metric_value_at(cube_id, "pixel_r2", az=2, rg=3)
    assert result["ok"] and result["value"] == float(r2[2, 3])

    nan_result = explorer.metric_value_at(cube_id, "pixel_r2", az=0, rg=0)
    assert nan_result["ok"] and nan_result["value"] is None

    clipped = explorer.metric_value_at(cube_id, "pixel_r2", az=99, rg=-1)
    assert clipped["ok"] and clipped["az"] == N_AZ - 1 and clipped["rg"] == 0

    assert not explorer.metric_value_at(cube_id, "banana", 0, 0)["ok"]


def test_cbar_png_whitelist(tmp_path):
    explorer, _ = _loaded_explorer(tmp_path)

    assert explorer.cbar_png("viridis")[:4] == b"\x89PNG"
    assert explorer.cbar_png("banana") is None
