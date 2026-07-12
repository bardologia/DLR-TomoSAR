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

    for source in ("pred", "gt", "reduced"):
        np.save(stamp / "cubes" / f"{source}_curves.npy", rng.random((N_ELEV, N_AZ, N_RG)).astype(np.float32))

    return stamp


def _loaded_explorer(base: Path) -> tuple[CubeExplorer, str]:
    _make_cube_run(base)
    explorer = CubeExplorer(paths=None, logger=WebLogger())

    listing = explorer.list_cubes(str(base))
    assert listing["ok"] and len(listing["cubes"]) == 1

    cube_id = listing["cubes"][0]["id"]
    assert explorer.start_load(cube_id)["ok"]

    deadline = time.time() + 30.0
    while explorer.load_status()["state"] == "loading" and time.time() < deadline:
        time.sleep(0.05)

    status = explorer.load_status()
    assert status["state"] == "ready", status
    return explorer, cube_id


def test_save_slices_writes_paper_figures(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    result = explorer.save_slices(cube_id, az=3, rg=2, space="physical")
    assert result["ok"], result

    out_dir = Path(result["dir"])
    assert out_dir == Path(cube_id) / "figures" / "cube_slices" / "az0003_rg0002"
    assert result["rel"] == "figures/cube_slices/az0003_rg0002"

    expected = {f"{axis}_{source}_physical.png" for source in ("pred", "gt", "reduced") for axis in ("range", "azimuth")}
    assert set(result["files"]) == expected

    for name in expected:
        target = out_dir / name
        assert target.is_file() and target.stat().st_size > 0


def test_save_slices_normalized_space_clips_indices(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    result = explorer.save_slices(cube_id, az=999, rg=-5, space="normalized")
    assert result["ok"], result
    assert result["az"] == N_AZ - 1 and result["rg"] == 0
    assert all(name.endswith("_normalized.png") for name in result["files"])
    assert all((Path(result["dir"]) / name).is_file() for name in result["files"])


def test_save_slices_restores_figure_style(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    from tools.reporting.plotting import PlotBase

    assert explorer.save_slices(cube_id, az=1, rg=1)["ok"]
    assert PlotBase.style == "report"


def test_save_slices_rejects_unknown_space(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    result = explorer.save_slices(cube_id, az=0, rg=0, space="banana")
    assert not result["ok"]


def test_save_slices_requires_loaded_cube(tmp_path):
    stamp    = _make_cube_run(tmp_path)
    explorer = CubeExplorer(paths=None, logger=WebLogger())

    result = explorer.save_slices(str(stamp), az=0, rg=0)
    assert not result["ok"]


def test_slice_png_still_serves_after_cut_refactor(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    png = explorer.slice_png(cube_id, "pred", "range", az=0, rg=2)
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"
