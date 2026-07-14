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


N_ELEV, N_AZ, N_RG, N_SLOTS = 5, 8, 6, 2


def _make_cube_run(base: Path, with_params: tuple = ("pred", "gt")) -> Path:
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

    for source in with_params:
        block = rng.random((3 * N_SLOTS, N_AZ, N_RG)).astype(np.float32)
        block[3] = 0.0
        np.save(stamp / "cubes" / f"params_{source}.npy", block)

    return stamp


def _loaded_explorer(base: Path, with_params: tuple = ("pred", "gt")) -> tuple[CubeExplorer, str]:
    _make_cube_run(base, with_params)
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


def test_meta_reports_param_block(tmp_path):
    explorer, _ = _loaded_explorer(tmp_path)

    meta = explorer.load_status()["cube"]["params"]

    assert meta["sources"] == ["pred", "gt"]
    assert meta["n_slots"] == N_SLOTS
    assert meta["error"] is True
    assert meta["threshold"] == 1e-3
    assert meta["ranges"]["mu"] == [-10.0, 30.0]
    assert meta["ranges"]["count"] == [0.0, float(N_SLOTS)]
    assert meta["ranges"]["error_count"] == [-float(N_SLOTS), float(N_SLOTS)]
    assert set(meta["ranges"]) >= {"amp", "sigma", "error_amp", "error_mu", "error_sigma"}


def test_meta_without_params_is_none(tmp_path):
    explorer, _ = _loaded_explorer(tmp_path, with_params=())

    assert explorer.load_status()["cube"]["params"] is None


def test_param_map_png_for_all_fields(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    for source in ("pred", "gt", "error"):
        for field in ("amp", "mu", "sigma", "count"):
            png = explorer.param_map_png(cube_id, source, field, slot=0)
            assert png and png[:4] == b"\x89PNG", (source, field)


def test_param_map_rejects_unknown_source_and_field(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.param_map_png(cube_id, "banana", "amp", 0) is None
    assert explorer.param_map_png(cube_id, "pred", "banana", 0) is None
    assert explorer.param_map_png("wrong_id", "pred", "amp", 0) is None


def test_param_map_error_requires_both_sources(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path, with_params=("pred",))

    assert explorer.param_map_png(cube_id, "pred", "amp", 0) is not None
    assert explorer.param_map_png(cube_id, "error", "amp", 0) is None
    assert explorer.param_map_png(cube_id, "gt", "amp", 0) is None


def test_param_cbar_png(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.param_cbar_png(cube_id, "pred", "amp")[:4] == b"\x89PNG"
    assert explorer.param_cbar_png(cube_id, "error", "count")[:4] == b"\x89PNG"
    assert explorer.param_cbar_png(cube_id, "banana", "amp") is None


def test_params_at_returns_slot_values(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    stamp  = Path(cube_id)
    pred   = np.load(stamp / "cubes" / "params_pred.npy")
    result = explorer.params_at(cube_id, az=2, rg=3)

    assert result["ok"]
    assert result["n_slots"] == N_SLOTS

    slot0 = result["sources"]["pred"][0]
    assert slot0["amp"]   == float(pred[0, 2, 3])
    assert slot0["mu"]    == float(pred[1, 2, 3])
    assert slot0["sigma"] == float(pred[2, 2, 3])
    assert slot0["active"] is True

    slot1 = result["sources"]["pred"][1]
    assert slot1["amp"]    == 0.0
    assert slot1["active"] is False


def test_params_at_clips_indices_and_requires_params(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    clipped = explorer.params_at(cube_id, az=999, rg=-4)
    assert clipped["ok"] and clipped["az"] == N_AZ - 1 and clipped["rg"] == 0

    bare_explorer, bare_id = _loaded_explorer(tmp_path / "bare", with_params=())
    assert not bare_explorer.params_at(bare_id, az=0, rg=0)["ok"]


def test_param_map_slot_clipped(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.param_map_png(cube_id, "pred", "amp", slot=99) is not None
    assert explorer.param_map_png(cube_id, "pred", "amp", slot=-3) is not None


def test_params_with_nan_survive_load_and_lookup(tmp_path):
    stamp = _make_cube_run(tmp_path)
    pred  = np.load(stamp / "cubes" / "params_pred.npy")
    pred[1, 0, 0] = np.nan
    pred[4]       = np.nan
    np.save(stamp / "cubes" / "params_pred.npy", pred)

    explorer = CubeExplorer(paths=None, logger=WebLogger())
    listing  = explorer.list_cubes(str(tmp_path))
    cube_id  = listing["cubes"][0]["id"]
    assert explorer.start_load(cube_id)["ok"]

    deadline = time.time() + 30.0
    while explorer.load_status()["state"] == "loading" and time.time() < deadline:
        time.sleep(0.05)
    assert explorer.load_status()["state"] == "ready"

    meta = explorer.load_status()["cube"]["params"]
    assert all(np.isfinite(v) for bounds in meta["ranges"].values() for v in bounds)

    result = explorer.params_at(cube_id, az=0, rg=0)
    assert result["ok"]
    assert np.isnan(result["sources"]["pred"][0]["mu"])
    assert result["sources"]["pred"][1]["active"] is False

    assert explorer.param_map_png(cube_id, "pred", "mu", 0) is not None
    assert explorer.param_map_png(cube_id, "error", "mu", 1) is not None
