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


def _make_preproc(base: Path) -> Path:
    rng     = np.random.default_rng(0)
    preproc = base / "preproc"
    (preproc / "data").mkdir(parents=True)

    primary = rng.normal(size=(N_AZ, N_RG)) + 1j * rng.normal(size=(N_AZ, N_RG))
    np.save(preproc / "data" / "primary.npy", primary)

    layout = {"global_crop": [0, N_AZ, 0, N_RG], "artifacts": {"primary": "primary.npy"}}
    (preproc / "data" / "dataset.json").write_text(json.dumps(layout))
    return preproc


def _make_run(base: Path, preproc: Path, name: str, seed: int, x_min: float = -10.0, shape: tuple = (N_ELEV, N_AZ, N_RG)) -> Path:
    rng   = np.random.default_rng(seed)
    run   = base / name
    stamp = run / "inference" / "stamp_1"
    (stamp / "cubes").mkdir(parents=True)
    (run / "meta").mkdir(parents=True)

    (run / "meta" / "dataset_creation_config.json").write_text(json.dumps({"preprocessing_run_directory": str(preproc)}))
    (stamp / "metrics.json").write_text(json.dumps({"x_axis_min": x_min, "x_axis_max": 30.0, "split_region": [0, N_AZ, 0, N_RG]}))

    for source in ("pred", "gt"):
        np.save(stamp / "cubes" / f"{source}_curves.npy", rng.random(shape).astype(np.float32))

    return stamp


def _loaded(base: Path) -> tuple[CubeExplorer, str, str]:
    preproc = _make_preproc(base)
    stamp_a = _make_run(base, preproc, "run_a", seed=1)
    stamp_b = _make_run(base, preproc, "run_b", seed=2)

    explorer = CubeExplorer(paths=None, logger=WebLogger())
    listing  = explorer.list_cubes(str(base))
    assert listing["ok"] and len(listing["cubes"]) == 2

    cube_a = str(stamp_a)
    assert explorer.start_load(cube_a)["ok"]

    deadline = time.time() + 30.0
    while explorer.load_status()["state"] == "loading" and time.time() < deadline:
        time.sleep(0.05)
    assert explorer.load_status()["state"] == "ready"

    return explorer, cube_a, str(stamp_b)


def test_attach_adds_predb_and_diff(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)

    result = explorer.attach_second(cube_a, cube_b)
    assert result["ok"], result

    meta = result["cube"]
    assert meta["sources"] == ["pred", "predb", "diff", "gt"]
    assert meta["attached"]["id"] == cube_b
    assert meta["attached"]["run"] == "run_b"

    pred_a = np.load(Path(cube_a) / "cubes" / "pred_curves.npy")
    pred_b = np.load(Path(cube_b) / "cubes" / "pred_curves.npy")

    profiles = explorer.profiles(cube_a, az=2, rg=3)
    assert set(profiles["sources"]) == {"pred", "predb", "diff", "gt"}

    expected_diff = (pred_a - pred_b)[:, 2, 3]
    order = np.argsort(np.linspace(-10.0, 30.0, N_ELEV))
    assert np.allclose(profiles["sources"]["diff"]["values"], expected_diff[order])

    assert explorer.slice_png(cube_a, "predb", "range", az=0, rg=0)[:4] == b"\x89PNG"
    assert explorer.slice_png(cube_a, "diff",  "range", az=0, rg=0)[:4] == b"\x89PNG"
    assert explorer.plane_png(cube_a, "diff", frac=0.5, space="normalized")[:4] == b"\x89PNG"


def test_diff_range_is_symmetric(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)
    explorer.attach_second(cube_a, cube_b)

    entry = explorer._entry(cube_a, "diff")
    assert entry["diverging"] is True
    assert entry["vmin"] == -entry["vmax"]
    assert entry["vmax"] > 0


def test_detach_removes_comparison(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)
    explorer.attach_second(cube_a, cube_b)

    result = explorer.detach_second(cube_a)
    assert result["ok"]
    assert result["cube"]["sources"] == ["pred", "gt"]
    assert result["cube"]["attached"] is None
    assert explorer.slice_png(cube_a, "predb", "range", az=0, rg=0) is None


def test_attach_rejects_self_and_unknown(tmp_path):
    explorer, cube_a, _ = _loaded(tmp_path)

    assert not explorer.attach_second(cube_a, cube_a)["ok"]
    assert not explorer.attach_second(cube_a, str(tmp_path / "nowhere"))["ok"]
    assert not explorer.attach_second(str(tmp_path / "other"), cube_a)["ok"]


def test_attach_rejects_mismatched_axis(tmp_path):
    preproc = _make_preproc(tmp_path)
    stamp_a = _make_run(tmp_path, preproc, "run_a", seed=1)
    stamp_c = _make_run(tmp_path, preproc, "run_c", seed=3, x_min=-99.0)

    explorer = CubeExplorer(paths=None, logger=WebLogger())
    explorer.list_cubes(str(tmp_path))
    explorer.start_load(str(stamp_a))

    deadline = time.time() + 30.0
    while explorer.load_status()["state"] == "loading" and time.time() < deadline:
        time.sleep(0.05)

    result = explorer.attach_second(str(stamp_a), str(stamp_c))
    assert not result["ok"] and "elevation axis" in result["error"]


def test_ssim_covers_predb(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)
    explorer.attach_second(cube_a, cube_b)

    result = explorer.slice_ssim(cube_a, az=2, rg=2)
    assert result["ok"]
    assert "predb" in result["range"]
    assert "diff" not in result["range"]


def test_save_slices_includes_comparison_sources(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)
    explorer.attach_second(cube_a, cube_b)

    result = explorer.save_slices(cube_a, az=1, rg=1)
    assert result["ok"]

    expected = {f"{axis}_{source}_physical.png" for source in ("pred", "predb", "diff", "gt") for axis in ("range", "azimuth")}
    assert set(result["files"]) == expected
