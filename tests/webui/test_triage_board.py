from __future__ import annotations

from pathlib import Path
from types   import SimpleNamespace

import numpy as np

from triage_board import TriageBoard, TriageStore
from web_logger   import WebLogger


N_AZ, N_RG = 48, 32


def _board(tmp_path: Path) -> TriageBoard:
    paths = SimpleNamespace(logs_dir=tmp_path / "logs")
    return TriageBoard(paths, WebLogger())


def _stamp(tmp_path: Path, with_modes: bool = True, with_aux: bool = True) -> Path:
    stamp = tmp_path / "run" / "inference" / "stamp"
    cubes = stamp / "cubes"
    cubes.mkdir(parents=True)

    error            = np.full((N_AZ, N_RG), 0.1, dtype=np.float32)
    error[0:16, 0:16] = 5.0

    np.save(cubes / "pixel_mse.npy", error)

    if with_modes:
        modes              = np.zeros((N_AZ, N_RG), dtype=np.int8)
        modes[0:16, 0:16]  = 1
        np.save(cubes / "failure_mode.npy", modes)

    if with_aux:
        np.save(cubes / "label_r2.npy", np.full((N_AZ, N_RG), 0.8, dtype=np.float32))

    return stamp


def test_cases_rank_the_worst_block_first(tmp_path):
    stamp = _stamp(tmp_path)
    out   = _board(tmp_path).cases(str(stamp), top_n=10)

    assert out["ok"] is True
    assert out["has_modes"] is True
    assert out["aux"] == ["label_r2"]

    worst = out["cases"][0]
    assert worst["az0"] == 0 and worst["rg0"] == 0
    assert worst["mse_mean"] > out["cases"][-1]["mse_mean"]
    assert worst["mode"] == "missed"
    assert worst["fail_frac"] == 1.0
    assert worst["label_r2"] == np.float32(0.8)
    assert 0 <= worst["worst_az"] < 16 and 0 <= worst["worst_rg"] < 16


def test_cases_without_optional_layers_still_work(tmp_path):
    stamp = _stamp(tmp_path, with_modes=False, with_aux=False)
    out   = _board(tmp_path).cases(str(stamp), top_n=5)

    assert out["ok"] is True
    assert out["has_modes"] is False
    assert out["aux"] == []
    assert "mode" not in out["cases"][0]


def test_cases_require_the_error_cube(tmp_path):
    stamp = tmp_path / "run" / "inference" / "empty"
    (stamp / "cubes").mkdir(parents=True)

    out = _board(tmp_path).cases(str(stamp))

    assert out["ok"] is False
    assert "pixel_mse" in out["error"]


def test_annotations_persist_and_clear(tmp_path):
    stamp = _stamp(tmp_path)
    board = _board(tmp_path)

    saved = board.annotate({"id": str(stamp), "case": "0_0", "verdict": "label problem", "note": "fit is garbage here"})
    assert saved["ok"] is True

    reloaded = board.cases(str(stamp), top_n=10)
    worst    = reloaded["cases"][0]
    assert worst["annotation"]["verdict"] == "label problem"
    assert worst["annotation"]["note"]    == "fit is garbage here"

    cleared = board.annotate({"id": str(stamp), "case": "0_0", "verdict": "", "note": ""})
    assert cleared["ok"] is True
    assert board.cases(str(stamp), top_n=10)["cases"][0]["annotation"] is None


def test_unknown_verdict_is_rejected(tmp_path):
    store = TriageStore(tmp_path / "logs" / "triage")

    out = store.annotate("cube", "0_0", "who knows", "")

    assert out["ok"] is False
