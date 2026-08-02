from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from ab_autopsy import AbAutopsy
from web_logger import WebLogger


N_ELEV, N_AZ, N_RG = 6, 32, 32


def _stamp(base: Path, name: str, mse: np.ndarray, rmse: float) -> Path:
    stamp = base / name / "inference" / "stamp"
    cubes = stamp / "cubes"
    cubes.mkdir(parents=True)

    (stamp / "metrics.json").write_text(json.dumps({
        "curve_rmse_gt" : rmse,
        "overall_r2_gt" : 1.0 - rmse,
        "x_axis_min"    : -10.0,
        "x_axis_max"    : 30.0,
        "n_pixels"      : N_AZ * N_RG,
    }), encoding="utf-8")

    rng = np.random.default_rng(hash(name) % 2**32)
    np.save(cubes / "pixel_mse.npy", mse.astype(np.float32))
    np.save(cubes / "pred_curves.npy", rng.random((N_ELEV, N_AZ, N_RG)).astype(np.float32))
    np.save(cubes / "gt_curves.npy",   rng.random((N_ELEV, N_AZ, N_RG)).astype(np.float32))

    return stamp


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    mse_a              = np.full((N_AZ, N_RG), 0.1)
    mse_b              = np.full((N_AZ, N_RG), 0.1)
    mse_a[0:16, 0:16]  = 3.0
    mse_b[16:32, 0:16] = 3.0

    return _stamp(tmp_path, "run_a", mse_a, rmse=0.2), _stamp(tmp_path, "run_b", mse_b, rmse=0.4)


def test_compare_ranks_metric_gaps_and_finds_hotspots(tmp_path):
    stamp_a, stamp_b = _pair(tmp_path)

    out = AbAutopsy(None, WebLogger()).compare(str(stamp_a), str(stamp_b))

    assert out["ok"] is True
    assert out["run_a"] == "run_a" and out["run_b"] == "run_b"

    rmse_row = next(row for row in out["metrics"] if row["key"] == "curve_rmse_gt")
    assert rmse_row["winner"] == "A"
    assert "n_pixels" not in {row["key"] for row in out["metrics"]}

    winners = {spot["winner"] for spot in out["hotspots"]}
    assert winners == {"A", "B"}

    b_wins = [spot for spot in out["hotspots"] if spot["winner"] == "B"]
    assert all(spot["az0"] < 16 and spot["rg0"] < 16 for spot in b_wins)


def test_compare_rejects_region_mismatch(tmp_path):
    stamp_a, _ = _pair(tmp_path)
    stamp_c    = _stamp(tmp_path, "run_c", np.full((8, 8), 0.1), rmse=0.3)

    out = AbAutopsy(None, WebLogger()).compare(str(stamp_a), str(stamp_c))

    assert out["ok"] is False
    assert "different regions" in out["error"]


def test_compare_missing_metrics_fails_cleanly(tmp_path):
    out = AbAutopsy(None, WebLogger()).compare(str(tmp_path / "nope"), str(tmp_path / "nope2"))

    assert out["ok"] is False


def test_profile_returns_both_predictions(tmp_path):
    stamp_a, stamp_b = _pair(tmp_path)

    out = AbAutopsy(None, WebLogger()).profile(str(stamp_a), str(stamp_b), az=4, rg=5)

    assert out["ok"] is True
    assert len(out["a"]) == N_ELEV
    assert len(out["b"]) == N_ELEV
    assert len(out["gt"]) == N_ELEV
    assert out["x_axis"][0] == -10.0


def test_profile_outside_region_fails(tmp_path):
    stamp_a, stamp_b = _pair(tmp_path)

    assert AbAutopsy(None, WebLogger()).profile(str(stamp_a), str(stamp_b), az=999, rg=0)["ok"] is False
