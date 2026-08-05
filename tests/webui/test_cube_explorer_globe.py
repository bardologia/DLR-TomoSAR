from __future__ import annotations

import numpy as np

from tests.webui.conftest import N_AZ, N_RG, N_SLOTS, loaded_cube


def globe_blob(explorer, cube_id, source="pred", amp_min=0.0, max_points=0):
    blob = explorer.globe_points_bin(cube_id, source, amp_min=amp_min, max_points=max_points)
    if blob is None:
        return None

    raw = np.frombuffer(blob, dtype=np.float32)
    return raw[:4], raw[4:].reshape(-1, 5)


def test_globe_meta_present_with_geo(tmp_path):
    explorer, cube_id = loaded_cube(tmp_path, with_params=("pred", "gt"), with_geo=True)
    globe             = explorer.load_status()["cube"]["globe"]

    assert globe is not None
    assert globe["residual_rms_m"] < 0.1
    assert globe["base_height"] == 680.0

    west, south, east, north = globe["bbox"]
    assert west < east and south < north
    assert 12.5 < west < 12.8 and 47.7 < south < 48.0

    anchor = np.array(globe["anchor_ecef"])
    assert 6.3e6 < np.linalg.norm(anchor) < 6.5e6


def test_globe_meta_none_without_geo(tmp_path):
    explorer, cube_id = loaded_cube(tmp_path, with_params=("pred",), with_spacing=True)

    assert explorer.load_status()["cube"]["globe"] is None
    assert explorer.globe_points_bin(cube_id, "pred", amp_min=0.0, max_points=0) is None


def test_globe_points_drop_nan_dem_pixels(tmp_path):
    explorer, cube_id = loaded_cube(tmp_path, with_params=("pred",), with_geo=True)
    header, rows      = globe_blob(explorer, cube_id)

    total = N_SLOTS * N_AZ * N_RG
    assert int(header[1]) == total
    assert rows.shape[0] == total - N_SLOTS

    assert np.all(np.isfinite(rows))
    assert float(np.max(np.abs(rows[:, :3]))) < 100.0


def test_globe_points_offsets_follow_elevation(tmp_path):
    explorer, cube_id = loaded_cube(tmp_path, with_params=("pred",), with_geo=True)
    globe             = explorer.load_status()["cube"]["globe"]
    _, rows           = globe_blob(explorer, cube_id)

    mus  = rows[:, 3]
    amps = rows[:, 4]
    assert np.all((mus >= 0.0) & (mus < 1.0))
    assert np.all((amps >= 0.0) & (amps < 1.0))

    up      = np.array(globe["anchor_ecef"])
    up      = up / np.linalg.norm(up)
    up_comp = rows[:, :3] @ up.astype(np.float32)

    assert float(np.corrcoef(mus, up_comp)[0, 1]) > 0.9


def test_globe_points_reduced_source(tmp_path):
    explorer, cube_id = loaded_cube(tmp_path, with_params=("pred",), with_reduced=True, with_geo=True)
    header, rows      = globe_blob(explorer, cube_id, source="reduced", amp_min=0.5)

    assert int(header[0]) == 2
    assert rows.shape[0] == 2


def test_globe_points_reject_bin_axis_and_unknown(tmp_path):
    explorer, cube_id = loaded_cube(tmp_path, with_params=("pred",), with_geo=True)

    assert explorer.globe_points_bin(cube_id, "full", amp_min=0.0, max_points=0) is None
    assert explorer.globe_points_bin(cube_id, "banana", amp_min=0.0, max_points=0) is None
    assert explorer.globe_points_bin("wrong", "pred", amp_min=0.0, max_points=0) is None
