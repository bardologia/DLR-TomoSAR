from __future__ import annotations

import math

import numpy as np
import pytest

from tools.sar.geocoding        import SceneGeocoder, Wgs84
from tools.sar.track_parameters import TrackParameters


@pytest.fixture(scope="module")
def track_parameters(meta_dir) -> TrackParameters:
    return TrackParameters.load(meta_dir / TrackParameters.FILENAME)


@pytest.fixture(scope="module")
def reference_geocoder(track_parameters) -> SceneGeocoder:
    return SceneGeocoder.from_track_parameters(track_parameters)


class TestWgs84:

    def test_geodetic_ecef_round_trip(self):
        lon = np.array([12.61, 12.70, -70.0, 0.0, 179.5])
        lat = np.array([47.84, 47.89, -33.0, 0.0, -89.0])
        height = np.array([684.0, 715.0, -30.0, 0.0, 4000.0])

        x, y, z = Wgs84.geodetic_to_ecef(lon, lat, height)
        lon_back, lat_back, height_back = Wgs84.ecef_to_geodetic(x, y, z)

        np.testing.assert_allclose(lon_back, lon, atol=1e-9)
        np.testing.assert_allclose(lat_back, lat, atol=1e-9)
        np.testing.assert_allclose(height_back, height, atol=1e-4)

    def test_enu_axes_orthonormal(self):
        east, north, up = Wgs84.enu_axes(12.65, 47.85)
        frame           = np.stack([east, north, up], axis=0)

        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.cross(east, north), up, atol=1e-12)

    def test_up_axis_points_outward(self):
        _, _, up = Wgs84.enu_axes(12.65, 47.85)
        x, y, z  = Wgs84.geodetic_to_ecef(12.65, 47.85, 0.0)
        radial   = np.array([float(x), float(y), float(z)])

        assert np.dot(up, radial / np.linalg.norm(radial)) > 0.99


@pytest.mark.real_data
class TestSceneGeocoderRealTracks:

    def test_reference_corners_fit_below_one_metre(self, reference_geocoder):
        assert reference_geocoder.residual_rms_m < 1.0
        assert reference_geocoder.residual_max_m < 2.0

    def test_every_track_fits_below_one_metre(self, track_parameters):
        for label, parameters in zip(track_parameters.labels, track_parameters.parameters):
            geocoder = SceneGeocoder(parameters)
            assert geocoder.residual_rms_m < 1.0, f"track {label} corner rms {geocoder.residual_rms_m:.2f} m"

    def test_fitted_rotation_is_small(self, reference_geocoder):
        angle = math.degrees(math.atan2(reference_geocoder.rot_sin, reference_geocoder.rot_cos))
        assert abs(angle) < 1.0

    def test_geocode_reproduces_the_corners(self, reference_geocoder):
        geo           = reference_geocoder
        lon, lat, _   = geo.geocode(geo.corner_az, geo.corner_rg, geo.corner_h)
        east_scale    = 111320.0 * math.cos(math.radians(geo.anchor_lat))
        north_scale   = 110574.0

        east_error  = (lon - geo.corner_lon) * east_scale
        north_error = (lat - geo.corner_lat) * north_scale
        assert float(np.max(np.hypot(east_error, north_error))) < 2.0

    def test_cube_crop_lands_inside_the_scene_polygon(self, reference_geocoder, dataset_json):
        az_lo, az_hi, rg_lo, rg_hi = dataset_json["global_crop"]

        corners_az = np.array([az_lo, az_lo, az_hi, az_hi], dtype=np.float64)
        corners_rg = np.array([rg_lo, rg_hi, rg_lo, rg_hi], dtype=np.float64)
        heights    = np.full(4, float(np.mean(reference_geocoder.corner_h)))

        lon, lat, _ = reference_geocoder.geocode(corners_az, corners_rg, heights)

        assert np.all(lon >= reference_geocoder.corner_lon.min() - 1e-3)
        assert np.all(lon <= reference_geocoder.corner_lon.max() + 1e-3)
        assert np.all(lat >= reference_geocoder.corner_lat.min() - 1e-3)
        assert np.all(lat <= reference_geocoder.corner_lat.max() + 1e-3)

    def test_higher_targets_shift_away_from_the_track(self, reference_geocoder):
        geo = reference_geocoder

        low_e, low_n   = geo._aligned_enu([8000.0], [2000.0], [650.0])
        high_e, high_n = geo._aligned_enu([8000.0], [2000.0], [750.0])

        head = np.array([math.sin(geo.heading_rad), math.cos(geo.heading_rad)])
        perp = np.array([math.cos(geo.heading_rad), -math.sin(geo.heading_rad)])

        low_across  = float(np.array([low_e[0], low_n[0]]) @ perp)
        high_across = float(np.array([high_e[0], high_n[0]]) @ perp)
        assert 50.0 < high_across - low_across < 150.0

        low_along  = float(np.array([low_e[0], low_n[0]]) @ head)
        high_along = float(np.array([high_e[0], high_n[0]]) @ head)
        assert abs(high_along - low_along) < 1.0

    def test_ecef_output_matches_lonlat(self, reference_geocoder):
        lon, lat, ecef = reference_geocoder.geocode([5000.0], [1000.0], [700.0])

        x, y, z = Wgs84.geodetic_to_ecef(lon, lat, np.array([700.0]))
        np.testing.assert_allclose(ecef[:, 0], x, atol=1e-6)
        np.testing.assert_allclose(ecef[:, 1], y, atol=1e-6)
        np.testing.assert_allclose(ecef[:, 2], z, atol=1e-6)


class TestSceneGeocoderValidation:

    @staticmethod
    def _reference() -> dict:
        return {
            "ps_az"    : 0.4,
            "ps_rg"    : 0.6,
            "h0"       : 3700.0,
            "heading"  : 43.0,
            "squint"   : -1.35,
            "antdir"   : 1,
            "r"        : [3300.0 + 0.6 * i for i in range(101)],
            "geo_poly" : {
                "pixels" : [0.0, 0.0, 100.0, 0.0, 100.0, 5000.0, 0.0, 5000.0],
                "lonlat" : [12.61, 47.846, 684.0, 12.652, 47.823, 686.0, 12.70, 47.858, 715.0, 12.663, 47.886, 648.0],
            },
        }

    def test_missing_key_raises(self):
        reference = self._reference()
        reference.pop("geo_poly")

        with pytest.raises(KeyError, match="geo_poly"):
            SceneGeocoder(reference)

    def test_left_looking_raises(self):
        reference           = self._reference()
        reference["antdir"] = -1

        with pytest.raises(ValueError, match="right-looking"):
            SceneGeocoder(reference)

    def test_non_uniform_slant_axis_raises(self):
        reference          = self._reference()
        reference["ps_rg"] = 0.7

        with pytest.raises(ValueError, match="not uniform"):
            SceneGeocoder(reference)

    def test_too_few_corners_raises(self):
        reference                       = self._reference()
        reference["geo_poly"]["pixels"] = [0.0, 0.0, 100.0, 0.0]
        reference["geo_poly"]["lonlat"] = [12.61, 47.846, 684.0, 12.652, 47.823, 686.0]

        with pytest.raises(ValueError, match="at least 3"):
            SceneGeocoder(reference)

    def test_target_above_slant_reach_raises(self):
        reference = self._reference()
        geocoder  = SceneGeocoder(reference)

        with pytest.raises(ValueError, match="platform height"):
            geocoder.geocode([0.0], [0.0], [350.0])
