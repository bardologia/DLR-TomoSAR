from __future__ import annotations

import math

import numpy as np

from tools.sar.track_parameters import TrackParameters


class Wgs84:
    SEMI_MAJOR_M    = 6378137.0
    FLATTENING      = 1.0 / 298.257223563
    SEMI_MINOR_M    = SEMI_MAJOR_M * (1.0 - FLATTENING)
    ECCENTRICITY_SQ = FLATTENING * (2.0 - FLATTENING)
    SECOND_ECC_SQ   = ECCENTRICITY_SQ / (1.0 - ECCENTRICITY_SQ)

    @classmethod
    def geodetic_to_ecef(cls, lon_deg, lat_deg, height_m):
        lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
        lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
        h   = np.asarray(height_m, dtype=np.float64)

        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        prime   = cls.SEMI_MAJOR_M / np.sqrt(1.0 - cls.ECCENTRICITY_SQ * sin_lat * sin_lat)

        x = (prime + h) * cos_lat * np.cos(lon)
        y = (prime + h) * cos_lat * np.sin(lon)
        z = (prime * (1.0 - cls.ECCENTRICITY_SQ) + h) * sin_lat
        return x, y, z

    @classmethod
    def ecef_to_geodetic(cls, x, y, z):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        radial = np.hypot(x, y)
        theta  = np.arctan2(z * cls.SEMI_MAJOR_M, radial * cls.SEMI_MINOR_M)

        lat = np.arctan2(
            z + cls.SECOND_ECC_SQ * cls.SEMI_MINOR_M * np.sin(theta) ** 3,
            radial - cls.ECCENTRICITY_SQ * cls.SEMI_MAJOR_M * np.cos(theta) ** 3,
        )
        lon = np.arctan2(y, x)

        sin_lat = np.sin(lat)
        prime   = cls.SEMI_MAJOR_M / np.sqrt(1.0 - cls.ECCENTRICITY_SQ * sin_lat * sin_lat)
        height  = radial / np.cos(lat) - prime

        return np.degrees(lon), np.degrees(lat), height

    @classmethod
    def enu_axes(cls, lon_deg, lat_deg):
        lon = math.radians(lon_deg)
        lat = math.radians(lat_deg)

        east  = np.array([-math.sin(lon), math.cos(lon), 0.0])
        north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
        up    = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
        return east, north, up


class SceneGeocoder:

    REQUIRED_KEYS = ("ps_az", "ps_rg", "h0", "heading", "squint", "antdir", "r", "geo_poly")

    def __init__(self, reference: dict) -> None:
        missing = [key for key in self.REQUIRED_KEYS if key not in reference]
        if missing:
            raise KeyError(f"reference track parameters lack {missing}; the scene cannot be geocoded")

        if reference["antdir"] <= 0:
            raise ValueError("the geocoding model assumes a right-looking geometry (antdir > 0)")

        self.ps_az       = float(reference["ps_az"])
        self.ps_rg       = float(reference["ps_rg"])
        self.h0          = float(reference["h0"])
        self.heading_rad = math.radians(float(reference["heading"]))
        self.squint_rad  = math.radians(float(reference["squint"]))

        slant   = np.asarray(reference["r"], dtype=np.float64)
        self.r0 = float(slant[0])

        drift = abs(self.r0 + (slant.size - 1) * self.ps_rg - float(slant[-1]))
        if drift > 0.5 * self.ps_rg:
            raise ValueError(f"slant-range axis is not uniform with spacing ps_rg={self.ps_rg}: end drift {drift:.3f} m")

        poly   = reference["geo_poly"]
        pixels = np.asarray(poly["pixels"], dtype=np.float64).reshape(-1, 2)
        lonlat = np.asarray(poly["lonlat"], dtype=np.float64).reshape(-1, 3)

        if pixels.shape[0] != lonlat.shape[0] or pixels.shape[0] < 3:
            raise ValueError(f"geo_poly needs at least 3 matched corners: got {pixels.shape[0]} pixel pairs and {lonlat.shape[0]} lonlat triples")

        self.corner_rg  = pixels[:, 0]
        self.corner_az  = pixels[:, 1]
        self.corner_lon = lonlat[:, 0]
        self.corner_lat = lonlat[:, 1]
        self.corner_h   = lonlat[:, 2]

        self._fit_anchor()

    @classmethod
    def from_track_parameters(cls, params: TrackParameters) -> "SceneGeocoder":
        return cls(params.parameters[0])

    @property
    def residual_rms_m(self) -> float:
        return float(np.sqrt(np.mean(self.corner_residuals_m ** 2)))

    @property
    def residual_max_m(self) -> float:
        return float(np.max(self.corner_residuals_m))

    def _model_plane(self, az_px, rg_px, height_m):
        slant = self.r0 + np.asarray(rg_px, dtype=np.float64) * self.ps_rg
        along = np.asarray(az_px, dtype=np.float64) * self.ps_az + slant * math.sin(self.squint_rad)
        drop  = self.h0 - np.asarray(height_m, dtype=np.float64)

        if np.any(slant <= np.abs(drop)):
            raise ValueError("slant range shorter than the platform height drop; heights and track altitude are inconsistent")

        ground   = np.sqrt(slant * slant - drop * drop)
        head_e   = math.sin(self.heading_rad)
        head_n   = math.cos(self.heading_rad)

        east  = along * head_e + ground * head_n
        north = along * head_n - ground * head_e
        return east, north

    def _observed_enu(self):
        x, y, z = Wgs84.geodetic_to_ecef(self.corner_lon, self.corner_lat, np.zeros_like(self.corner_lon))
        delta   = np.stack([x, y, z], axis=1) - np.asarray(self.origin_ecef)

        return np.stack([delta @ self.east_axis, delta @ self.north_axis], axis=1)

    def _fit_anchor(self) -> None:
        self.anchor_lon = float(np.mean(self.corner_lon))
        self.anchor_lat = float(np.mean(self.corner_lat))

        origin           = Wgs84.geodetic_to_ecef(self.anchor_lon, self.anchor_lat, 0.0)
        self.origin_ecef = np.array([float(origin[0]), float(origin[1]), float(origin[2])])

        self.east_axis, self.north_axis, self.up_axis = Wgs84.enu_axes(self.anchor_lon, self.anchor_lat)

        model_e, model_n = self._model_plane(self.corner_az, self.corner_rg, self.corner_h)
        model            = np.stack([model_e, model_n], axis=1)
        observed         = self._observed_enu()

        model_center    = model.mean(axis=0)
        observed_center = observed.mean(axis=0)
        model_c         = model - model_center
        observed_c      = observed - observed_center

        angle    = math.atan2(
            float(np.sum(model_c[:, 0] * observed_c[:, 1] - model_c[:, 1] * observed_c[:, 0])),
            float(np.sum(model_c[:, 0] * observed_c[:, 0] + model_c[:, 1] * observed_c[:, 1])),
        )
        self.rot_cos = math.cos(angle)
        self.rot_sin = math.sin(angle)

        rotated_center = np.array([
            self.rot_cos * model_center[0] - self.rot_sin * model_center[1],
            self.rot_sin * model_center[0] + self.rot_cos * model_center[1],
        ])
        self.shift = observed_center - rotated_center

        aligned = np.stack([
            self.rot_cos * model[:, 0] - self.rot_sin * model[:, 1] + self.shift[0],
            self.rot_sin * model[:, 0] + self.rot_cos * model[:, 1] + self.shift[1],
        ], axis=1)
        self.corner_residuals_m = np.linalg.norm(aligned - observed, axis=1)

    def _aligned_enu(self, az_px, rg_px, height_m):
        raw_e, raw_n = self._model_plane(az_px, rg_px, height_m)

        east  = self.rot_cos * raw_e - self.rot_sin * raw_n + self.shift[0]
        north = self.rot_sin * raw_e + self.rot_cos * raw_n + self.shift[1]
        return east, north

    def geocode(self, az_px, rg_px, height_m):
        height       = np.asarray(height_m, dtype=np.float64)
        east, north  = self._aligned_enu(az_px, rg_px, height)

        ground        = self.origin_ecef[None, :] + np.outer(east, self.east_axis) + np.outer(north, self.north_axis)
        lon, lat, _   = Wgs84.ecef_to_geodetic(ground[:, 0], ground[:, 1], ground[:, 2])
        x, y, z       = Wgs84.geodetic_to_ecef(lon, lat, np.broadcast_to(height, lon.shape))

        return lon, lat, np.stack([x, y, z], axis=1)
