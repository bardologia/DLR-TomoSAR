from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from configuration.training_config import GeometryConfig
from tools.config_cli              import ConfigCli
from tools.logger                  import Logger
from tools.tomo_geometry           import TomoGeometry
from tools.track_baselines         import BaselineExtractor, TrackBaselines


def _default_track_paths() -> dict:
    return {
        "PS03": "/ste/rnd/17SARTOM/FL01/PS03/T01L/INF/INF-TRACK/track_sar_resa_17sartom0103_Lhh_t01L.rat",
        "PS07": "/ste/rnd/17SARTOM/FL01/PS07/T01L/INF/INF-TRACK/track_sar_resa_17sartom0107_Lhh_t01L.rat",
        "PS11": "/ste/rnd/17SARTOM/FL01/PS11/T01L/INF/INF-TRACK/track_sar_resa_17sartom0111_Lhh_t01L.rat",
        "PS15": "/ste/rnd/17SARTOM/FL01/PS15/T01L/INF/INF-TRACK/track_sar_resa_17sartom0115_Lhh_t01L.rat",
    }


@dataclass
class ExtractBaselinesEntryConfig:
    track_paths      : dict = field(default_factory=_default_track_paths)
    pass_directories : list = field(default_factory=list)

    azimuth_start    : int  = 1000
    azimuth_end      : int  = 16000

    component        : str  = "perpendicular"
    wavelength       : float = 0.23
    slant_range      : float = 5000.0
    look_angle_deg   : float = 45.0

    dataset_dir      : str  = ""


class BaselineExtractionRun:
    def __init__(self, config: ExtractBaselinesEntryConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger

    def _build_extractor(self) -> BaselineExtractor:
        window = (self.config.azimuth_start, self.config.azimuth_end)

        if self.config.pass_directories:
            return BaselineExtractor.from_pass_directories(self.config.pass_directories, azimuth_window=window)

        return BaselineExtractor(self.config.track_paths, azimuth_window=window)

    def _log_geometry(self, table: TrackBaselines) -> None:
        geometry_cfg = GeometryConfig(
            wavelength       = self.config.wavelength,
            slant_range      = self.config.slant_range,
            look_angle_deg   = self.config.look_angle_deg,
            baselines        = table.baselines(self.config.component, look_angle_deg=self.config.look_angle_deg),
            baselines_origin = "extracted",
        )

        x_axis   = torch.linspace(-20.0, 80.0, 101)
        geometry = TomoGeometry(geometry_cfg, x_axis)

        self.logger.kv_table(geometry.describe(), title="Resulting Tomographic Geometry")
        self.logger.kv_table({
            "Extracted baselines [m]" : ", ".join(f"{b:.3f}" for b in geometry_cfg.baselines),
            "Manual default [m]"      : ", ".join(f"{b:.3f}" for b in GeometryConfig().baselines),
            "kz [rad/m]"              : ", ".join(f"{float(k):.4f}" for k in geometry.kz),
        }, title="Extracted vs Manual Defaults")

    def _maybe_save(self, table: TrackBaselines) -> None:
        if not self.config.dataset_dir:
            return

        out_path = GeometryConfig().baselines_file(self.config.dataset_dir)
        table.save(out_path)
        self.logger.subsection(f"Baselines written: {out_path}")

    def run(self) -> TrackBaselines:
        extractor = self._build_extractor()
        table     = extractor.extract()

        self.logger.kv_table(table.describe(), title="Track Baselines")
        self._log_geometry(table)
        self._maybe_save(table)

        return table


def main() -> None:
    config = ConfigCli(ExtractBaselinesEntryConfig(), description="Extract per-track baselines from INF-TRACK files and optionally write them into a dataset meta directory").apply()
    logger = Logger(log_dir="logs", name="extract_baselines")

    BaselineExtractionRun(config, logger).run()

    logger.close()


if __name__ == "__main__":
    main()
