from __future__ import annotations

from datetime import datetime
from pathlib  import Path
from typing   import Dict

from configuration.inference_config import InferenceConfig, InferencePaths


class InferenceMetadata:
    def __init__(self, config: InferenceConfig) -> None:
        self.config     = config
        self._p         = config.paths          # shorthand
        self.output_dir = self._resolve_output_dir()

    def _resolve_output_dir(self) -> Path:
        base = self.config.run_directory / "inference"
        if self.config.output_subdir:
            return base / self.config.output_subdir
        return base / datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / self._p.figures_subdir

    @property
    def animations_dir(self) -> Path:
        return self.output_dir / self._p.animations_subdir

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / self._p.logs_subdir

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / self._p.metrics_filename

    @property
    def report_path(self) -> Path:
        return self.output_dir / self._p.report_filename

    def figure_path(self, name: str, ext: str = "png") -> Path:
        return self.figures_dir / f"{name}.{ext}"

    def gif_path(self, axis: str) -> Path:
        return self.animations_dir / f"walk_{axis}.gif"

    def cube_dir(self) -> Path:
        return self.output_dir / self._p.cubes_subdir

    def all_figure_paths(
        self,
        n_range_slices    : int,
        n_azimuth_slices  : int,
        n_elevation_slices: int,
        range_offsets     : list[int] | None = None,
        az_offsets        : list[int] | None = None,
        elev_indices      : list[int] | None = None,
    ) -> Dict[str, Path]:
     
        paths: Dict[str, Path] = {}

        for tag in ("best", "worst", "random"):
            paths[f"profiles_{tag}"] = self.figure_path(f"profiles_{tag}")

        for name in ("pixel_mse_map", "pixel_r2_map", "pixel_peak_map"):
            paths[name] = self.figure_path(name)

        for name in (
            "metric_histograms",
            "param_maps",
            "param_distributions",
            "param_scatter",
            "param_error_maps",
        ):
            paths[name] = self.figure_path(name)

        for ax in ("range", "azimuth", "elev"):
            paths[f"ssim_{ax}"] = self.figure_path(f"ssim_{ax}")

        paths["elev_metric_curves"] = self.figure_path("elev_metric_curves")

        for i in range(n_range_slices):
            idx = range_offsets[i] if range_offsets else i
            paths[f"slice_range_{idx}"] = self.figure_path(f"slice_range_{idx}")

        for i in range(n_azimuth_slices):
            idx = az_offsets[i] if az_offsets else i
            paths[f"slice_azimuth_{idx}"] = self.figure_path(f"slice_azimuth_{idx}")

        for i in range(n_elevation_slices):
            idx = elev_indices[i] if elev_indices else i
            paths[f"slice_elev_idx_{idx}"] = self.figure_path(f"slice_elev_idx_{idx}")

        return paths

 
    def create_dirs(self) -> None:
        for d in (
            self.output_dir,
            self.figures_dir,
            self.animations_dir,
            self.logs_dir,
            self.cube_dir(),
        ):
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"InferenceMetadata(\n"
            f"  output_dir    = {self.output_dir}\n"
            f"  figures_dir   = {self.figures_dir}\n"
            f"  animations_dir= {self.animations_dir}\n"
            f"  logs_dir      = {self.logs_dir}\n"
            f"  metrics_path  = {self.metrics_path}\n"
            f")"
        )
