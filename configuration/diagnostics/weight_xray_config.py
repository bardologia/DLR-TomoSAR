from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional


@dataclass
class WeightXrayThresholds:
    dead_abs_threshold     : float = 1e-6
    dead_fraction_warn     : float = 0.50
    dead_fraction_critical : float = 0.95

    dead_unit_norm_frac    : float = 1e-3
    dead_unit_fraction_warn: float = 0.10

    uniform_cv_threshold   : float = 0.05
    constant_std_threshold : float = 1e-9

    rank_ratio_warn        : float = 0.50
    rank_ratio_critical    : float = 0.10

    explode_abs_threshold  : float = 1e3
    spectral_norm_warn     : float = 1e2
    bias_abs_threshold     : float = 1e2

    norm_scale_dead_value  : float = 1e-3
    norm_scale_dead_frac   : float = 0.10
    running_var_dead_value : float = 1e-6
    running_var_dead_frac  : float = 0.10

    init_ratio_low         : float = 0.10
    init_ratio_high        : float = 3.00

    duplicate_cosine       : float = 0.999
    duplicate_fraction_warn: float = 0.05


@dataclass
class WeightXrayConfig:
    checkpoint_path : Path = Path("logs")
    output_dir      : Path = Path("results/weight_xray")

    checkpoint_filename  : str  = "best_model.pt"
    state_dict_keys      : list = field(default_factory=lambda: ["params", "state_dict", "model_state_dict", "model", "weights", "net"])

    thresholds : WeightXrayThresholds = field(default_factory=WeightXrayThresholds)

    svd_max_dim          : int = 4096
    duplicate_max_units  : int = 512
    histogram_sample_max : int = 2_000_000
    max_layer_histograms : int = 24

    make_plots    : bool = True
    embed_images  : bool = False

    @property
    def resolved_checkpoint_path(self) -> Path:
        path = Path(self.checkpoint_path)
        if path.is_dir():
            return path / self.checkpoint_filename
        return path

    @property
    def report_directory(self) -> Path:
        return Path(self.output_dir)

    @property
    def plots_directory(self) -> Path:
        return self.report_directory / "plots"

    @property
    def report_markdown_path(self) -> Path:
        return self.report_directory / "weight_xray.md"

    @property
    def report_json_path(self) -> Path:
        return self.report_directory / "weight_xray.json"


@dataclass
class WeightXrayEntryConfig:
    runs_dir : Path = Path("runs")

    checkpoint_filename : str = "best_model.pt"
    output_subdir       : str = "weight_xray"

    make_plots   : bool = True
    embed_images : bool = False

    dead_abs_threshold      : float = 1e-6
    dead_fraction_warn      : float = 0.50
    dead_unit_fraction_warn : float = 0.10
    uniform_cv_threshold    : float = 0.05
    rank_ratio_warn         : float = 0.50
    explode_abs_threshold   : float = 1e3
    duplicate_cosine        : float = 0.999

    svd_max_dim          : int = 4096
    duplicate_max_units  : int = 512
    max_layer_histograms : int = 24

    def to_config(self, run_dir: Path) -> WeightXrayConfig:
        thresholds = WeightXrayThresholds(
            dead_abs_threshold      = self.dead_abs_threshold,
            dead_fraction_warn      = self.dead_fraction_warn,
            dead_unit_fraction_warn = self.dead_unit_fraction_warn,
            uniform_cv_threshold    = self.uniform_cv_threshold,
            rank_ratio_warn         = self.rank_ratio_warn,
            explode_abs_threshold   = self.explode_abs_threshold,
            duplicate_cosine        = self.duplicate_cosine,
        )

        return WeightXrayConfig(
            checkpoint_path     = run_dir,
            output_dir          = Path(run_dir) / self.output_subdir,
            checkpoint_filename = self.checkpoint_filename,
            thresholds          = thresholds,
            svd_max_dim          = self.svd_max_dim,
            duplicate_max_units  = self.duplicate_max_units,
            max_layer_histograms = self.max_layer_histograms,
            make_plots           = self.make_plots,
            embed_images         = self.embed_images,
        )
