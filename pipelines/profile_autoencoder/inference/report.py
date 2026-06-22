from __future__ import annotations

from pathlib import Path
from typing  import Dict, List

from tools.reporting.markdown  import MarkdownDoc, ScalarFormatter
from tools.reporting.reporting import ReportAssets


class ProfileAeReport:
    METRIC_GROUPS = [
        ("Dataset",                 ["n_curves", "n_active_curves", "profile_length", "embedding_dim"]),
        ("Reconstruction (physical)", ["mse_mean", "mse_median", "mae_mean", "rmse", "max_abs_error_mean", "r2"]),
        ("Reconstruction (normalized)", ["mse_mean_normalized", "mae_mean_normalized"]),
        ("Profile shape",           ["pearson_mean", "pearson_median", "relative_l2_mean", "relative_l2_median"]),
        ("Power and peak",          ["power_rel_error_mean", "power_rel_error_median", "peak_location_mae", "peak_amplitude_rel_err_mean"]),
        ("Latent embedding",        ["embedding_norm_mean", "embedding_dim_std_mean", "embedding_active_dim_fraction"]),
    ]

    FIGURE_SECTIONS = [
        ("Mean profile",        ["mean_profile"]),
        ("Error distribution",  ["error_histogram"]),
        ("Integrated power",    ["power_scatter"]),
        ("Embedding norm",      ["embedding_norm"]),
        ("Best reconstructions",  ["best"]),
        ("Worst reconstructions", ["worst"]),
        ("Random reconstructions", ["random"]),
    ]

    def __init__(self, output_dir: Path, run, config, metrics: dict, figures: Dict[str, List[Path]], report_path: Path) -> None:
        self.output_dir  = Path(output_dir)
        self.run         = run
        self.config      = config
        self.metrics     = metrics
        self.figures     = figures
        self.report_path = Path(report_path)
        self.assets      = ReportAssets(self.output_dir)

    def _summary(self) -> dict:
        return {
            "model_name"        : self.run.ae_name,
            "embedding_dim"     : self.run.embedding_dim,
            "profile_length"    : int(self.run.x_axis.shape[0]),
            "x_axis_min"        : float(self.run.x_axis.min()),
            "x_axis_max"        : float(self.run.x_axis.max()),
            "split"             : self.run.split_name,
            "split_region"      : str(self.run.split_region.as_tuple()),
            "best_epoch"        : self.run.checkpoint_meta["best_epoch"],
            "best_val_loss"     : self.run.checkpoint_meta["best_val_loss"],
            "preprocessing_dir" : str(self.run.preprocessing_run_directory),
        }

    def _write_metrics(self, doc: MarkdownDoc) -> None:
        for title, keys in self.METRIC_GROUPS:
            rows = [(key, ScalarFormatter.format_scalar(self.metrics[key], adaptive=True)) for key in keys if key in self.metrics]
            if not rows:
                continue

            doc.heading(title, level=3)
            doc.kv_table(rows, header=("Metric", "Value"))

    def _write_figures(self, doc: MarkdownDoc) -> None:
        for title, group_keys in self.FIGURE_SECTIONS:
            paths = [path for key in group_keys for path in self.figures.get(key, [])]
            if not paths:
                continue

            doc.heading(title, level=3)
            for path in paths:
                doc.image(Path(path).stem, self.assets.rel(Path(path)))

    def assemble(self) -> Path:
        doc = MarkdownDoc("Profile Autoencoder Inference Report")

        doc.heading("Run", level=2)
        doc.kv_table(self._summary())

        doc.heading("Metrics", level=2)
        self._write_metrics(doc)

        if self.figures:
            doc.heading("Figures", level=2)
            self._write_figures(doc)

        return doc.save(self.report_path)
