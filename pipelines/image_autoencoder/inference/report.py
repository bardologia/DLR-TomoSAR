from __future__ import annotations

from pathlib import Path
from typing  import Dict, List

from tools.reporting.markdown  import MarkdownDoc, ScalarFormatter
from tools.reporting.reporting import ReportAssets


class ImageAeReport:
    METRIC_GROUPS = [
        ("Dataset",                     ["n_patches", "n_channels", "patch_height", "patch_width", "embedding_dim"]),
        ("Reconstruction (physical)",   ["mse_mean", "mse_median", "mae_mean", "rmse", "max_abs_error_mean", "r2", "psnr"]),
        ("Reconstruction (normalized)", ["mse_mean_normalized", "mae_mean_normalized"]),
        ("Latent embedding",            ["embedding_norm_mean", "embedding_dim_std_mean", "embedding_active_dim_fraction"]),
    ]

    FIGURE_SECTIONS = [
        ("Error distribution",     ["error_histogram"]),
        ("Per-channel error",      ["channel_mse"]),
        ("Mean patch intensity",   ["intensity_scatter"]),
        ("Embedding norm",         ["embedding_norm"]),
        ("Best reconstructions",   ["best"]),
        ("Worst reconstructions",  ["worst"]),
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
            "in_channels"       : self.run.in_channels,
            "patch_size"        : self.run.patch_size,
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
        doc = MarkdownDoc("Image Autoencoder Inference Report")

        doc.heading("Run", level=2)
        doc.kv_table(self._summary())

        doc.heading("Metrics", level=2)
        self._write_metrics(doc)

        if self.figures:
            doc.heading("Figures", level=2)
            self._write_figures(doc)

        return doc.save(self.report_path)
