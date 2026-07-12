from __future__ import annotations

import os
from pathlib import Path

from pipelines.backbone.inference.report            import Report
from pipelines.shared.inference.inference_scheduler import InferenceScheduler
from tools.data.io             import FileIO
from tools.metrics.scoring     import SeedAggregation
from tools.monitoring.logger   import Logger
from tools.reporting.markdown  import MarkdownTable, ScalarFormatter
from tools.runtime.run_tag     import RunTag


class SeedGroupReport:
    def __init__(self, run_dirs: list[Path], inference_config, logger: Logger) -> None:
        if len(run_dirs) < 2:
            raise ValueError(f"A seed-group report needs at least two seed runs, got {len(run_dirs)}: {[str(d) for d in run_dirs]}")

        self.run_dirs = [Path(d) for d in run_dirs]
        self.config   = inference_config
        self.logger   = logger

        self.subdir     = inference_config.output_subdir
        self.group_dir  = Path(os.path.commonpath([str(d) for d in self.run_dirs]))
        self.output_dir = self.group_dir / "inference" / self.subdir
        self.seeds      = [str(d.relative_to(self.group_dir)) for d in self.run_dirs]

    def _run_output_dir(self, run_dir: Path) -> Path:
        return run_dir / "inference" / self.subdir

    def _load_metrics(self) -> list[dict]:
        per_seed = []

        for run_dir in self.run_dirs:
            path = self._run_output_dir(run_dir) / self.config.paths.metrics_filename
            if not path.is_file():
                raise FileNotFoundError(f"No metrics found at {path}; every seed run must finish its inference before the seed-group report can aggregate them")
            per_seed.append(FileIO.load_json(path))

        return per_seed

    def _aggregate(self, per_seed: list[dict]) -> tuple[dict, dict]:
        keys = sorted({key for metrics in per_seed for key in metrics})
        return SeedAggregation.aggregate(per_seed, keys)

    def _write_metrics(self, means: dict, stds: dict) -> Path:
        payload = {
            "n_seeds"   : len(self.run_dirs),
            "seed_runs" : {seed: str(run_dir) for seed, run_dir in zip(self.seeds, self.run_dirs)},
            "mean"      : means,
            "std"       : stds,
        }

        path = self.output_dir / self.config.paths.metrics_filename
        FileIO.save_json(payload, path)

        return path

    @staticmethod
    def _fmt(value) -> str:
        return ScalarFormatter.format_scalar(value, precision=6, adaptive=True)

    def _mean_std_cell(self, mean: float, std: float | None) -> str:
        if std is None:
            return self._fmt(mean)

        return f"{self._fmt(mean)} ± {self._fmt(std)}"

    def _build_seed_runs_section(self) -> list[str]:
        out = ["\n## 1. Seed runs\n"]

        table = MarkdownTable(("Seed", "Run directory", "Report"))
        for seed, run_dir in zip(self.seeds, self.run_dirs):
            report_path = self._run_output_dir(run_dir) / self.config.paths.report_filename
            table.add_row(f"`{seed}`", str(run_dir), f"[{self.config.paths.report_filename}]({os.path.relpath(report_path, self.output_dir)})")

        out.append("\n".join(table.render()))
        out.append("")

        return out

    def _report_keys(self, means: dict) -> list[str]:
        return [key for key in means if not Report._is_per_slice_ssim(key) and "_raw" not in key and key not in Report._METRIC_SKIP_KEYS]

    def _grouped_keys(self, keys: list[str]) -> list[tuple[str, list[str]]]:
        groups: dict[str, list[str]] = {title: [] for title, _match in Report._METRIC_TAXONOMY}

        for key in sorted(keys):
            title = next(title for title, match in Report._METRIC_TAXONOMY if match(key))
            groups[title].append(key)

        ordered = sorted((title for title in groups if groups[title]), key=Report._section_order)
        return [(title.split(" ", 1)[1], groups[title]) for title in ordered]

    def _build_metrics_section(self, per_seed: list[dict], means: dict, stds: dict) -> list[str]:
        out = ["\n## 2. Aggregated metrics\n"]
        out.append(
            f"Every scalar metric aggregated over the {len(self.seeds)} seed runs: the cell is the across-seed mean, "
            "the ± term the across-seed sample standard deviation (ddof = 1). Per-seed columns show each run's raw value.\n"
        )

        for index, (title, keys) in enumerate(self._grouped_keys(self._report_keys(means)), start=1):
            out.append(f"\n### 2.{index} {title}\n")

            table = MarkdownTable(("Metric", "Mean ± seed std", *self.seeds))
            for key in keys:
                per_seed_cells = [self._fmt(metrics.get(key)) if metrics.get(key) is not None else None for metrics in per_seed]
                table.add_row(f"`{key}`", self._mean_std_cell(means[key], stds.get(key)), *per_seed_cells)

            out.append("\n".join(table.render()))
            out.append("")

        return out

    def _write_report(self, per_seed: list[dict], means: dict, stds: dict) -> Path:
        lines = ["# TomoSAR Seed-Group Inference Report", ""]
        lines.append(f"_Generated on {RunTag.timestamp()}_")
        lines.append("")
        lines.append(f"Aggregate over {len(self.seeds)} seed replicas of one training, inferred under the shared stamp `{self.subdir}`. Each seed keeps its own full report (linked below); this report reduces their metrics to mean ± std.")

        lines += self._build_seed_runs_section()
        lines += self._build_metrics_section(per_seed, means, stds)

        path = self.output_dir / self.config.paths.report_filename
        path.write_text("\n".join(lines), encoding="utf-8")

        return path

    def run(self) -> Path:
        FileIO.ensure_dirs(self.output_dir)

        per_seed     = self._load_metrics()
        means, stds  = self._aggregate(per_seed)
        metrics_path = self._write_metrics(means, stds)
        report_path  = self._write_report(per_seed, means, stds)

        self.logger.info(f"Seed-group metrics : {metrics_path}")
        self.logger.info(f"Seed-group report  : {report_path}")

        return report_path


class BackboneInferenceScheduler(InferenceScheduler):

    def _stamp_group_subdir(self) -> None:
        if not self.config.inference.output_subdir:
            self.config.inference.output_subdir = RunTag.now()

    def _write_group_report(self, results) -> None:
        with Logger(log_dir=str(self.work_dir), name="seed_group") as logger:
            failed = [result for result in results if result.status != "DONE"]
            if failed:
                logger.error(f"Seed-group report skipped: {len(failed)} of {len(results)} seed inferences failed")
                return

            SeedGroupReport(self.run_dirs, self.config.inference, logger).run()

    def run(self):
        if not self.config.seed_group:
            return super().run()

        self._stamp_group_subdir()
        results = super().run()
        self._write_group_report(results)

        return results
