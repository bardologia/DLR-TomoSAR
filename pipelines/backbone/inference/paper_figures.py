from __future__ import annotations

import shutil
import sys
from pathlib import Path

from configuration.diagnostics                    import PaperFigurePackConfig
from pipelines.backbone.inference.seed_comparison import SeedInferenceResolver
from tools.data.io                                import FileIO
from tools.runtime.run_selector                   import ReportRunSelector


class PaperFigurePack:

    MANIFEST = "figure_manifest.json"

    def __init__(self, config: PaperFigurePackConfig, logger) -> None:
        self.config = config
        self.logger = logger

        self.output_dir = Path(config.output_dir)

    def _select_runs(self) -> list[Path]:
        selector = ReportRunSelector(self.config.runs_dir, "inference", self.config.report_filename, self.logger)

        if self.config.run_filter:
            return selector.filter(self.config.run_filter)
        if sys.stdin.isatty():
            return selector.select()
        return selector.all()

    @staticmethod
    def _stable_name(run_dir: Path, figures_dir: Path, figure: Path) -> str:
        relative = figure.relative_to(figures_dir)
        return f"{run_dir.name}__{'__'.join(relative.with_suffix('').parts)}{figure.suffix}"

    def _collect_run(self, run_dir: Path, resolver: SeedInferenceResolver) -> list[dict]:
        inference_dir = resolver.resolve(run_dir)
        figures_dir   = inference_dir / self.config.figures_subdir

        if not figures_dir.is_dir():
            raise FileNotFoundError(f"{figures_dir} is missing; the inference saved no figures (save_plots disabled?)")

        entries = []
        for pattern in self.config.patterns:
            for figure in sorted(figures_dir.glob(pattern)):
                target = self.output_dir / self._stable_name(run_dir, figures_dir, figure)
                shutil.copy2(figure, target)
                entries.append({"run": run_dir.name, "source": str(figure), "target": target.name})

        if not entries:
            raise FileNotFoundError(f"No figure under {figures_dir} matches {self.config.patterns}; adjust patterns or re-run inference with save_plots")

        return entries

    def run(self) -> dict:
        FileIO.ensure_dirs(self.output_dir)

        resolver = SeedInferenceResolver(self.config.inference_subdir, self.config.metrics_filename)
        manifest = []

        for run_dir in self._select_runs():
            self.logger.subsection(f"Run: {run_dir}")
            entries   = self._collect_run(run_dir, resolver)
            manifest += entries
            self.logger.ok(f"{run_dir.name}: {len(entries)} figures packed")

        payload = {
            "output_dir" : str(self.output_dir),
            "patterns"   : list(self.config.patterns),
            "note"       : "Figures are copied as rendered by each run's inference; re-run inference with figure_style=paper for publication styling before packing.",
            "figures"    : manifest,
        }
        FileIO.save_json(payload, self.output_dir / self.MANIFEST)

        self.logger.ok(f"Figure pack: {len(manifest)} figures -> {self.output_dir}")

        return payload
