from __future__ import annotations

from itertools import combinations
from pathlib   import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as ovm

from pipelines.shared.plotting import PlotBase


class StudyPlotter(PlotBase):
    save_dpi = 300

    CONTOUR_MAX_PARAMS = 8

    def __init__(self, logger) -> None:
        self.logger = logger

    def render(self, study: optuna.Study, out_dir: Path) -> list[Path]:
        self._apply_style()

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved  = []
        saved += self._render_single(study, out_dir)
        saved += self._render_per_param(study, out_dir)
        saved += self._render_contours(study, out_dir)

        return saved

    def _render_single(self, study: optuna.Study, out_dir: Path) -> list[Path]:
        saved = []
        for name, plotter in self._plotters(study).items():
            try:
                axes = plotter()
                saved.append(self._save(axes.figure, out_dir / f"{name}.png"))
            except Exception as exc:
                self.logger.warning(f"Study plot '{name}' skipped: {exc}")
                plt.close("all")

        return saved

    def _render_per_param(self, study: optuna.Study, out_dir: Path) -> list[Path]:
        params = self._study_params(study)
        saved  = []

        for name, plotter in (("slice", ovm.plot_slice), ("rank", ovm.plot_rank)):
            for param in params:
                try:
                    axes = plotter(study, params=[param])
                    saved.append(self._save(axes.figure, out_dir / name / f"{param}.png"))
                except Exception as exc:
                    self.logger.warning(f"Study plot '{name}/{param}' skipped: {exc}")
                    plt.close("all")

        return saved

    def _render_contours(self, study: optuna.Study, out_dir: Path) -> list[Path]:
        params = self._contour_params(study)
        saved  = []

        for p1, p2 in combinations(params, 2):
            try:
                axes = ovm.plot_contour(study, params=[p1, p2])
                saved.append(self._save(axes.figure, out_dir / "contour" / f"{p1}__{p2}.png"))
            except Exception as exc:
                self.logger.warning(f"Study plot 'contour/{p1}__{p2}' skipped: {exc}")
                plt.close("all")

        return saved

    def _plotters(self, study: optuna.Study) -> dict:
        return {
            "optimization_history" : lambda: ovm.plot_optimization_history(study),
            "intermediate_values"  : lambda: ovm.plot_intermediate_values(study),
            "parallel_coordinate"  : lambda: ovm.plot_parallel_coordinate(study),
            "param_importances"    : lambda: ovm.plot_param_importances(study),
            "duration_importances" : lambda: ovm.plot_param_importances(study, target=lambda t: t.duration.total_seconds(), target_name="duration"),
            "edf"                  : lambda: ovm.plot_edf(study),
            "timeline"             : lambda: ovm.plot_timeline(study),
        }

    @staticmethod
    def _study_params(study: optuna.Study) -> list[str]:
        return sorted({p for t in study.trials for p in t.params})

    def _contour_params(self, study: optuna.Study) -> list[str]:
        params = self._study_params(study)
        if len(params) <= self.CONTOUR_MAX_PARAMS:
            return params

        importances = optuna.importance.get_param_importances(study)
        top         = list(importances)[: self.CONTOUR_MAX_PARAMS]
        self.logger.info(f"Contour plots limited to top {len(top)} of {len(params)} parameters by importance")
        return sorted(top)
