from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np
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

        saved = []
        for name, plotter in self._plotters(study).items():
            try:
                axes = plotter()
                fig  = self._figure_of(axes)
                saved.append(self._save(fig, out_dir / f"{name}.png"))
            except Exception as exc:
                self.logger.warning(f"Study plot '{name}' skipped: {exc}")
                plt.close("all")

        return saved

    def _plotters(self, study: optuna.Study) -> dict:
        return {
            "optimization_history" : lambda: ovm.plot_optimization_history(study),
            "intermediate_values"  : lambda: ovm.plot_intermediate_values(study),
            "parallel_coordinate"  : lambda: ovm.plot_parallel_coordinate(study),
            "slice"                : lambda: ovm.plot_slice(study),
            "contour"              : lambda: ovm.plot_contour(study, params=self._contour_params(study)),
            "param_importances"    : lambda: ovm.plot_param_importances(study),
            "duration_importances" : lambda: ovm.plot_param_importances(study, target=lambda t: t.duration.total_seconds(), target_name="duration"),
            "edf"                  : lambda: ovm.plot_edf(study),
            "rank"                 : lambda: ovm.plot_rank(study),
            "timeline"             : lambda: ovm.plot_timeline(study),
        }

    def _contour_params(self, study: optuna.Study) -> list[str] | None:
        params = sorted({p for t in study.trials for p in t.params})
        if len(params) <= self.CONTOUR_MAX_PARAMS:
            return None

        importances = optuna.importance.get_param_importances(study)
        top         = list(importances)[: self.CONTOUR_MAX_PARAMS]
        self.logger.info(f"Contour plot limited to top {len(top)} of {len(params)} parameters by importance")
        return top

    @staticmethod
    def _figure_of(axes):
        if isinstance(axes, np.ndarray):
            return axes.flat[0].figure
        return axes.figure
