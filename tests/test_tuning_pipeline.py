from __future__ import annotations

import json
import signal
from pathlib import Path

import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import TrialState

from configuration.tuning_config import TuningEntryConfig
from pipelines.tuning_pipeline.pipeline import TuningOrchestrator
from pipelines.tuning_pipeline.plots import StudyPlotter
from pipelines.tuning_pipeline.tuners import BestConfigWriter, ParamSampler


SPACE = {
    "encoder_lr" : {"type": "float",               "low": 1e-5, "high": 1e-2, "log": True},
    "activation" : {"type": "categorical",         "choices": ["relu", "gelu"]},
    "features"   : {"type": "indexed_categorical", "choices": [[32, 64], [64, 128]]},
}


class TrialFactory:
    @staticmethod
    def completed(value: float, lr: float, idx: int, intermediate: dict | None = None):
        return optuna.trial.create_trial(
            params        = {"encoder_lr": lr, "activation": ["relu", "gelu"][idx], "features__idx": idx},
            distributions = {
                "encoder_lr"    : FloatDistribution(1e-5, 1e-2, log=True),
                "activation"    : CategoricalDistribution(["relu", "gelu"]),
                "features__idx" : CategoricalDistribution([0, 1]),
            },
            value               = value,
            intermediate_values = intermediate or {},
        )


class RecordingLogger:
    def __init__(self) -> None:
        self.infos    = []
        self.warnings = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class StubProc:
    def __init__(self, alive: bool = True) -> None:
        self.alive      = alive
        self.terminated = False
        self.killed     = False

    def poll(self):
        return None if self.alive else 0

    def terminate(self) -> None:
        self.terminated = True
        self.alive      = False

    def wait(self, timeout=None) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


class TestParamSampler:
    def test_sample_resolves_indexed_categorical(self):
        trial   = optuna.trial.FixedTrial({"encoder_lr": 1e-3, "activation": "gelu", "features__idx": 1})
        sampled = ParamSampler().sample(trial, SPACE)

        assert sampled["encoder_lr"] == 1e-3
        assert sampled["activation"] == "gelu"
        assert sampled["features"]   == [64, 128]

    def test_decode_inverts_raw_study_params(self):
        raw     = {"encoder_lr": 1e-3, "activation": "gelu", "features__idx": 0}
        decoded = ParamSampler().decode(raw, SPACE)

        assert decoded == {"encoder_lr": 1e-3, "activation": "gelu", "features": [32, 64]}

    def test_decode_drops_unknown_index_keys(self):
        decoded = ParamSampler().decode({"ghost__idx": 0}, SPACE)

        assert decoded == {}


class TestBestConfigWriter:
    def _completed_trial(self, value: float, lr: float, idx: int):
        return TrialFactory.completed(value, lr, idx)

    def test_write_without_completed_trials_returns_none(self, tmp_path):
        study  = optuna.create_study(direction="minimize")
        writer = BestConfigWriter("unet", SPACE, tmp_path / "best_config.json")

        assert writer.write(study) is None
        assert not (tmp_path / "best_config.json").exists()

    def test_write_persists_decoded_best(self, tmp_path):
        study = optuna.create_study(direction="minimize")
        study.add_trial(self._completed_trial(0.5, 1e-3, 0))
        study.add_trial(self._completed_trial(0.2, 5e-4, 1))

        writer  = BestConfigWriter("unet", SPACE, tmp_path / "best_config.json")
        payload = writer.write(study)

        saved = json.loads((tmp_path / "best_config.json").read_text())

        assert payload["val_loss"]          == 0.2
        assert saved["model"]               == "unet"
        assert saved["params"]["features"]  == [64, 128]
        assert "features__idx" not in saved["params"]

    def test_write_tracks_improvement_across_chunks(self, tmp_path):
        study  = optuna.create_study(direction="minimize")
        writer = BestConfigWriter("unet", SPACE, tmp_path / "best_config.json")

        study.add_trial(self._completed_trial(0.5, 1e-3, 0))
        writer.write(study)
        study.add_trial(self._completed_trial(0.1, 2e-4, 1))
        writer.write(study)

        saved = json.loads((tmp_path / "best_config.json").read_text())

        assert saved["val_loss"] == 0.1
        assert saved["trial"]    == 1


class TestTuningOrchestrator:
    def _orchestrator(self, tmp_path):
        config = TuningEntryConfig()
        config.paths.log_base_dir = tmp_path
        return TuningOrchestrator(tag="t", config=config, entry_script=tmp_path / "tune.py")

    def test_distribute_trials_sums_and_balances(self, tmp_path):
        orch   = self._orchestrator(tmp_path)
        counts = orch._distribute_trials(10, 4)

        assert sum(counts) == 10
        assert max(counts) - min(counts) <= 1

    def test_count_done_ignores_failed_and_running(self, tmp_path):
        orch  = self._orchestrator(tmp_path)
        study = optuna.create_study(direction="minimize")

        study.add_trial(optuna.trial.create_trial(state=TrialState.COMPLETE, value=0.3))
        study.add_trial(optuna.trial.create_trial(state=TrialState.PRUNED))
        study.add_trial(optuna.trial.create_trial(state=TrialState.FAIL))
        study.add_trial(optuna.trial.create_trial(state=TrialState.RUNNING))

        assert orch._count_done(study) == 2

    def test_fail_stale_trials_marks_running_as_failed(self, tmp_path):
        orch  = self._orchestrator(tmp_path)
        study = optuna.create_study(direction="minimize")

        study.add_trial(optuna.trial.create_trial(state=TrialState.RUNNING))
        study.add_trial(optuna.trial.create_trial(state=TrialState.COMPLETE, value=0.3))

        assert orch._fail_stale_trials(study) == 1
        assert orch._count_done(study)        == 1
        assert len(study.get_trials(states=(TrialState.RUNNING,))) == 0

    def test_terminate_workers_stops_live_procs_and_exits(self, tmp_path):
        orch = self._orchestrator(tmp_path)
        live = StubProc(alive=True)
        done = StubProc(alive=False)

        orch.active_procs = [(live, 0, tmp_path / "a.log", None), (done, 1, tmp_path / "b.log", None)]

        with pytest.raises(SystemExit) as excinfo:
            orch._terminate_workers(signal.SIGTERM, None)

        assert live.terminated
        assert not done.terminated
        assert excinfo.value.code == 128 + signal.SIGTERM


class TestStudyPlotter:
    def _study(self, n_trials: int = 8):
        study = optuna.create_study(direction="minimize")
        for i in range(n_trials):
            value = 1.0 / (i + 1)
            study.add_trial(TrialFactory.completed(value, 1e-4 * (i + 1), i % 2, intermediate={0: 1.0, 1: value}))
        return study

    def test_render_writes_core_plots(self, tmp_path):
        plotter  = StudyPlotter(RecordingLogger())
        saved    = plotter.render(self._study(), tmp_path)
        relative = {str(p.relative_to(tmp_path)) for p in saved}

        assert "optimization_history.png" in relative
        assert "param_importances.png"    in relative
        assert all(p.exists() for p in saved)

    def test_render_writes_one_image_per_param_and_pair(self, tmp_path):
        plotter  = StudyPlotter(RecordingLogger())
        study    = self._study()
        saved    = plotter.render(study, tmp_path)
        relative = {str(p.relative_to(tmp_path)) for p in saved}
        params   = sorted({p for t in study.trials for p in t.params})

        for param in params:
            assert f"slice/{param}.png" in relative
            assert f"rank/{param}.png"  in relative

        for i, p1 in enumerate(params):
            for p2 in params[i + 1 :]:
                assert f"contour/{p1}__{p2}.png" in relative

    def test_render_on_empty_study_does_not_raise(self, tmp_path):
        logger  = RecordingLogger()
        plotter = StudyPlotter(logger)
        saved   = plotter.render(optuna.create_study(direction="minimize"), tmp_path)

        assert isinstance(saved, list)

    def test_contour_params_capped_by_importance(self, tmp_path):
        logger  = RecordingLogger()
        plotter = StudyPlotter(logger)

        plotter.CONTOUR_MAX_PARAMS = 1
        top = plotter._contour_params(self._study())

        assert len(top) == 1
        assert len(logger.infos) == 1

    def test_contour_params_uncapped_returns_all(self, tmp_path):
        plotter = StudyPlotter(RecordingLogger())
        study   = self._study()
        params  = sorted({p for t in study.trials for p in t.params})

        assert plotter._contour_params(study) == params
