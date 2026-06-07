from __future__ import annotations

import json
from pathlib import Path

import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import TrialState

from configuration.tuning_config import TuningEntryConfig
from pipelines.tuning_pipeline.pipeline import TuningOrchestrator
from pipelines.tuning_pipeline.tuners import BestConfigWriter, ParamSampler


SPACE = {
    "encoder_lr" : {"type": "float",               "low": 1e-5, "high": 1e-2, "log": True},
    "activation" : {"type": "categorical",         "choices": ["relu", "gelu"]},
    "features"   : {"type": "indexed_categorical", "choices": [[32, 64], [64, 128]]},
}


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
        return optuna.trial.create_trial(
            params        = {"encoder_lr": lr, "activation": "gelu", "features__idx": idx},
            distributions = {
                "encoder_lr"    : FloatDistribution(1e-5, 1e-2, log=True),
                "activation"    : CategoricalDistribution(["relu", "gelu"]),
                "features__idx" : CategoricalDistribution([0, 1]),
            },
            value = value,
        )

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
