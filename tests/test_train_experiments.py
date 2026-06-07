from __future__ import annotations

import pytest

from configuration.train_config import SecondaryTrialsConfig
from pipelines.training_pipeline.experiments import CurriculumTrialPlanner, SecondaryTrialPlanner, WarmupTrialPlanner

CANDIDATES = [f"PS{index:02d}" for index in range(1, 11)]


class TestCurriculumTrialPlanner:
    def test_cartesian_product(self):
        warmup   = {"a": {"use_param_l1": True, "weight_param_l1": 1.0}}
        complete = {"b": {"use_mse_curve": True}, "c": {"use_l1_curve": True}}
        plans    = CurriculumTrialPlanner("resunet", warmup, complete).plan()

        assert len(plans) == 2
        assert [name for name, _ in plans] == ["resunet_w-a_c-b", "resunet_w-a_c-c"]

        for _, overrides in plans:
            assert overrides["curriculum.enabled"] is True
            assert overrides["curriculum.warmup.use_param_l1"] is True

        assert plans[0][1]["curriculum.complete.use_mse_curve"] is True
        assert plans[1][1]["curriculum.complete.use_l1_curve"] is True


class TestWarmupTrialPlanner:
    def test_disables_curriculum_per_loss(self):
        warmup = {
            "pL11" : {"use_param_l1": True, "weight_param_l1": 1.0},
            "mse"  : {"use_mse_curve": True, "weight_mse_curve": 1.0},
        }
        plans = WarmupTrialPlanner("resunet", warmup).plan()

        assert len(plans) == 2
        assert [name for name, _ in plans] == ["resunet_nc-pL11", "resunet_nc-mse"]

        for _, overrides in plans:
            assert overrides["curriculum.enabled"] is False

        assert plans[0][1]["curriculum.warmup.weight_param_l1"] == 1.0
        assert plans[1][1]["curriculum.warmup.use_mse_curve"] is True


class TestSecondaryTrialPlanner:
    def make(self, **kwargs) -> SecondaryTrialPlanner:
        return SecondaryTrialPlanner("resunet", SecondaryTrialsConfig(**kwargs), CANDIDATES)

    def labels(self, plans) -> list:
        return [overrides["paths.secondary_labels"] for _, overrides in plans]

    def test_consecutive_blocks(self):
        plans = self.make(strategy="consecutive", n_secondaries=3, block_step=2).plan()

        assert self.labels(plans) == [
            ("PS01", "PS02", "PS03"),
            ("PS03", "PS04", "PS05"),
            ("PS05", "PS06", "PS07"),
            ("PS07", "PS08", "PS09"),
        ]

    def test_consecutive_full_coverage_step_one(self):
        plans = self.make(strategy="consecutive", n_secondaries=4, block_step=1).plan()

        assert len(plans) == 7
        assert self.labels(plans)[0]  == ("PS01", "PS02", "PS03", "PS04")
        assert self.labels(plans)[-1] == ("PS07", "PS08", "PS09", "PS10")

    def test_spaced_blocks(self):
        plans = self.make(strategy="spaced", n_secondaries=3, spacing=3, block_step=2).plan()

        assert self.labels(plans) == [
            ("PS01", "PS04", "PS07"),
            ("PS03", "PS06", "PS09"),
        ]

    def test_spaced_span_too_wide(self):
        with pytest.raises(ValueError, match="spans"):
            self.make(strategy="spaced", n_secondaries=4, spacing=4).plan()

    def test_uniform_deterministic_and_distinct(self):
        config = dict(strategy="uniform", n_secondaries=4, n_trials=6, seed=7)

        first  = self.labels(self.make(**config).plan())
        second = self.labels(self.make(**config).plan())

        assert first == second
        assert len(first) == 6
        assert len(set(first)) == 6

        for selection in first:
            assert len(selection) == 4
            assert len(set(selection)) == 4
            assert all(label in CANDIDATES for label in selection)
            assert list(selection) == sorted(selection)

    def test_uniform_different_seed_differs(self):
        first  = self.labels(self.make(strategy="uniform", n_secondaries=4, n_trials=6, seed=0).plan())
        second = self.labels(self.make(strategy="uniform", n_secondaries=4, n_trials=6, seed=1).plan())

        assert first != second

    def test_gaussian_deterministic_and_within_bounds(self):
        config = dict(strategy="gaussian", n_secondaries=3, n_trials=5, mean=4.5, sigma=2.0, seed=3)

        first  = self.labels(self.make(**config).plan())
        second = self.labels(self.make(**config).plan())

        assert first == second
        assert len(first) == 5
        assert len(set(first)) == 5

        for selection in first:
            assert len(set(selection)) == 3
            assert all(label in CANDIDATES for label in selection)

    def test_gaussian_requires_mean_and_sigma(self):
        with pytest.raises(ValueError, match="mean and sigma"):
            self.make(strategy="gaussian", n_secondaries=3, n_trials=5)

    def test_gaussian_exhaustion_raises(self):
        with pytest.raises(RuntimeError, match="distinct secondary sets"):
            self.make(strategy="gaussian", n_secondaries=3, n_trials=200, mean=4.5, sigma=0.5, seed=0).plan()

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            self.make(strategy="alternating")

    def test_too_many_secondaries(self):
        with pytest.raises(ValueError, match="n_secondaries"):
            self.make(strategy="consecutive", n_secondaries=11)

    def test_run_names_carry_strategy_and_labels(self):
        plans = self.make(strategy="consecutive", n_secondaries=2, block_step=4).plan()

        assert plans[0][0] == "resunet_sec-consecutive-t00_PS01-PS02"
        assert plans[1][0] == "resunet_sec-consecutive-t01_PS05-PS06"

    def test_overrides_are_tuples(self):
        plans = self.make(strategy="uniform", n_secondaries=3, n_trials=2, seed=0).plan()

        for _, overrides in plans:
            assert isinstance(overrides["paths.secondary_labels"], tuple)
