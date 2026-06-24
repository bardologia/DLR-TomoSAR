from __future__ import annotations

from dataclasses import dataclass

from pipelines.shared.seed_sweep import SeedSweepRunner


@dataclass
class _Config:
    run_name : str | None = None
    seed     : int        = 0
    n_seeds  : int        = 1


def _factory(record):
    class _Runner:
        def __init__(self, config):
            self.config = config
        def run(self):
            record.append((self.config.seed, self.config.run_name))
            return self.config.seed
    return _Runner


def test_single_seed_runs_once_without_renaming():
    record = []

    result = SeedSweepRunner(_Config(), _factory(record)).run()

    assert record == [(0, None)]
    assert result == 0


def test_single_seed_preserves_explicit_run_name():
    record = []

    SeedSweepRunner(_Config(run_name="exp", seed=7), _factory(record)).run()

    assert record == [(7, "exp")]


def test_multi_seed_offsets_from_base_seed():
    record = []

    SeedSweepRunner(_Config(run_name="exp", seed=10, n_seeds=3), _factory(record)).run()

    assert [seed for seed, _ in record] == [10, 11, 12]


def test_multi_seed_suffixes_run_name_per_seed():
    record = []

    SeedSweepRunner(_Config(run_name="exp", seed=10, n_seeds=3), _factory(record)).run()

    assert [name for _, name in record] == ["exp_seed10", "exp_seed11", "exp_seed12"]


def test_multi_seed_returns_results_keyed_by_seed():
    record = []

    result = SeedSweepRunner(_Config(seed=4, n_seeds=2), _factory(record)).run()

    assert result == {4: 4, 5: 5}


def test_multi_seed_without_run_name_groups_under_shared_base():
    record = []

    SeedSweepRunner(_Config(seed=0, n_seeds=2), _factory(record)).run()

    bases = {name.rsplit("_seed", 1)[0] for _, name in record}
    assert len(bases) == 1
    assert [name.endswith("_seed0") or name.endswith("_seed1") for _, name in record] == [True, True]
