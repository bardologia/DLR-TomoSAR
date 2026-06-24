from __future__ import annotations

from dataclasses import dataclass, field

from pipelines.shared.seed_sweep import SeedSweepRunner


@dataclass
class _Config:
    run_name : str | None = None
    seed     : int        = 0
    seeds    : list[int]  = field(default_factory=list)


def _factory(record):
    class _Runner:
        def __init__(self, config):
            self.config = config
        def run(self):
            record.append((self.config.seed, self.config.run_name))
            return self.config.seed
    return _Runner


def test_empty_seeds_runs_once_with_base_seed_and_no_renaming():
    record = []

    result = SeedSweepRunner(_Config(seed=3), _factory(record)).run()

    assert record == [(3, None)]
    assert result == 3


def test_empty_seeds_preserves_explicit_run_name():
    record = []

    SeedSweepRunner(_Config(run_name="exp", seed=7), _factory(record)).run()

    assert record == [(7, "exp")]


def test_single_explicit_seed_runs_once_with_that_seed():
    record = []

    SeedSweepRunner(_Config(run_name="exp", seed=0, seeds=[5]), _factory(record)).run()

    assert record == [(5, "exp")]


def test_explicit_seeds_run_in_listed_order():
    record = []

    SeedSweepRunner(_Config(run_name="exp", seeds=[3, 0, 17]), _factory(record)).run()

    assert [seed for seed, _ in record] == [3, 0, 17]


def test_explicit_seeds_suffix_run_name_per_seed():
    record = []

    SeedSweepRunner(_Config(run_name="exp", seeds=[3, 0, 17]), _factory(record)).run()

    assert [name for _, name in record] == ["exp_seed3", "exp_seed0", "exp_seed17"]


def test_explicit_seeds_return_results_keyed_by_seed():
    record = []

    result = SeedSweepRunner(_Config(seeds=[4, 5]), _factory(record)).run()

    assert result == {4: 4, 5: 5}


def test_explicit_seeds_without_run_name_group_under_shared_base():
    record = []

    SeedSweepRunner(_Config(seeds=[0, 1]), _factory(record)).run()

    bases = {name.rsplit("_seed", 1)[0] for _, name in record}
    assert len(bases) == 1
    assert sorted(name.rsplit("_seed", 1)[1] for _, name in record) == ["0", "1"]
