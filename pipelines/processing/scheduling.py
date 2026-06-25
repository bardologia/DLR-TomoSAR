from __future__ import annotations

import math
import os
from typing import Tuple

from configuration.sar.processing_config import ParallelConfig


class ParallelPlanner:
    def __init__(self, config: ParallelConfig) -> None:
        self.config = config

    def available_cores(self) -> int:
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 1

    def core_budget(self) -> int:
        if self.config.effort not in self.config.EFFORT_FRACTIONS:
            raise ValueError(f"Unknown effort '{self.config.effort}', expected one of {sorted(self.config.EFFORT_FRACTIONS)}")
        return max(1, int(self.available_cores() * self.config.EFFORT_FRACTIONS[self.config.effort]))

    def interferogram_threads(self) -> int:
        if self.config.pyrat_threads is not None:
            return max(1, self.config.pyrat_threads)
        return self.core_budget()

    def resolve_plan(self, subsection_count: int) -> Tuple[int, int]:
        budget = self.core_budget()

        if self.config.tomogram_workers is not None and self.config.pyrat_threads is not None:
            return max(1, min(subsection_count, self.config.tomogram_workers)), max(1, self.config.pyrat_threads)

        if self.config.tomogram_workers is not None:
            workers = max(1, min(subsection_count, self.config.tomogram_workers))
            return workers, max(1, min(self.config.THREAD_CAP, budget // workers))

        if self.config.pyrat_threads is not None:
            threads = max(1, self.config.pyrat_threads)
            return max(1, min(subsection_count, budget // threads)), threads

        best_plan  = None
        best_waves = None

        for workers in range(1, min(subsection_count, budget) + 1):
            threads = max(1, min(self.config.THREAD_CAP, budget // workers))
            waves   = math.ceil(subsection_count / workers)

            if best_plan is None or waves < best_waves or (waves == best_waves and threads > best_plan[1]):
                best_plan, best_waves = (workers, threads), waves

        return best_plan
