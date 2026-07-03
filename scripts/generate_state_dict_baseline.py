from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.models_baseline.test_state_dict_baseline import BASELINE_PATH, MODEL_FAMILIES, StateDictSignature
from tools.monitoring.logger import Logger


class BaselineGenerator:
    def __init__(self):
        self.logger = Logger(log_dir="logs", name="state_dict_baseline")

    def run(self) -> None:
        baseline = {}

        for family, (registry, factory) in MODEL_FAMILIES.items():
            baseline[family] = {}

            for name in sorted(registry):
                model, _config          = factory(name)
                baseline[family][name]  = StateDictSignature.compute(model)
                self.logger.info(f"{family}/{name}: {baseline[family][name]['num_keys']} keys, {baseline[family][name]['num_parameters']} parameters")

        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
        self.logger.info(f"Baseline written to {BASELINE_PATH}")


if __name__ == "__main__":
    BaselineGenerator().run()
