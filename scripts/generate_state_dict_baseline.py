from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import MODEL_REGISTRY, get_model
from tests.test_models import BASELINE_PATH, StateDictSignature
from tools.logger import Logger


class BaselineGenerator:
    def __init__(self):
        self.logger = Logger(name="state_dict_baseline")

    def run(self) -> None:
        baseline = {}

        for name in sorted(MODEL_REGISTRY.keys()):
            model, _       = get_model(name)
            baseline[name] = StateDictSignature.compute(model)
            self.logger.info(f"{name}: {baseline[name]['num_keys']} keys, {baseline[name]['num_parameters']} parameters")

        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
        self.logger.ok(f"Baseline written to {BASELINE_PATH}")


if __name__ == "__main__":
    BaselineGenerator().run()
