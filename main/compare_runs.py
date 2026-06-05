from __future__ import annotations

import os
os.environ["MKL_NUM_THREADS"]     = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"]     = "4"

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from configuration.benchmark_config                import BenchmarkConfig
from pipelines.benchmark_pipeline.comparison_stage import ComparisonStage
from tools.config_cli                              import ConfigCli
from tools.logger                                  import Logger


def _resolve_run_tag(config: BenchmarkConfig) -> str:
    if config.run_tag is not None:
        return config.run_tag

    base       = Path(config.paths.log_base_dir)
    candidates = sorted(d.name for d in base.iterdir() if d.is_dir() and (d / "training").is_dir()) if base.is_dir() else []

    if not candidates:
        sys.exit(f"ERROR: no benchmark runs found under {base}")

    return candidates[-1]


def main() -> None:
    config = ConfigCli(BenchmarkConfig(), description="Standalone benchmark comparison").apply()
    tag    = _resolve_run_tag(config)

    logger = Logger(log_dir=str(Path(config.paths.log_base_dir) / tag / "pipeline"), name="compare_runs")

    stage   = ComparisonStage(config=config, run_tag=tag, logger=logger)
    out_dir = stage.run()

    logger.info(f"Comparison written to: {out_dir}")
    logger.close()


if __name__ == "__main__":
    main()
