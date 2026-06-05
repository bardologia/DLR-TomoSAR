from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.gpu()

    from configuration.inference_config import BatchInferenceConfig
    from pipelines.inference_pipeline.pipeline import InferencePipeline
    from tools.config_cli import ConfigCli
    from tools.logger import Logger

    config   = ConfigCli(BatchInferenceConfig(), description="Batch inference over run directories").apply()
    logs_dir = Path(config.logs_dir)

    run_dirs = sorted(
        [d for d in logs_dir.iterdir() if d.is_dir()]
        if not config.run_filter
        else [logs_dir / name for name in config.run_filter]
    )

    with Logger(log_dir=str(logs_dir), name="batch_inference") as logger:
        logger.section("Batch inference")
        logger.kv_table({
            "Runs"     : len(run_dirs),
            "Logs dir" : str(logs_dir),
        }, title="Configuration")

        for run_dir in run_dirs:
            logger.subsection(run_dir.name)

            pipeline    = InferencePipeline(replace(config.inference, run_directory=run_dir, output_subdir=None))
            report_path = pipeline.run()

            logger.info(f"{run_dir.name}  :  {report_path}")


if __name__ == "__main__":
    main()
