from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.gpu()

    from configuration.inference.inference_config import InferenceEntryConfig
    from pipelines.inference_pipeline.pipeline import InferencePipeline
    from tools.runtime.config_cli import ConfigCli
    from tools.monitoring.logger import Logger

    config   = ConfigCli(InferenceEntryConfig(), description="Inference over one or more run directories; backbone vs JEPA runs are auto-detected").apply()
    logs_dir = Path(config.logs_dir)

    run_dirs = sorted(
        [d for d in logs_dir.iterdir() if d.is_dir()]
        if not config.run_filter
        else [logs_dir / name for name in config.run_filter]
    )

    with Logger(log_dir=str(logs_dir), name="inference") as logger:
        logger.section("Inference")
        logger.kv_table({
            "Runs"     : len(run_dirs),
            "Logs dir" : str(logs_dir),
            "Filter"   : config.run_filter or "all run directories",
        }, title="Configuration")

        for run_dir in run_dirs:
            logger.subsection(run_dir.name)

            pipeline    = InferencePipeline(replace(config.inference, run_directory=run_dir, output_subdir=None))
            report_path = pipeline.run()

            logger.info(f"{run_dir.name}  :  {report_path}")


if __name__ == "__main__":
    main()
