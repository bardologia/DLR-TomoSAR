from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.gpu()

    from configuration.inference_config import SingleInferenceConfig
    from pipelines.inference_pipeline.pipeline import InferencePipeline
    from tools.config_cli import ConfigCli
    from tools.logger import Logger

    config = ConfigCli(SingleInferenceConfig(), description="Single inference run").apply()

    with Logger(log_dir=str(Path(config.run_directory) / "logs"), name="single_infer") as logger:
        pipeline    = InferencePipeline(replace(config.inference, run_directory=Path(config.run_directory), output_subdir=None))
        report_path = pipeline.run()

        logger.info(f"Inference report written to: {report_path}")


if __name__ == "__main__":
    main()
