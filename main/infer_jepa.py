from __future__ import annotations

from dataclasses import replace
from pathlib     import Path

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.gpu()

    from configuration.inference_config       import InferenceEntryConfig
    from pipelines.jepa_pipeline.inference     import JepaInferencePipeline
    from tools.config_cli                      import ConfigCli
    from tools.logger                          import Logger

    config   = ConfigCli(InferenceEntryConfig(), description="JEPA inference: decode predicted embeddings then stitch/metrics/report").apply()
    logs_dir = Path(config.logs_dir)

    all_dirs = sorted([d for d in logs_dir.iterdir() if d.is_dir()])
    run_dirs = [d for d in all_dirs if d.name in config.run_filter] if config.run_filter else all_dirs

    with Logger(log_dir=str(logs_dir), name="jepa_inference") as logger:
        for run_dir in run_dirs:
            if not (run_dir / "meta" / "autoencoder_config.json").is_file():
                continue
            report_path = JepaInferencePipeline(replace(config.inference, run_directory=run_dir, output_subdir=None), run_dir).run()
            logger.info(f"{run_dir.name} : {report_path}")


if __name__ == "__main__":
    main()
