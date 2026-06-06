from __future__ import annotations

from pathlib import Path

import _bootstrap

from configuration.param_extraction_config import ExtractParamsEntryConfig, ExtractionConfig, FitMode, FitSettings
from pipelines.param_pipeline.pipeline import ParamExtractionPipeline
from tools.config_cli import ConfigCli
from tools.logger import Logger


def main() -> None:
    config    = ConfigCli(ExtractParamsEntryConfig(), description="Gaussian parameter extraction over one or more dataset directories").apply()
    logger    = Logger(log_dir="logs", name="extract_params")
    base_path = Path(config.dataset_base_path)

    dataset_dirs = sorted(
        [d for d in base_path.iterdir() if d.is_dir()]
        if not config.dataset_filter
        else [base_path / name for name in config.dataset_filter]
    )

    logger.section("Extraction queue")
    logger.kv_table({
        "Datasets" : len(dataset_dirs),
        "Base path": str(base_path),
        "Filter"   : config.dataset_filter or "all dataset directories",
    }, title="Configuration")

    for index, processed_data_path in enumerate(dataset_dirs):
        logger.section(f"[Run {index + 1}/{len(dataset_dirs)}] {processed_data_path.name}")

        extraction_config = ExtractionConfig(
            processed_data_path = processed_data_path,
            pyrat_directory     = config.pyrat_directory,

            output_prefix = config.output_prefix,
            output_suffix = config.output_suffix,

            tomogram_filename = config.tomogram_filename,
            height_range      = config.height_range,

            fit_settings = FitSettings(fit_config=FitMode.SigmaOnly(k_max=config.fit_k_max, lambda_k=config.fit_lambda_k)),

            parameter_workers = config.parameter_workers,
        )

        pipeline = ParamExtractionPipeline(extraction_config)
        outputs  = pipeline.run()

        logger.kv_table({name: str(path) for name, path in outputs.items()}, title="Outputs")

    logger.close()


if __name__ == "__main__":
    main()
