from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path

from _bootstrap import EnvironmentPinner

from configuration.param_extraction import ExtractParamsEntryConfig, ExtractionConfig, FitMode, FitSettings
from tools.runtime.config_cli                    import ConfigCli
from tools.monitoring.logger                     import Logger


def main() -> None:
    config = ConfigCli(ExtractParamsEntryConfig(), description="Gaussian parameter extraction over one or more dataset directories").apply()

    EnvironmentPinner.gpus(config.gpu_device_ids)

    from pipelines.processing.param_extraction.pipeline import DatasetQueueResolver, ParamExtractionPipeline

    logger       = Logger(log_dir="logs", name="extract_params")
    base_path    = Path(config.dataset_base_path)
    dataset_dirs = DatasetQueueResolver(base_path, config.dataset_filter).resolve()

    logger.section("Extraction queue")
    logger.kv_table({
        "Datasets" : len(dataset_dirs),
        "Queue"    : ", ".join(d.name for d in dataset_dirs),
        "Base path": str(base_path),
        "Filter"   : config.dataset_filter or "all dataset directories",
        "GPUs"     : config.gpu_device_ids,
    }, title="Configuration")

    for index, processed_data_path in enumerate(dataset_dirs):
        logger.section(f"[Run {index + 1}/{len(dataset_dirs)}] {processed_data_path.name}")

        extraction_config = ExtractionConfig(
            processed_data_path = processed_data_path,
            pyrat_directory     = config.pyrat_directory,

            output_prefix = config.output_prefix,
            output_suffix = config.output_suffix,

            height_range = config.height_range,

            fit_settings = FitSettings(fit_config=FitMode.SigmaOnly(k_max=config.fit_k_max, lambda_k=config.fit_lambda_k, sigma_init_divisor=config.fit_sigma_init_divisor, fit_amplitude=config.fit_amplitude, fit_mean=config.fit_mean)),

            range_batch_size  = config.range_batch_size,
            parameter_workers = config.parameter_workers,
        )

        pipeline = ParamExtractionPipeline(extraction_config)
        outputs  = pipeline.run()

        logger.kv_table({name: str(path) for name, path in outputs.items()}, title="Outputs")

    logger.close()


if __name__ == "__main__":
    main()
