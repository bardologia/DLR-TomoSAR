from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path

from _bootstrap import EnvironmentPinner

from configuration.param_extraction import ExtractParamsEntryConfig
from tools.runtime.config_cli                    import ConfigCli
from tools.monitoring.logger                     import Logger


def main() -> None:
    config = ConfigCli(ExtractParamsEntryConfig(), description="Gaussian parameter extraction sweep over datasets, K values, lambda values, and fit modes").apply()

    EnvironmentPinner.gpus(config.gpu_device_ids)

    from pipelines.processing.param_extraction.pipeline import DatasetQueueResolver, ExtractionPlanResolver, ParamExtractionPipeline

    logger       = Logger(log_dir="logs", name="extract_params")
    base_path    = Path(config.dataset_base_path)
    dataset_dirs = DatasetQueueResolver(base_path, config.dataset_filter).resolve()
    plans        = ExtractionPlanResolver(config, dataset_dirs).resolve()

    logger.section("Extraction queue")
    logger.kv_table({
        "Datasets"      : len(dataset_dirs),
        "Queue"         : ", ".join(d.name for d in dataset_dirs),
        "K values"      : config.fit_k_values,
        "Lambda values" : config.fit_lambda_values,
        "Fit modes"     : config.fit_modes,
        "Permutations"  : len(plans),
        "Base path"     : str(base_path),
        "Filter"        : config.dataset_filter or "all dataset directories",
        "GPUs"          : config.gpu_device_ids,
    }, title="Configuration")

    for index, extraction_config in enumerate(plans):
        logger.section(f"[Run {index + 1}/{len(plans)}] {extraction_config.processed_data_path.name} :: {extraction_config.output_subdir_name}")

        pipeline = ParamExtractionPipeline(extraction_config)
        outputs  = pipeline.run()

        logger.kv_table({name: str(path) for name, path in outputs.items()}, title="Outputs")

    logger.close()


if __name__ == "__main__":
    main()
