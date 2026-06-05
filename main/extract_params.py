from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning)

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from configuration.param_extraction_config import ExtractParamsEntryConfig, ExtractionConfig, FitMode, FitSettings
from pipelines.param_pipeline.pipeline import ParamExtractionPipeline
from tools.config_cli import ConfigCli
from tools.logger import Logger


def main() -> None:
    config = ConfigCli(ExtractParamsEntryConfig(), description="Gaussian parameter extraction sweep").apply()
    logger = Logger(log_dir="logs", name="extract_params")

    dataset_dirs = sorted(p for p in Path(config.dataset_base_path).iterdir() if p.is_dir())

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
