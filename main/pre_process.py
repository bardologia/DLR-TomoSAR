from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from configuration.processing_config import (
    CropRegion,
    PathConfiguration,
    PreProcessEntryConfig,
    ProcessingConfiguration,
    TomogramConfiguration,
)
from pipelines.processing_pipeline.pipeline import ProcessingPipeline
from tools.config_cli import ConfigCli
from tools.logger import Logger


def main() -> None:
    config = ConfigCli(PreProcessEntryConfig(), description="SAR pre-processing sweep").apply()
    logger = Logger(log_dir="logs", name="pre_process")

    global_crop            = CropRegion(azimuth_start=config.azimuth_start, azimuth_end=config.azimuth_end, range_start=config.range_start, range_end=config.range_end)
    max_crop_azimuth_width = (global_crop.azimuth_end - global_crop.azimuth_start) // 16

    for index, filter_arguments in enumerate(config.filter_arguments_list):
        win          = filter_arguments.get("win", [])
        win_str      = "_".join(str(w) for w in win)
        dataset_name = f"base_dataset_w{win_str}"

        logger.section(f"[Run {index + 1}/{len(config.filter_arguments_list)}] {dataset_name}")
        logger.kv_table({"Filter arguments": str(filter_arguments)})

        shared_tomo = dict(
            fusar_project_path     = config.fusar_project_path,
            base_directory         = config.base_directory,
            track_selection        = config.track_selection,
            polarisation           = config.polarisation,
            beamforming_method     = config.beamforming_method,
            filter_method          = config.filter_method,
            filter_arguments       = filter_arguments,
            max_crop_azimuth_width = max_crop_azimuth_width,
        )

        processing_config = ProcessingConfiguration(
            crop = global_crop,

            input_configs  = TomogramConfiguration(**shared_tomo),
            output_configs = TomogramConfiguration(**shared_tomo, height_range=config.height_range),

            paths                    = PathConfiguration(run_subdirectory=dataset_name),
            dataset_type             = config.dataset_type,
            full_stack_identifier    = config.full_stack_identifier,
            reduced_stack_identifier = config.reduced_stack_identifier,
            tomogram_output_tag      = config.tomogram_output_tag,
            parameter_output_tag     = config.parameter_output_tag,
        )

        pipeline = ProcessingPipeline(processing_config)
        outputs  = pipeline.run()

        logger.kv_table({name: str(path) for name, path in outputs.items()}, title="Outputs")

    logger.close()


if __name__ == "__main__":
    main()
