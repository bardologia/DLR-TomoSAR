from __future__ import annotations

import _bootstrap

from datetime import datetime

from configuration.processing_config import (
    CropRegion,
    ParallelConfiguration,
    PathConfiguration,
    PreProcessEntryConfig,
    ProcessingConfiguration,
    TomogramConfiguration,
)
from pipelines.processing_pipeline.pipeline import PreProcessScheduler, PreProcessSession
from tools.config_cli import ConfigCli
from tools.logger import Logger


def main() -> None:
    config = ConfigCli(PreProcessEntryConfig(), description="SAR pre-processing, runs win filters as concurrent sessions").apply()
    logger = Logger(log_dir="logs", name="pre_process")

    global_crop    = CropRegion(azimuth_start=config.azimuth_start, azimuth_end=config.azimuth_end, range_start=config.range_start, range_end=config.range_end)
    run_identifier = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.section("Pre-processing queue")
    logger.kv_table({
        "Win filters"  : ", ".join(str(win) for win in config.win_list),
        "Runs"         : len(config.win_list),
        "Max sessions" : config.max_sessions,
        "Crop"         : global_crop.as_tuple(),
    }, title="Configuration")

    sessions = []

    for index, win in enumerate(config.win_list):
        filter_arguments = {"win": list(win)}
        dataset_name     = config.resolve_dataset_name(win, run_identifier)

        logger.subsection(f"[Session {index + 1}/{len(config.win_list)}] {dataset_name} queued with filter arguments {filter_arguments}")

        tomogram_config = TomogramConfiguration(
            fusar_project_path = config.fusar_project_path,
            base_directory     = config.base_directory,
            track_selection    = config.track_selection,
            polarisation       = config.polarisation,
            beamforming_method = config.beamforming_method,
            filter_method      = config.filter_method,
            filter_arguments   = filter_arguments,
            height_range       = tuple(config.height_range),
        )

        processing_config = ProcessingConfiguration(
            crop            = global_crop,
            tomogram_config = tomogram_config,

            parallel = ParallelConfiguration(effort=config.effort),

            paths                = PathConfiguration(run_subdirectory=dataset_name),
            dataset_type         = config.dataset_type,
            stack_identifier     = config.stack_identifier,
            tomogram_output_tag  = config.tomogram_output_tag,
            parameter_output_tag = config.parameter_output_tag,
        )

        sessions.append(PreProcessSession(index=index, total=len(config.win_list), dataset_name=dataset_name, config=processing_config))

    scheduler = PreProcessScheduler(sessions=sessions, max_sessions=config.max_sessions, logger=logger)
    scheduler.run()

    logger.close()


if __name__ == "__main__":
    main()
