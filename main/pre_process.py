from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuration.processing_config import (
    CropRegion,
    ParallelConfiguration,
    PathConfiguration,
    ProcessingConfiguration,
    TomogramConfiguration,
)
from pipelines.processing_pipeline.pipeline import ProcessingPipeline


def main() -> None:
    global_crop = CropRegion(
        azimuth_start = 1000,
        azimuth_end   = 16000,
        range_start   = 500,
        range_end     = 4000,
    )

    tomogram_workers = 16
    pyrat_threads    = 8

    total_azimuth_width    = global_crop.azimuth_end - global_crop.azimuth_start
    max_crop_azimuth_width = total_azimuth_width // tomogram_workers   

    config = ProcessingConfiguration(
        crop = global_crop,

        input_configs = TomogramConfiguration(
            fusar_project_path     = "/ste/rnd/User/sera_se/17sartom-traun_L.csv",
            base_directory         = "/ste/rnd/",
            track_selection        = "*",
            polarisation           = "hv",
            beamforming_method     = "Capon",
            filter_method          = "Boxcar",
            filter_arguments       = {"win": [20, 10]},
            max_crop_azimuth_width = max_crop_azimuth_width,
        ),

        output_configs = TomogramConfiguration(
            fusar_project_path     = "/ste/rnd/User/sera_se/17sartom-traun_L.csv",
            base_directory         = "/ste/rnd/",
            polarisation           = "hv",
            track_selection        = "*",
            beamforming_method     = "Capon",
            filter_method          = "Boxcar",
            filter_arguments       = {"win": [20, 10]},
            height_range           = (-20.0, 80.0),
            max_crop_azimuth_width = max_crop_azimuth_width,
        ),

        parallel = ParallelConfiguration(
            tomogram_workers = tomogram_workers,
            pyrat_threads    = pyrat_threads,
        ),

        paths = PathConfiguration(
            main_directory = Path("/ste/rnd/User/vice_vi/Dataset"),
        ),

        dataset_type             = "FSAR",
        full_stack_identifier    = "1",
        reduced_stack_identifier = "dtmf",
        tomogram_output_tag      = "Xtomo_id2X",
        parameter_output_tag     = "Xparams_id2X",
    )

    pipeline = ProcessingPipeline(config)
    outputs  = pipeline.run()

    print("[Execution Successful] Outputs:")
    for name, path in outputs.items():
        print(f"  {name:>16}: {path}")


if __name__ == "__main__":
    main()
