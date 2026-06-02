from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuration.processing_config import (
    CropRegion,
    PathConfiguration,
    ProcessingConfiguration,
    TomogramConfiguration,
)
from pipelines.processing_pipeline.pipeline import ProcessingPipeline


azimuth_start            = 1000
azimuth_end              = 16000
range_start              = 500
range_end                = 4000

fusar_project_path       = "/ste/rnd/User/sera_se/17sartom-traun_L.csv"
base_directory           = "/ste/rnd/"
track_selection          = "*"
polarisation             = "hv"
beamforming_method       = "Capon"
filter_method            = "Boxcar"

filter_arguments_list    = [
    {"win": [40, 20]},
    {"win": [20, 10]},
    {"win": [10, 10]},
    {"win": [30, 20]},
]

height_range             = (-20.0, 80.0)

dataset_type             = "FSAR"
full_stack_identifier    = "1"
reduced_stack_identifier = "dtmf"
tomogram_output_tag      = "Xtomo_id2X"
parameter_output_tag     = "Xparams_id2X"


def main() -> None:
    global_crop            = CropRegion(azimuth_start=azimuth_start, azimuth_end=azimuth_end, range_start=range_start, range_end=range_end)
    max_crop_azimuth_width = (global_crop.azimuth_end - global_crop.azimuth_start) // 16

    for i, filter_arguments in enumerate(filter_arguments_list):
        win = filter_arguments.get("win", [])
        win_str = "_".join(str(w) for w in win)
        dataset_name = f"base_dataset_w{win_str}"
        print(f"\n[Run {i + 1}/{len(filter_arguments_list)}] {dataset_name}  filter_arguments={filter_arguments}")

        shared_tomo = dict(
            fusar_project_path     = fusar_project_path,
            base_directory         = base_directory,
            track_selection        = track_selection,
            polarisation           = polarisation,
            beamforming_method     = beamforming_method,
            filter_method          = filter_method,
            filter_arguments       = filter_arguments,
            max_crop_azimuth_width = max_crop_azimuth_width,
        )

        config = ProcessingConfiguration(
            crop = global_crop,

            input_configs  = TomogramConfiguration(**shared_tomo),
            output_configs = TomogramConfiguration(**shared_tomo, height_range=height_range),
            
            paths                    = PathConfiguration(run_subdirectory=dataset_name),
            dataset_type             = dataset_type,
            full_stack_identifier    = full_stack_identifier,
            reduced_stack_identifier = reduced_stack_identifier,
            tomogram_output_tag      = tomogram_output_tag,
            parameter_output_tag     = parameter_output_tag,
        )

        pipeline = ProcessingPipeline(config)
        outputs  = pipeline.run()

        print("[Execution Successful] Outputs:")
        for name, path in outputs.items():
            print(f"  {name:>16}: {path}")


if __name__ == "__main__":
    main()
