from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuration.param_extraction_config import (
    FitMode,
    ExtractionConfig,
    FitSettings,
)
from pipelines.param_extraction_pipeline.pipeline import ParamExtractionPipeline


def main() -> None:
    config = ExtractionConfig(
        processed_data_path = Path("/ste/rnd/User/vice_vi/Dataset/base_dataset"),
        pyrat_directory     = Path("/ste/rnd/User/vice_vi/pyrat"),

        output_prefix     = "params",
        output_suffix     = None,
        
        tomogram_filename = "tomofull_1000a1050a500a550_1_Xparams_id2X.npy",
        height_range      = None,

        fit_settings = FitSettings(
            number_of_gaussians = 2,
            max_fit_iterations  = 5000,
            fit_config          = FitMode.Adaptive(
                initial_guess = None,
                lower_bounds  = None,
                upper_bounds  = None,
            ),
        ),

        parameter_workers = 50,
    )

    pipeline  = ParamExtractionPipeline(config)
    outputs   = pipeline.run()

    print("[Execution Successful] Outputs:")
    for name, path in outputs.items():
        print(f"  {name:>18}: {path}")


if __name__ == "__main__":
    main()
