from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuration.param_extraction_config import (
    FitMode,
    ExtractionConfig,
    FitSettings,
)
from pipelines.param_extraction_pipeline.pipeline import ParamExtractionPipeline


def main() -> None:
    for n_gaussians in range(2, 6):
        print(f"\n{'='*50}")
        print(f"  Running extraction with {n_gaussians} Gaussians")
        print(f"{'='*50}")

        config = ExtractionConfig(
            processed_data_path = Path("/ste/rnd/User/vice_vi/Dataset/run_1000a16000a500a4000_dtmf_Xtomo_id2X_20260513_155626"),
            pyrat_directory     = Path("/ste/rnd/User/vice_vi/pyrat"),

            output_prefix     = f"params_g{n_gaussians}",
            output_suffix     = None,
            tomogram_filename = "tomofull_1000a16000a500a4000_1_Xparams_id2X.npy",
            height_range      = None,

            fit_settings = FitSettings(
                number_of_gaussians = n_gaussians,
                max_fit_iterations  = 5000,
                fit_config          = FitMode.Adaptive(
                    initial_guess = None,
                    lower_bounds  = None,
                    upper_bounds  = None,
                ),
            ),

            parameter_workers = 50,
        )

        pipeline = ParamExtractionPipeline(config)
        outputs  = pipeline.run()

        print(f"[Done — {n_gaussians} Gaussians] Outputs:")
        for name, path in outputs.items():
            print(f"  {name:>18}: {path}")


if __name__ == "__main__":
    main()
