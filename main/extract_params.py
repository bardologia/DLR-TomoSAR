from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning)

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
from pipelines.param_pipeline.pipeline import ParamExtractionPipeline


processed_data_path = Path("/ste/rnd/User/vice_vi/Dataset/clean_dataset")
pyrat_directory     = Path("/ste/rnd/User/vice_vi/pyrat")

output_prefix     = "params"
output_suffix     = None

tomogram_filename = "tomogram_full_1000a16000a500a4000_1_Xparams_id2X.npy"
height_range      = None

fit_config        = FitMode.SigmaOnly(k_max=5, lambda_k=3e-3)

parameter_workers = 50


def main() -> None:
    config = ExtractionConfig(
        processed_data_path = processed_data_path,
        pyrat_directory     = pyrat_directory,

        output_prefix     = output_prefix,
        output_suffix     = output_suffix,

        tomogram_filename = tomogram_filename,
        height_range      = height_range,

        fit_settings = FitSettings(fit_config = fit_config),

        parameter_workers = parameter_workers,
    )

    pipeline  = ParamExtractionPipeline(config)
    outputs   = pipeline.run()

    print("[Execution Successful] Outputs:")
    for name, path in outputs.items():
        print(f"  {name:>18}: {path}")


if __name__ == "__main__":
    main()
