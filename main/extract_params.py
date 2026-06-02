from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning)

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuration.param_extraction_config import FitMode, ExtractionConfig, FitSettings
from pipelines.param_pipeline.pipeline     import ParamExtractionPipeline


dataset_base_path   = Path("/ste/rnd/User/vice_vi/Dataset")
pyrat_directory     = Path("/ste/rnd/User/vice_vi/pyrat")
tomogram_filename   = "tomogram_full_1000a16000a500a4000_1_Xtomo_id2X.npy"

output_prefix       = "params"
output_suffix       = None
height_range        = None

fit_k_max           = 5
fit_lambda_k        = 3e-3
parameter_workers   = 50


def main() -> None:
    dataset_dirs = sorted(p for p in dataset_base_path.iterdir() if p.is_dir())

    for i, processed_data_path in enumerate(dataset_dirs):
        print(f"\n[Run {i + 1}/{len(dataset_dirs)}] {processed_data_path.name}")

        config = ExtractionConfig(
            processed_data_path = processed_data_path,
            pyrat_directory     = pyrat_directory,

            output_prefix     = output_prefix,
            output_suffix     = output_suffix,

            tomogram_filename = tomogram_filename,
            height_range      = height_range,

            fit_settings      = FitSettings(fit_config=FitMode.SigmaOnly(k_max=fit_k_max, lambda_k=fit_lambda_k)),

            parameter_workers = parameter_workers,
        )

        pipeline  = ParamExtractionPipeline(config)
        outputs   = pipeline.run()

        print("[Execution Successful] Outputs:")
        for name, path in outputs.items():
            print(f"  {name:>18}: {path}")


if __name__ == "__main__":
    main()
