from __future__ import annotations

import os

from _bootstrap import EnvironmentPinner

from configuration.benchmark.dataloader_tuning import DataLoaderTuningEntryConfig
from tools.runtime.config_cli                  import ConfigCli


def main() -> None:
    config = ConfigCli(DataLoaderTuningEntryConfig(), description="Sweep DataLoader settings (batch size, workers, prefetch, pin-memory) to find the configuration that keeps the GPU fed").apply()

    EnvironmentPinner.gpu(config.gpu, expandable_segments=True)

    for thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[thread_var] = "1"

    from pipelines.dataloader_tuning.pipeline import DataLoaderTuningPipeline

    DataLoaderTuningPipeline(config).run()


if __name__ == "__main__":
    main()
