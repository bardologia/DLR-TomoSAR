from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib     import Path

from _bootstrap import EnvironmentPinner


def _scheduler() -> None:
    EnvironmentPinner.threads()

    from configuration.inference.profile_autoencoder             import ProfileAeInferenceEntryConfig
    from pipelines.profile_autoencoder.inference.scheduler       import ProfileAeInferenceScheduler
    from tools.runtime.config_cli                                import ConfigCli

    config = ConfigCli(ProfileAeInferenceEntryConfig(), description="Standalone profile-autoencoder inference: reconstruct held-out profiles, score reconstruction quality, and report, fanned out across the selected GPUs").apply()

    results = ProfileAeInferenceScheduler(config=config, entry_script=Path(__file__).resolve()).run()

    if any(result.status != "DONE" for result in results):
        sys.exit(1)


def _worker(run_dir: str, config_path: str, gpu_id: int) -> None:
    EnvironmentPinner.gpu(gpu_id)

    from configuration.inference.profile_autoencoder            import ProfileAeInferenceEntryConfig
    from pipelines.profile_autoencoder.inference.pipeline       import ProfileAeInferencePipeline
    from tools.runtime.config_cli                               import ConfigCli

    config   = ConfigCli.load_resolved(ProfileAeInferenceEntryConfig(), Path(config_path))
    pipeline = ProfileAeInferencePipeline(replace(config.inference, run_directory=Path(run_dir), output_subdir=None), entry_config=config)
    pipeline.run()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker",  action="store_true")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--config",  type=str, default=None)
    parser.add_argument("--gpu",     type=int, default=0)
    args, _ = parser.parse_known_args()

    if args.worker:
        if not args.run_dir or not args.config:
            sys.exit("ERROR: --worker requires --run-dir and --config")
        _worker(run_dir=args.run_dir, config_path=args.config, gpu_id=args.gpu)
    else:
        _scheduler()


if __name__ == "__main__":
    main()
