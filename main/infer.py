from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import EnvironmentPinner


def _scheduler() -> None:
    EnvironmentPinner.threads()

    from configuration.inference              import InferenceEntryConfig
    from pipelines.shared.inference_scheduler import InferenceScheduler
    from tools.runtime.config_cli             import ConfigCli

    config = ConfigCli(InferenceEntryConfig(), description="Inference over one or more run directories; backbone, the two autoencoders, and JEPA runs are auto-detected from their persisted configs and fanned out across the selected GPUs").apply()

    results = InferenceScheduler(config=config, entry_script=Path(__file__).resolve()).run()

    if any(result.status != "DONE" for result in results):
        sys.exit(1)


def _worker(run_dir: str, config_path: str, gpu_id: int) -> None:
    EnvironmentPinner.gpu(gpu_id)

    from configuration.inference             import InferenceEntryConfig
    from pipelines.shared.inference_dispatch import InferenceDispatcher
    from tools.runtime.config_cli            import ConfigCli

    config = ConfigCli.load_resolved(InferenceEntryConfig(), Path(config_path))

    InferenceDispatcher(config).run(Path(run_dir))


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
