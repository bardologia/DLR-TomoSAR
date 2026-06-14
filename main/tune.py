from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from _bootstrap import EnvironmentPinner


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker",      action="store_true")
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--model",       type=str,  default=None, help="(scheduler) tune only this model; (worker) model being tuned")
    parser.add_argument("--gpu",         type=int,  default=0)
    parser.add_argument("--n-trials",    type=int,  default=8)
    parser.add_argument("--study-name",  type=str,  default=None)
    parser.add_argument("--storage-url", type=str,  default=None)
    parser.add_argument("--run-tag",     type=str,  default=None)
    parser.add_argument("--run-dir",     type=str,  default=None)
    args, _ = parser.parse_known_args()

    if args.worker:
        EnvironmentPinner.gpu(args.gpu)
    else:
        EnvironmentPinner.threads()

    from configuration.experiments.tuning_config import TuningEntryConfig
    from pipelines.tuning_pipeline.pipeline import TuningOrchestrator
    from tools.runtime.config_cli import ConfigCli

    entry_script = Path(__file__).resolve()

    if args.worker:
        if args.model is None:
            sys.exit("ERROR: --worker requires --model")
        if args.study_name is None or args.storage_url is None:
            sys.exit("ERROR: --worker requires --study-name and --storage-url")

        tag    = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        config = TuningEntryConfig()
        if args.run_dir:
            config = ConfigCli.load_resolved(config, Path(args.run_dir) / "resolved_config.json")

        orchestrator = TuningOrchestrator(tag=tag, config=config, entry_script=entry_script)
        orchestrator.run_worker(
            model_name  = args.model,
            gpu_id      = args.gpu,
            n_trials    = args.n_trials,
            study_name  = args.study_name,
            storage_url = args.storage_url,
        )

    else:
        cli    = ConfigCli(TuningEntryConfig(), description="Optuna hyperparameter tuning")
        config = cli.apply()
        tag    = args.run_tag or config.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

        if args.resume:
            if args.run_tag is None and config.run_tag is None:
                sys.exit("ERROR: --resume requires --run-tag")

            resolved_path = Path(config.paths.log_base_dir) / tag / "resolved_config.json"
            if not resolved_path.exists():
                sys.exit(f"ERROR: --resume given but no resolved config found at {resolved_path}")

            config = ConfigCli.load_resolved(TuningEntryConfig(), resolved_path)
            config = ConfigCli.apply_overrides(config, cli.overrides)

        orchestrator = TuningOrchestrator(tag=tag, config=config, entry_script=entry_script)
        orchestrator.schedule(target_model=args.model, resume=args.resume)


if __name__ == "__main__":
    main()
