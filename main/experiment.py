from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib  import Path

from _bootstrap import EnvironmentPinner


MODES = ("benchmark", "cv", "tune")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode",        choices=MODES, default=None)
    parser.add_argument("--worker",      nargs="?", const=True, default=None)
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--model",       type=str, default=None)
    parser.add_argument("--fold",        type=int, default=None)
    parser.add_argument("--split",       type=str, default=None)
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--n-trials",    type=int, default=8)
    parser.add_argument("--study-name",  type=str, default=None)
    parser.add_argument("--storage-url", type=str, default=None)
    parser.add_argument("--run-tag",     type=str, default=None)
    parser.add_argument("--run-dir",     type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


def _benchmark_scheduler(config, entry_script: Path) -> None:
    from pipelines.benchmark.pipeline import BenchmarkPipeline
    from tools.runtime.config_cli     import ConfigCli

    config = ConfigCli(config, description="Full architecture benchmark").apply()
    BenchmarkPipeline(config=config, entry_script=entry_script).run()


def _cv_scheduler(config, entry_script: Path) -> None:
    from pipelines.cross_validation.pipeline import CrossValidationPipeline
    from tools.runtime.config_cli            import ConfigCli

    config = ConfigCli(config, description="K-fold cross-validation").apply()
    CrossValidationPipeline(config=config, entry_script=entry_script).run()


def _tune_scheduler(config, entry_script: Path, args: argparse.Namespace) -> None:
    from pipelines.tuning.pipeline import TuningOrchestrator
    from tools.runtime.config_cli  import ConfigCli

    cli    = ConfigCli(config, description="Optuna hyperparameter tuning")
    config = cli.apply()
    tag    = args.run_tag or config.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.resume:
        if args.run_tag is None and config.run_tag is None:
            sys.exit("ERROR: --resume requires --run-tag")

        resolved_path = Path(config.paths.log_base_dir) / tag / "resolved_config.json"
        if not resolved_path.exists():
            sys.exit(f"ERROR: --resume given but no resolved config found at {resolved_path}")

        config = ConfigCli.load_resolved(type(config)(), resolved_path)
        config = ConfigCli.apply_overrides(config, cli.overrides)

    orchestrator = TuningOrchestrator(tag=tag, config=config, entry_script=entry_script)
    orchestrator.schedule(target_model=args.model, resume=args.resume)


def _scheduler(args: argparse.Namespace) -> None:
    EnvironmentPinner.threads()

    from configuration.experiments.experiment_config import ExperimentEntryConfig

    entry        = ExperimentEntryConfig()
    mode         = args.mode or entry.mode
    entry_script = Path(__file__).resolve()

    if mode == "benchmark":
        _benchmark_scheduler(entry.benchmark, entry_script)
    elif mode == "cv":
        _cv_scheduler(entry.cv, entry_script)
    else:
        _tune_scheduler(entry.tune, entry_script, args)


def _benchmark_worker(args: argparse.Namespace) -> None:
    from configuration.experiments.benchmark_config import BenchmarkConfig
    from pipelines.benchmark.workers                import InferenceWorker, OverfitWorker, TrainingWorker
    from tools.runtime.config_cli                   import ConfigCli

    if args.model is None or args.run_tag is None:
        sys.exit("ERROR: benchmark worker requires --model and --run-tag")

    config        = BenchmarkConfig()
    resolved_path = Path(args.run_dir) / "pipeline" / "resolved_config.json" if args.run_dir else Path(config.paths.log_base_dir) / args.run_tag / "pipeline" / "resolved_config.json"
    config        = ConfigCli.load_resolved(config, resolved_path)

    workers = {
        "overfit" : OverfitWorker,
        "train"   : TrainingWorker,
        "infer"   : InferenceWorker,
    }

    workers[args.worker](config=config, run_tag=args.run_tag).run(model_name=args.model)


def _cv_worker(args: argparse.Namespace) -> None:
    from configuration.experiments.cross_validation_config import CrossValidationConfig
    from pipelines.cross_validation.workers                import FoldInferenceWorker, FoldTrainingWorker
    from tools.runtime.config_cli                          import ConfigCli

    if args.fold is None or args.run_tag is None:
        sys.exit("ERROR: cv worker requires --fold and --run-tag")
    if args.worker == "infer" and args.split is None:
        sys.exit("ERROR: cv worker infer requires --split")

    config        = CrossValidationConfig()
    resolved_path = Path(args.run_dir) / "pipeline" / "resolved_config.json" if args.run_dir else Path(config.paths.log_base_dir) / args.run_tag / "pipeline" / "resolved_config.json"
    config        = ConfigCli.load_resolved(config, resolved_path)

    if args.worker == "train":
        FoldTrainingWorker(config=config, run_tag=args.run_tag).run(fold_index=args.fold)
    else:
        FoldInferenceWorker(config=config, run_tag=args.run_tag).run(fold_index=args.fold, split=args.split)


def _tune_worker(args: argparse.Namespace, entry_script: Path) -> None:
    from configuration.experiments.tuning_config import TuningEntryConfig
    from pipelines.tuning.pipeline               import TuningOrchestrator
    from tools.runtime.config_cli                import ConfigCli

    if args.model is None:
        sys.exit("ERROR: tune worker requires --model")
    if args.study_name is None or args.storage_url is None:
        sys.exit("ERROR: tune worker requires --study-name and --storage-url")

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


def _worker(args: argparse.Namespace) -> None:
    EnvironmentPinner.gpu(args.gpu)
    entry_script = Path(__file__).resolve()

    if args.study_name is not None or args.storage_url is not None:
        _tune_worker(args, entry_script)
    elif args.fold is not None:
        _cv_worker(args)
    else:
        _benchmark_worker(args)


def main() -> None:
    args = _parse_args()
    if args.worker is not None:
        _worker(args)
    else:
        _scheduler(args)


if __name__ == "__main__":
    main()
