from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import EnvironmentPinner


def _scheduler() -> None:
    EnvironmentPinner.threads()

    from configuration.experiments.benchmark_config        import BenchmarkConfig
    from pipelines.benchmark.pipeline import BenchmarkPipeline
    from tools.runtime.config_cli                      import ConfigCli

    config = ConfigCli(BenchmarkConfig(), description="Full architecture benchmark").apply()

    pipeline = BenchmarkPipeline(config=config, entry_script=Path(__file__).resolve())
    pipeline.run()


def _worker(stage: str, model_name: str, gpu_id: int, run_tag: str, run_dir: str | None) -> None:
    EnvironmentPinner.gpu(gpu_id)

    from configuration.experiments.benchmark_config       import BenchmarkConfig
    from pipelines.benchmark.workers import InferenceWorker, OverfitWorker, TrainingWorker
    from tools.runtime.config_cli                     import ConfigCli

    config        = BenchmarkConfig()
    resolved_path = Path(run_dir) / "pipeline" / "resolved_config.json" if run_dir else Path(config.paths.log_base_dir) / run_tag / "pipeline" / "resolved_config.json"
    config        = ConfigCli.load_resolved(config, resolved_path)

    workers = {
        "overfit" : OverfitWorker,
        "train"   : TrainingWorker,
        "infer"   : InferenceWorker,
    }

    worker = workers[stage](config=config, run_tag=run_tag)
    worker.run(model_name=model_name)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker",  type=str, default=None, choices=["overfit", "train", "infer"])
    parser.add_argument("--model",   type=str, default=None)
    parser.add_argument("--gpu",     type=int, default=0)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.worker:
        if args.model is None or args.run_tag is None:
            sys.exit("ERROR: --worker requires --model and --run-tag")
        _worker(stage=args.worker, model_name=args.model, gpu_id=args.gpu, run_tag=args.run_tag, run_dir=args.run_dir)
    else:
        _scheduler()


if __name__ == "__main__":
    main()
