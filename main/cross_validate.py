from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _scheduler() -> None:
    os.environ["MKL_NUM_THREADS"]     = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    os.environ["OMP_NUM_THREADS"]     = "4"

    from configuration.cross_validation_config        import CrossValidationConfig
    from pipelines.cross_validation_pipeline.pipeline import CrossValidationPipeline
    from tools.config_cli                             import ConfigCli

    config = ConfigCli(CrossValidationConfig(), description="K-fold cross-validation").apply()

    pipeline = CrossValidationPipeline(config=config, entry_script=Path(__file__).resolve())
    pipeline.run()


def _worker(stage: str, fold_index: int, split: str | None, gpu_id: int, run_tag: str, run_dir: str | None) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MKL_NUM_THREADS"]      = "4"
    os.environ["NUMEXPR_NUM_THREADS"]  = "4"
    os.environ["OMP_NUM_THREADS"]      = "4"

    from configuration.cross_validation_config       import CrossValidationConfig
    from pipelines.cross_validation_pipeline.workers import FoldInferenceWorker, FoldTrainingWorker
    from tools.config_cli                            import ConfigCli

    config        = CrossValidationConfig()
    resolved_path = Path(run_dir) / "pipeline" / "resolved_config.json" if run_dir else Path(config.paths.log_base_dir) / run_tag / "pipeline" / "resolved_config.json"
    config        = ConfigCli.load_resolved(config, resolved_path)

    if stage == "train":
        FoldTrainingWorker(config=config, run_tag=run_tag).run(fold_index=fold_index)
    else:
        FoldInferenceWorker(config=config, run_tag=run_tag).run(fold_index=fold_index, split=split)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker",  type=str, default=None, choices=["train", "infer"])
    parser.add_argument("--fold",    type=int, default=None)
    parser.add_argument("--split",   type=str, default=None)
    parser.add_argument("--gpu",     type=int, default=0)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.worker:
        if args.fold is None or args.run_tag is None:
            sys.exit("ERROR: --worker requires --fold and --run-tag")
        if args.worker == "infer" and args.split is None:
            sys.exit("ERROR: --worker infer requires --split")
        _worker(stage=args.worker, fold_index=args.fold, split=args.split, gpu_id=args.gpu, run_tag=args.run_tag, run_dir=args.run_dir)
    else:
        _scheduler()


if __name__ == "__main__":
    main()
