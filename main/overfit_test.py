from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _pin_environment() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=0)
    args, _ = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MKL_NUM_THREADS"]      = "4"
    os.environ["NUMEXPR_NUM_THREADS"]  = "4"
    os.environ["OMP_NUM_THREADS"]      = "4"


def main() -> None:
    _pin_environment()

    from configuration.benchmark_config import OverfitTestConfig
    from models import CONFIG_REGISTRY
    from pipelines.benchmark_pipeline.workers import OverfitWorker
    from tools.config_cli import ConfigCli
    from tools.logger import Logger

    config  = ConfigCli(OverfitTestConfig(), description="Sequential in-process overfit sanity check").apply()
    run_tag = config.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.paths.log_base_dir) / run_tag

    logger = Logger(log_dir=str(run_dir / "pipeline"), name="overfit_test")
    worker = OverfitWorker(config=config, run_tag=run_tag)
    models = [m for m in CONFIG_REGISTRY.keys() if m not in set(config.skip_models)]

    logger.section("Overfit sanity-check")
    logger.kv_table({
        "Models"          : len(models),
        "Steps per model" : config.overfit.max_steps,
        "Stop threshold"  : config.overfit.stop_threshold,
        "Batch size"      : config.overfit.batch_size,
        "Run dir"         : str(run_dir),
    }, title="Configuration")

    logger.section("Running tests")
    results = []
    for model_name in models:
        logger.subsection(model_name)

        try:
            worker.run(model_name=model_name)
        except SystemExit:
            pass
        except Exception as e:
            logger.error(f"{model_name}  :  {type(e).__name__}: {e}")

        result_path = run_dir / "overfit" / model_name / "overfit_result.json"
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            result = {"model": model_name, "status": "FAIL", "final_loss": None, "converged": None, "error": f"missing result file: {result_path}"}

        results.append(result)
        logger.info(f"{model_name}  :  {result['status']}")

    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] != "PASS"]

    logger.section("Summary")
    logger.kv_table({
        "Total"  : len(results),
        "Passed" : len(passed),
        "Failed" : len(failed),
    }, title=f"{len(passed)}/{len(results)} passed")

    for r in failed:
        logger.error(f"FAILED  {r['model']}")

    output_path = run_dir / "pipeline" / "overfit_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")

    logger.close()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
