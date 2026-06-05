from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from configuration.benchmark_config              import BenchmarkConfig
from pipelines.benchmark_pipeline.results        import ComparisonReport, TrialCollector
from pipelines.benchmark_pipeline.sizing         import SizeMatcher, SizeMatchResult
from pipelines.shared.orchestration              import ExperimentStage, GpuJob
from tools.logger                                import Logger


class OverfitStage(ExperimentStage):
    def __init__(self, config: BenchmarkConfig, entry_script: Path, run_tag: str, models: list[str], logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger, entry_script=entry_script)
        self.models       = models
        self.stage_dir    = self.run_dir / "overfit"
        self.results_path = self.run_dir / "pipeline" / "overfit_results.json"
        self.report_path  = self.run_dir / "pipeline" / "overfit_report.md"

    def run(self) -> list[dict]:
        self.logger.section("Overfit gate")
        self.logger.kv_table({
            "Models"              : len(self.models),
            "Max steps"           : self.config.overfit.max_steps,
            "Stop threshold"      : self.config.overfit.stop_threshold,
            "Batch size"          : self.config.overfit.batch_size,
            "Require convergence" : self.config.overfit.require_convergence,
            "GPUs"                : self.config.gpus,
        }, title="Configuration")

        cached  = [m for m in self.models if self._has_result(m)]
        pending = [m for m in self.models if m not in cached]

        for model_name in cached:
            self.logger.info(f"{model_name}: cached result reused")

        if pending:
            self._run_queue([self._job(m) for m in pending])

        results = [self._load_result(m) for m in self.models]

        self._write_results(results, self.results_path)
        self._write_report(results)
        self._log_summary(results)

        return results

    def passed(self, results: list[dict]) -> bool:
        return bool(results) and all(r["status"] == "PASS" for r in results)

    def _job(self, model_name: str) -> GpuJob:
        return GpuJob(
            name     = model_name,
            command  = [sys.executable, str(self.entry_script), "--worker", "overfit", "--model", model_name, "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self.stage_dir / model_name / "worker.log",
        )

    def _result_path(self, model_name: str) -> Path:
        return self.stage_dir / model_name / "overfit_result.json"

    def _has_result(self, model_name: str) -> bool:
        return self.config.resume and self._result_path(model_name).exists()

    def _load_result(self, model_name: str) -> dict:
        path = self._result_path(model_name)

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "model"      : model_name,
                "status"     : "FAIL",
                "final_loss" : None,
                "converged"  : None,
                "threshold"  : self.config.overfit.stop_threshold,
                "error"      : f"missing or unreadable result file: {path}",
            }

    def _write_report(self, results: list[dict]) -> None:
        passed = [r for r in results if r["status"] == "PASS"]
        failed = [r for r in results if r["status"] != "PASS"]

        lines = [
            "# Overfit Gate Report",
            f"\n_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  —  run tag `{self.run_tag}`_\n",
            f"**Verdict: {'GATE PASSED' if not failed else 'GATE FAILED'}**  —  {len(passed)}/{len(results)} models passed.\n",
            "## Configuration\n",
            "| Parameter | Value |",
            "| --- | --- |",
            f"| Max steps | {self.config.overfit.max_steps} |",
            f"| Stop threshold | {self.config.overfit.stop_threshold:g} |",
            f"| Batch size | {self.config.overfit.batch_size} |",
            f"| Crop | {self.config.overfit.azimuth_lines} × {self.config.overfit.range_lines} (azimuth start {self.config.overfit.azimuth_start}) |",
            f"| Require convergence | {self.config.overfit.require_convergence} |",
            f"| Seed | {self.config.overfit.seed} |",
            "",
            "## Results\n",
            "| Model | Status | Final loss | Converged | Error |",
            "| --- | --- | --- | --- | --- |",
        ]

        for r in results:
            final_loss = f"{r['final_loss']:.4e}" if r.get("final_loss") is not None else "—"
            converged  = {True: "yes", False: "no", None: "—"}[r.get("converged")]
            error      = (r.get("error") or "—").strip().splitlines()[-1]
            lines.append(f"| `{r['model']}` | {r['status']} | {final_loss} | {converged} | {error} |")

        lines.append("")

        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Report written to: {self.report_path}")

    def _log_summary(self, results: list[dict]) -> None:
        passed = [r for r in results if r["status"] == "PASS"]
        failed = [r for r in results if r["status"] != "PASS"]

        self.logger.subsection("Gate summary")
        self.logger.kv_table({
            "Total"  : len(results),
            "Passed" : len(passed),
            "Failed" : len(failed),
        }, title=f"{len(passed)}/{len(results)} passed")

        for r in failed:
            self.logger.error(f"FAILED  {r['model']}")


class SizeMatchStage(ExperimentStage):
    def __init__(self, config: BenchmarkConfig, run_tag: str, models: list[str], logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger)
        self.models       = models
        self.records_path = self.run_dir / "pipeline" / "size_match.json"
        self.report_path  = self.run_dir / "pipeline" / "size_match_report.md"

    def run(self) -> dict:
        self.logger.section("Capacity matching")
        self.logger.kv_table({
            "Reference model" : self.config.size_match.reference_model,
            "Tolerance"       : f"{100.0 * self.config.size_match.tolerance:.2f} %",
            "Max iterations"  : self.config.size_match.max_iterations,
            "In channels"     : self.config.size_match.in_channels,
            "Out channels"    : self.config.n_gaussians * 3,
        }, title="Configuration")

        cached = self._load_cached()
        if cached is not None:
            self.logger.info(f"Cached size match reused from: {self.records_path}")
            return cached

        matcher   = SizeMatcher(config=self.config, logger=self.logger)
        reference = self.config.size_match.reference_model
        target    = matcher.reference_count()

        self.logger.info(f"Reference '{reference}': {target:,} parameters")

        records = {reference: self._reference_record(reference, target)}

        for model_name in self.models:
            if model_name == reference:
                continue

            result               = matcher.match(model_name, target)
            records[model_name]  = asdict(result)

            self.logger.info(f"{model_name:<18} {result.parameters:>14,}  Δ {result.deviation_pct:+7.3f} %  (scale {result.scale:.4f}, {result.iterations} iterations)")

        self._write_results(records, self.records_path)
        self._write_report(records, target)

        return records

    def _reference_record(self, reference: str, target: int) -> dict:
        return asdict(SizeMatchResult(
            model         = reference,
            scale         = 1.0,
            overrides     = {},
            parameters    = target,
            target        = target,
            deviation_pct = 0.0,
            iterations    = 0,
        ))

    def _load_cached(self) -> dict | None:
        if not self.config.resume or not self.records_path.exists():
            return None

        try:
            with open(self.records_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            return None

        if all(m in records for m in self.models):
            return records

        return None

    def _write_report(self, records: dict, target: int) -> None:
        lines = [
            "# Capacity Matching Report",
            f"\n_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  —  run tag `{self.run_tag}`_\n",
            f"Reference model `{self.config.size_match.reference_model}` at **{target:,}** parameters.",
            f"Counting performed with {self.config.size_match.in_channels} input channels, {self.config.n_gaussians * 3} output channels, image size {self.config.training.patch_size[0]}.\n",
            "## Matched Widths\n",
            "| Model | Scale | Scaled attributes | Parameters | Δ vs reference | Iterations |",
            "| --- | --- | --- | --- | --- | --- |",
        ]

        for model_name in sorted(records.keys()):
            record    = records[model_name]
            overrides = ", ".join(f"`{k}={v}`" for k, v in record["overrides"].items()) or "_(defaults)_"
            lines.append(f"| `{model_name}` | {record['scale']:.4f} | {overrides} | {record['parameters']:,} | {record['deviation_pct']:+.3f} % | {record['iterations']} |")

        lines.append("")

        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Report written to: {self.report_path}")


class TrainingStage(ExperimentStage):
    def __init__(self, config: BenchmarkConfig, entry_script: Path, run_tag: str, models: list[str], logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger, entry_script=entry_script)
        self.models       = models
        self.stage_dir    = self.run_dir / "training"
        self.results_path = self.run_dir / "pipeline" / "training_results.json"

    def run(self) -> list[dict]:
        self.logger.section("Training queue")
        self.logger.kv_table({
            "Models"     : len(self.models),
            "Epochs"     : self.config.training.epochs,
            "Batch size" : self.config.training.batch_size,
            "GPUs"       : self.config.gpus,
            "Stage dir"  : str(self.stage_dir),
        }, title="Configuration")

        cached  = [m for m in self.models if self._has_checkpoint(m)]
        pending = [m for m in self.models if m not in cached]

        for model_name in cached:
            self.logger.info(f"{model_name}: existing checkpoint reused")

        ran = []
        if pending:
            ran = self._run_queue([self._job(m) for m in pending])

        results = self._order_results(ran + [self._cached_result(m) for m in cached], self.models)

        self._write_results(results, self.results_path)
        self._log_summary(results)

        return results

    def _job(self, model_name: str) -> GpuJob:
        return GpuJob(
            name     = model_name,
            command  = [sys.executable, str(self.entry_script), "--worker", "train", "--model", model_name, "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self.stage_dir / model_name / "worker.log",
        )

    def _has_checkpoint(self, model_name: str) -> bool:
        if not self.config.resume:
            return False

        model_dir = self.stage_dir / model_name
        if not model_dir.is_dir():
            return False

        return next(model_dir.rglob(self.config.inference.checkpoint_name), None) is not None

    def _cached_result(self, model_name: str) -> dict:
        return {
            "name"       : model_name,
            "gpu"        : None,
            "status"     : "DONE",
            "returncode" : 0,
            "duration_s" : None,
            "log_file"   : str(self.stage_dir / model_name / "worker.log"),
        }

    def _log_summary(self, results: list[dict]) -> None:
        done   = [r for r in results if r["status"] == "DONE"]
        failed = [r for r in results if r["status"] != "DONE"]

        self.logger.subsection("Training summary")
        self.logger.kv_table({
            "Total"  : len(results),
            "Done"   : len(done),
            "Failed" : len(failed),
        }, title=f"{len(done)}/{len(results)} finished")

        self._log_failures(failed)


class InferenceStage(ExperimentStage):
    def __init__(self, config: BenchmarkConfig, entry_script: Path, run_tag: str, models: list[str], logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger, entry_script=entry_script)
        self.models       = models
        self.training_dir = self.run_dir / "training"
        self.results_path = self.run_dir / "pipeline" / "inference_results.json"

    def run(self) -> list[dict]:
        self.logger.section("Batch inference")
        self.logger.kv_table({
            "Models" : len(self.models),
            "Split"  : self.config.inference.split,
            "EMA"    : self.config.inference.use_ema,
            "GPUs"   : self.config.gpus,
        }, title="Configuration")

        skipped = [m for m in self.models if not self._has_checkpoint(m)]
        cached  = [m for m in self.models if m not in skipped and self._has_inference(m)]
        pending = [m for m in self.models if m not in skipped and m not in cached]

        for model_name in skipped:
            self.logger.warning(f"{model_name}: no checkpoint, inference skipped")

        for model_name in cached:
            self.logger.info(f"{model_name}: existing inference reused")

        ran = []
        if pending:
            ran = self._run_queue([self._job(m) for m in pending])

        results = ran + [self._static_result(m, "DONE") for m in cached] + [self._static_result(m, "SKIPPED") for m in skipped]
        results = self._order_results(results, self.models)

        self._write_results(results, self.results_path)
        self._log_summary(results)

        return results

    def _job(self, model_name: str) -> GpuJob:
        return GpuJob(
            name     = model_name,
            command  = [sys.executable, str(self.entry_script), "--worker", "infer", "--model", model_name, "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self.training_dir / model_name / "inference_worker.log",
        )

    def _has_checkpoint(self, model_name: str) -> bool:
        model_dir = self.training_dir / model_name
        if not model_dir.is_dir():
            return False

        return next(model_dir.rglob(self.config.inference.checkpoint_name), None) is not None

    def _has_inference(self, model_name: str) -> bool:
        if not self.config.resume:
            return False

        inference_dir = self.training_dir / model_name / "inference"
        if not inference_dir.is_dir():
            return False

        return next(inference_dir.glob("*/metrics.json"), None) is not None

    def _static_result(self, model_name: str, status: str) -> dict:
        return {
            "name"       : model_name,
            "gpu"        : None,
            "status"     : status,
            "returncode" : 0 if status == "DONE" else None,
            "duration_s" : None,
            "log_file"   : str(self.training_dir / model_name / "inference_worker.log"),
        }

    def _log_summary(self, results: list[dict]) -> None:
        done   = [r for r in results if r["status"] == "DONE"]
        failed = [r for r in results if r["status"] == "FAILED"]

        self.logger.subsection("Inference summary")
        self.logger.kv_table({
            "Total"   : len(results),
            "Done"    : len(done),
            "Failed"  : len(failed),
            "Skipped" : len(results) - len(done) - len(failed),
        }, title=f"{len(done)}/{len(results)} finished")

        self._log_failures(failed)


class ComparisonStage(ExperimentStage):
    def __init__(self, config: BenchmarkConfig, run_tag: str, logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger)

    def run(self) -> Path:
        self.logger.section("Comparison reports")

        collector = TrialCollector(run_dir=self.run_dir, logger=self.logger)
        records   = collector.collect()

        out_dir = self.run_dir / "comparison" / datetime.now().strftime("%Y%m%d_%H%M%S")

        report = ComparisonReport(
            records         = records,
            out_dir         = out_dir,
            reference_model = self.config.size_match.reference_model,
            embed_images    = self.config.comparison.embed_images,
            logger          = self.logger,
        )

        written = report.write_all()

        self.logger.subsection("Reports written")
        for path in written:
            self.logger.info(f"{path}")

        return out_dir
