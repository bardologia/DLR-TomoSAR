from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from configuration.benchmark_config import BenchmarkConfig
from pipelines.benchmark_pipeline.gpu_queue import GpuJob, GpuQueue
from tools.logger import Logger


class OverfitStage:
    def __init__(self, config: BenchmarkConfig, entry_script: Path, run_tag: str, models: list[str], logger: Logger) -> None:
        self.config       = config
        self.entry_script = entry_script
        self.run_tag      = run_tag
        self.models       = models
        self.logger       = logger

        self.run_dir      = Path(config.paths.log_base_dir) / run_tag
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
            queue = GpuQueue(gpus=self.config.gpus, logger=self.logger, poll_interval_s=self.config.poll_interval_s)
            queue.run([self._job(m) for m in pending])

        results = [self._load_result(m) for m in self.models]

        self._write_results(results)
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

    def _write_results(self, results: list[dict]) -> None:
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

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
