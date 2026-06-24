from __future__ import annotations

import traceback
from pathlib import Path
from typing  import Callable, Optional

from tools.data.io           import FileIO
from tools.monitoring.logger import Logger


class OverfitGate:
    def __init__(
        self,
        run_overfit         : Callable[[], Optional[float]],
        stop_threshold      : float,
        logger              : Optional[Logger] = None,
        label               : Optional[str]    = None,
        require_convergence : bool             = True,
        abort_on_fail       : bool             = True,
        result_path         : Optional[Path]   = None,
    ) -> None:
        self.run_overfit         = run_overfit
        self.stop_threshold      = float(stop_threshold)
        self.logger              = logger
        self.label               = label
        self.require_convergence = bool(require_convergence)
        self.abort_on_fail       = bool(abort_on_fail)
        self.result_path         = Path(result_path) if result_path is not None else None

    @staticmethod
    def evaluate(result: dict, require_convergence: bool) -> dict:
        threshold = result["threshold"]

        if result["final_loss"] is not None:
            result["converged"] = bool(result["final_loss"] <= threshold)

        if result["status"] == "PASS" and result["final_loss"] is None:
            result["status"] = "FAIL"
            result["error"]  = result["error"] or "overfit gate produced no final loss to evaluate"

        if result["status"] == "PASS" and require_convergence and result["converged"] is not True:
            result["status"] = "FAIL"
            result["error"]  = f"final loss {result['final_loss']:.3e} above stop threshold {threshold:.0e}"

        return result

    def _execute(self) -> dict:
        result = {
            "name"       : self.label,
            "status"     : None,
            "final_loss" : None,
            "converged"  : None,
            "threshold"  : self.stop_threshold,
            "error"      : None,
        }

        try:
            result["final_loss"] = self.run_overfit()
            result["status"]     = "PASS"
        except SystemExit:
            result["status"] = "PASS"
            result["error"]  = "overfit run exited via SystemExit before reporting a final loss"
        except Exception:
            result["status"] = "FAIL"
            result["error"]  = traceback.format_exc()

        return result

    def _finalize(self, result: dict) -> dict:
        self.evaluate(result, self.require_convergence)

        if self.result_path is not None:
            FileIO.save_json(result, self.result_path, indent=2)

        if self.logger is not None:
            self.logger.subsection(f"Overfit gate: {result['status']} (final loss {result['final_loss']}, threshold {self.stop_threshold:.0e})")

        if self.abort_on_fail and result["status"] == "FAIL":
            raise SystemExit(1)

        return result

    def run(self) -> dict:
        result = self._execute()

        return self._finalize(result)
