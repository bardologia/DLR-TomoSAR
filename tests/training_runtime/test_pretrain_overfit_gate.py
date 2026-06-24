from __future__ import annotations

import pytest

from tools.training.pretraining.overfit_gate import OverfitGate


def test_pass_when_loss_below_threshold():
    gate   = OverfitGate(run_overfit=lambda: 1e-4, stop_threshold=1e-3, require_convergence=True, abort_on_fail=True)
    result = gate.run()

    assert result["status"]     == "PASS"
    assert result["converged"]  is True
    assert result["final_loss"] == 1e-4


def test_fail_and_abort_when_no_final_loss():
    gate = OverfitGate(run_overfit=lambda: None, stop_threshold=1e-3)

    with pytest.raises(SystemExit):
        gate.run()


def test_fail_and_abort_on_nonconvergence_when_required():
    gate = OverfitGate(run_overfit=lambda: 1.0, stop_threshold=1e-3, require_convergence=True)

    with pytest.raises(SystemExit):
        gate.run()


def test_no_abort_returns_failure_dict():
    gate   = OverfitGate(run_overfit=lambda: 1.0, stop_threshold=1e-3, require_convergence=True, abort_on_fail=False)
    result = gate.run()

    assert result["status"]    == "FAIL"
    assert result["converged"] is False


def test_inner_systemexit_without_loss_fails():
    def boom():
        raise SystemExit(2)

    gate   = OverfitGate(run_overfit=boom, stop_threshold=1e-3, abort_on_fail=False)
    result = gate.run()

    assert result["status"] == "FAIL"


def test_evaluate_is_pure_decision_logic():
    result = {"status": "PASS", "final_loss": 0.5, "converged": None, "threshold": 1.0, "error": None}

    OverfitGate.evaluate(result, require_convergence=True)

    assert result["converged"] is True
    assert result["status"]    == "PASS"
