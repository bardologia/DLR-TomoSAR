from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools.orchestration.gpu_queue import GpuJob, GpuQueue, GpuJobResult


class NullLogger:
    def section(self, *a, **k):    pass
    def subsection(self, *a, **k): pass
    def info(self, *a, **k):       pass
    def warning(self, *a, **k):    pass
    def error(self, *a, **k):      pass
    def kv_table(self, *a, **k):   pass


@pytest.fixture
def logger():
    return NullLogger()


def _ok_command(payload: str = "ok") -> list[str]:
    return [sys.executable, "-c", f"print('{payload}')"]


def _fail_command(code: int = 7) -> list[str]:
    return [sys.executable, "-c", f"import sys; sys.exit({code})"]


def _job(tmp_path: Path, name: str, command: list[str]) -> GpuJob:
    return GpuJob(name=name, command=command, log_path=tmp_path / name / "worker.log")


def test_empty_jobs_returns_empty(logger):
    queue   = GpuQueue(gpus=[0], logger=logger, poll_interval_s=0.0, handle_signals=False)
    results = queue.run([])

    assert results == []


def test_single_job_succeeds(tmp_path, logger):
    queue   = GpuQueue(gpus=[0], logger=logger, poll_interval_s=0.0, handle_signals=False)
    results = queue.run([_job(tmp_path, "a", _ok_command())])

    assert len(results) == 1

    result = results[0]
    assert isinstance(result, GpuJobResult)
    assert result.name       == "a"
    assert result.status     == "DONE"
    assert result.returncode == 0
    assert result.gpu        == 0
    assert result.duration_s >= 0.0


def test_failed_job_reports_failed_status(tmp_path, logger):
    queue   = GpuQueue(gpus=[0], logger=logger, poll_interval_s=0.0, handle_signals=False)
    results = queue.run([_job(tmp_path, "boom", _fail_command(7))])

    result = results[0]
    assert result.status     == "FAILED"
    assert result.returncode == 7


def test_log_file_is_written_with_stdout(tmp_path, logger):
    queue = GpuQueue(gpus=[0], logger=logger, poll_interval_s=0.0, handle_signals=False)
    queue.run([_job(tmp_path, "a", _ok_command("hello_log"))])

    log_text = (tmp_path / "a" / "worker.log").read_text()
    assert "hello_log" in log_text


def test_gpu_flag_is_appended_to_command(tmp_path, logger):
    capture_path = tmp_path / "args.txt"
    command      = [sys.executable, "-c", f"import sys; open(r'{capture_path}','w').write(' '.join(sys.argv[1:]))"]

    queue = GpuQueue(gpus=[3], logger=logger, poll_interval_s=0.0, handle_signals=False)
    queue.run([_job(tmp_path, "a", command)])

    written = capture_path.read_text()
    assert written.endswith("--gpu 3")


def _concurrency_probe_command(counter: Path, peak: Path) -> list[str]:
    body = (
        "import time, fcntl;"
        f"c=r'{counter}'; p=r'{peak}';"
        "open(c,'a').close();"
        "f=open(c,'r+'); fcntl.flock(f, fcntl.LOCK_EX);"
        "n=int(f.read() or '0')+1; f.seek(0); f.truncate(); f.write(str(n)); f.flush();"
        "g=open(p,'a'); g.write(str(n)+'\\n'); g.close();"
        "fcntl.flock(f, fcntl.LOCK_UN); f.close();"
        "time.sleep(0.2);"
        "f=open(c,'r+'); fcntl.flock(f, fcntl.LOCK_EX);"
        "n=int(f.read() or '1')-1; f.seek(0); f.truncate(); f.write(str(n)); f.flush();"
        "fcntl.flock(f, fcntl.LOCK_UN); f.close()"
    )
    return [sys.executable, "-c", body]


def test_capacity_limit_serialises_jobs_on_one_gpu(tmp_path, logger):
    counter = tmp_path / "live.txt"
    peak    = tmp_path / "peak.txt"
    command = _concurrency_probe_command(counter, peak)

    jobs  = [_job(tmp_path, f"j{i}", [*command]) for i in range(3)]
    queue = GpuQueue(gpus=[0], logger=logger, poll_interval_s=0.0, handle_signals=False)
    queue.run(jobs)

    observed = [int(x) for x in peak.read_text().split()]
    assert max(observed) == 1


def test_two_gpus_run_two_jobs_concurrently(tmp_path, logger):
    counter = tmp_path / "live.txt"
    peak    = tmp_path / "peak.txt"
    command = _concurrency_probe_command(counter, peak)

    jobs  = [_job(tmp_path, f"j{i}", [*command]) for i in range(2)]
    queue = GpuQueue(gpus=[0, 1], logger=logger, poll_interval_s=0.0, handle_signals=False)
    queue.run(jobs)

    observed = [int(x) for x in peak.read_text().split()]
    assert max(observed) == 2


def test_device_assignment_uses_lowest_free_gpu_first(tmp_path, logger):
    capture = tmp_path / "gpu_seen.txt"
    body    = f"import sys; open(r'{capture}','a').write(sys.argv[-1]+'\\n')"
    command = [sys.executable, "-c", body]

    queue = GpuQueue(gpus=[2, 5], logger=logger, poll_interval_s=0.0, handle_signals=False)
    queue.run([_job(tmp_path, "solo", [*command])])

    assert capture.read_text().strip() == "2"


def test_released_gpu_is_reused_for_next_job(tmp_path, logger):
    queue   = GpuQueue(gpus=[4], logger=logger, poll_interval_s=0.0, handle_signals=False)
    results = queue.run([_job(tmp_path, f"j{i}", _ok_command()) for i in range(3)])

    assert {r.gpu for r in results} == {4}
    assert len(results) == 3


def test_all_jobs_complete_no_deadlock(tmp_path, logger):
    jobs = [_job(tmp_path, f"j{i}", _ok_command()) for i in range(6)]

    queue   = GpuQueue(gpus=[0, 1], logger=logger, poll_interval_s=0.0, handle_signals=False)
    results = queue.run(jobs)

    assert sorted(r.name for r in results) == sorted(j.name for j in jobs)
    assert all(r.status == "DONE" for r in results)


def test_more_gpus_than_jobs(tmp_path, logger):
    jobs    = [_job(tmp_path, "only", _ok_command())]
    queue   = GpuQueue(gpus=[0, 1, 2, 3], logger=logger, poll_interval_s=0.0, handle_signals=False)
    results = queue.run(jobs)

    assert len(results) == 1
    assert results[0].gpu == 0


def test_result_log_file_matches_job(tmp_path, logger):
    job     = _job(tmp_path, "a", _ok_command())
    queue   = GpuQueue(gpus=[0], logger=logger, poll_interval_s=0.0, handle_signals=False)
    results = queue.run([job])

    assert results[0].log_file == str(job.log_path)


def test_mixed_success_and_failure(tmp_path, logger):
    jobs = [
        _job(tmp_path, "good", _ok_command()),
        _job(tmp_path, "bad",  _fail_command(3)),
    ]

    queue   = GpuQueue(gpus=[0, 1], logger=logger, poll_interval_s=0.0, handle_signals=False)
    results = queue.run(jobs)

    by_name = {r.name: r for r in results}
    assert by_name["good"].status == "DONE"
    assert by_name["bad"].status  == "FAILED"
    assert by_name["bad"].returncode == 3


def test_signal_handlers_restored_after_run(tmp_path):
    import signal

    before_term = signal.getsignal(signal.SIGTERM)
    before_int  = signal.getsignal(signal.SIGINT)

    queue = GpuQueue(gpus=[0], logger=NullLogger(), poll_interval_s=0.0, handle_signals=True)
    queue.run([_job(tmp_path, "a", _ok_command())])

    assert signal.getsignal(signal.SIGTERM) is before_term
    assert signal.getsignal(signal.SIGINT)  is before_int
