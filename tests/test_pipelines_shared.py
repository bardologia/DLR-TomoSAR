from __future__ import annotations

import json
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np
import pytest

from pipelines.shared.io            import FileIO
from pipelines.shared.metadata      import MetadataBase
from pipelines.shared.orchestration import GpuJob, GpuJobResult, GpuQueue, ExperimentStage
from pipelines.shared.plotting      import PlotBase


class RecordingLogger:
    def __init__(self) -> None:
        self.infos       = []
        self.errors      = []
        self.subsections = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def subsection(self, message: str) -> None:
        self.subsections.append(message)


class TestFileIOEnsureDir:
    def test_ensure_dir_creates_nested_path(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"

        result = FileIO.ensure_dir(target)

        assert target.is_dir()
        assert result == target
        assert isinstance(result, Path)

    def test_ensure_dir_is_idempotent(self, tmp_path):
        target = tmp_path / "x"

        FileIO.ensure_dir(target)
        result = FileIO.ensure_dir(target)

        assert target.is_dir()
        assert result == target

    def test_ensure_dir_accepts_string_path(self, tmp_path):
        target = tmp_path / "from_string"

        result = FileIO.ensure_dir(str(target))

        assert target.is_dir()
        assert isinstance(result, Path)

    def test_ensure_dir_existing_directory_preserves_contents(self, tmp_path):
        target = tmp_path / "keep"
        target.mkdir()
        sentinel = target / "file.txt"
        sentinel.write_text("data", encoding="utf-8")

        FileIO.ensure_dir(target)

        assert sentinel.read_text(encoding="utf-8") == "data"


class TestFileIOEnsureDirs:
    def test_ensure_dirs_creates_all_paths(self, tmp_path):
        a = tmp_path / "one"
        b = tmp_path / "two" / "nested"
        c = tmp_path / "three"

        result = FileIO.ensure_dirs(a, b, c)

        assert result is None
        assert a.is_dir()
        assert b.is_dir()
        assert c.is_dir()

    def test_ensure_dirs_with_no_arguments_is_noop(self):
        assert FileIO.ensure_dirs() is None

    def test_ensure_dirs_idempotent(self, tmp_path):
        a = tmp_path / "dir_a"

        FileIO.ensure_dirs(a)
        FileIO.ensure_dirs(a)

        assert a.is_dir()


class TestFileIOJson:
    def test_save_json_writes_readable_file(self, tmp_path):
        payload = {"name": "run", "count": 3, "values": [1, 2, 3]}
        path    = tmp_path / "out.json"

        result = FileIO.save_json(payload, path)

        assert result == path
        assert path.is_file()
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == payload

    def test_save_json_creates_parent_directory(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "out.json"

        FileIO.save_json({"a": 1}, path)

        assert path.is_file()

    def test_save_json_honors_indent(self, tmp_path):
        path = tmp_path / "indented.json"

        FileIO.save_json({"a": 1}, path, indent=2)

        text = path.read_text(encoding="utf-8")
        assert "\n  " in text

    def test_save_json_serializes_non_native_via_default_str(self, tmp_path):
        path    = tmp_path / "paths.json"
        payload = {"location": Path("/tmp/example")}

        FileIO.save_json(payload, path)

        loaded = FileIO.load_json(path)
        assert loaded["location"] == "/tmp/example"

    def test_save_then_load_json_round_trip(self, tmp_path):
        payload = {"nested": {"x": [1, 2], "y": True}, "z": None}
        path    = tmp_path / "rt.json"

        FileIO.save_json(payload, path)
        loaded = FileIO.load_json(path)

        assert loaded == payload

    def test_load_json_returns_list_payload(self, tmp_path):
        path = tmp_path / "list.json"
        FileIO.save_json([{"a": 1}, {"b": 2}], path)

        loaded = FileIO.load_json(path)

        assert loaded == [{"a": 1}, {"b": 2}]

    def test_load_json_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FileIO.load_json(tmp_path / "absent.json")

    def test_load_json_invalid_content_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            FileIO.load_json(path)


class TestFileIOTextMetadata:
    def test_save_text_metadata_writes_key_value_lines(self, tmp_path):
        entries = {"alpha": 1, "beta": "two", "gamma": 3.5}
        path    = tmp_path / "meta.txt"

        result = FileIO.save_text_metadata(entries, path)

        assert result == path
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines == ["alpha: 1", "beta: two", "gamma: 3.5"]

    def test_save_text_metadata_creates_parent_directory(self, tmp_path):
        path = tmp_path / "sub" / "meta.txt"

        FileIO.save_text_metadata({"k": "v"}, path)

        assert path.is_file()

    def test_save_text_metadata_empty_entries_writes_empty_file(self, tmp_path):
        path = tmp_path / "empty.txt"

        FileIO.save_text_metadata({}, path)

        assert path.is_file()
        assert path.read_text(encoding="utf-8") == ""

    def test_save_text_metadata_preserves_insertion_order(self, tmp_path):
        path    = tmp_path / "ordered.txt"
        entries = {"z": 1, "a": 2, "m": 3}

        FileIO.save_text_metadata(entries, path)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines == ["z: 1", "a: 2", "m: 3"]


@dataclass
class DummyConfig:
    value: int = 7


class TestMetadataBase:
    def test_init_stores_config_and_logger(self):
        config = DummyConfig()
        logger = RecordingLogger()

        meta = MetadataBase(config, logger)

        assert meta.config is config
        assert meta.logger is logger

    def test_init_logger_defaults_to_none(self):
        meta = MetadataBase(DummyConfig())

        assert meta.logger is None

    def test_timestamp_format_is_fixed_length(self):
        stamp = MetadataBase.timestamp()

        assert len(stamp) == len("YYYYMMDD_HHMMSS")
        date_part, time_part = stamp.split("_")
        assert date_part.isdigit() and len(date_part) == 8
        assert time_part.isdigit() and len(time_part) == 6

    def test_save_json_writes_file_and_returns_path(self, tmp_path):
        meta = MetadataBase(DummyConfig())
        path = tmp_path / "meta.json"

        out_path = meta._save_json({"a": 1}, path)

        assert out_path == path
        assert FileIO.load_json(path) == {"a": 1}

    def test_save_json_logs_subsection_when_message_and_logger_present(self, tmp_path):
        logger = RecordingLogger()
        meta   = MetadataBase(DummyConfig(), logger)
        path   = tmp_path / "meta.json"

        meta._save_json({"a": 1}, path, message="Saved metadata")

        assert len(logger.subsections) == 1
        assert "Saved metadata" in logger.subsections[0]
        assert str(path) in logger.subsections[0]

    def test_save_json_no_log_without_message(self, tmp_path):
        logger = RecordingLogger()
        meta   = MetadataBase(DummyConfig(), logger)

        meta._save_json({"a": 1}, tmp_path / "m.json")

        assert logger.subsections == []

    def test_save_json_no_log_without_logger(self, tmp_path):
        meta = MetadataBase(DummyConfig(), logger=None)

        out_path = meta._save_json({"a": 1}, tmp_path / "m.json", message="ignored")

        assert out_path.is_file()

    def test_save_text_writes_file_and_returns_path(self, tmp_path):
        meta = MetadataBase(DummyConfig())
        path = tmp_path / "meta.txt"

        out_path = meta._save_text({"k": "v"}, path)

        assert out_path == path
        assert path.read_text(encoding="utf-8") == "k: v\n"

    def test_save_text_logs_subsection_when_message_and_logger_present(self, tmp_path):
        logger = RecordingLogger()
        meta   = MetadataBase(DummyConfig(), logger)
        path   = tmp_path / "meta.txt"

        meta._save_text({"k": "v"}, path, message="Wrote text")

        assert len(logger.subsections) == 1
        assert "Wrote text" in logger.subsections[0]

    def test_save_text_no_log_without_message(self, tmp_path):
        logger = RecordingLogger()
        meta   = MetadataBase(DummyConfig(), logger)

        meta._save_text({"k": "v"}, tmp_path / "m.txt")

        assert logger.subsections == []


class TestGpuJobDataclasses:
    def test_gpu_job_fields(self):
        job = GpuJob(name="train", command=["python", "run.py"], log_path=Path("/tmp/log"))

        assert job.name == "train"
        assert job.command == ["python", "run.py"]
        assert job.log_path == Path("/tmp/log")

    def test_gpu_job_result_fields_and_asdict(self):
        result = GpuJobResult(
            name       = "train",
            gpu        = 0,
            status     = "DONE",
            returncode = 0,
            duration_s = 1.5,
            log_file   = "/tmp/log",
        )

        as_dict = asdict(result)
        assert is_dataclass(result)
        assert as_dict["name"] == "train"
        assert as_dict["gpu"] == 0
        assert as_dict["status"] == "DONE"
        assert as_dict["returncode"] == 0
        assert as_dict["duration_s"] == 1.5
        assert as_dict["log_file"] == "/tmp/log"


class FakeProcess:
    def __init__(self, returncode: int = 0, poll_after: int = 0) -> None:
        self._returncode = returncode
        self._poll_after = poll_after
        self._polls      = 0

    def poll(self):
        if self._polls >= self._poll_after:
            return self._returncode
        self._polls += 1
        return None

    @property
    def returncode(self):
        return self._returncode


class TestGpuQueue:
    def test_init_copies_gpu_list(self):
        gpus   = [0, 1]
        logger = RecordingLogger()

        queue = GpuQueue(gpus=gpus, logger=logger, poll_interval_s=0.0)

        assert queue.gpus == [0, 1]
        assert queue.gpus is not gpus
        assert queue.poll_interval_s == 0.0
        assert queue.logger is logger

    def _patch_subprocess(self, monkeypatch, returncode_by_name):
        launched = []

        def fake_popen(command, stdout=None, stderr=None):
            rc = 0
            for key, value in returncode_by_name.items():
                if key in command:
                    rc = value
            launched.append(list(command))
            return FakeProcess(returncode=rc, poll_after=0)

        import pipelines.shared.orchestration as orch
        monkeypatch.setattr(orch.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(orch.time, "sleep", lambda *_: None)
        return launched

    def test_run_executes_single_successful_job(self, monkeypatch, tmp_path):
        launched = self._patch_subprocess(monkeypatch, {})
        logger   = RecordingLogger()
        queue    = GpuQueue(gpus=[0], logger=logger, poll_interval_s=0.0)
        job      = GpuJob(name="solo", command=["python", "run.py"], log_path=tmp_path / "solo.log")

        results = queue.run([job])

        assert len(results) == 1
        assert results[0].name == "solo"
        assert results[0].gpu == 0
        assert results[0].status == "DONE"
        assert results[0].returncode == 0
        assert results[0].log_file == str(tmp_path / "solo.log")
        assert ["--gpu", "0"] == launched[0][-2:]

    def test_run_appends_gpu_argument(self, monkeypatch, tmp_path):
        launched = self._patch_subprocess(monkeypatch, {})
        queue    = GpuQueue(gpus=[3], logger=RecordingLogger(), poll_interval_s=0.0)
        job      = GpuJob(name="j", command=["cmd"], log_path=tmp_path / "j.log")

        queue.run([job])

        assert launched[0] == ["cmd", "--gpu", "3"]

    def test_run_creates_log_parent_directory(self, monkeypatch, tmp_path):
        self._patch_subprocess(monkeypatch, {})
        log_path = tmp_path / "nested" / "logs" / "job.log"
        queue    = GpuQueue(gpus=[0], logger=RecordingLogger(), poll_interval_s=0.0)
        job      = GpuJob(name="j", command=["cmd"], log_path=log_path)

        queue.run([job])

        assert log_path.parent.is_dir()
        assert log_path.is_file()

    def test_run_failed_job_reports_failed_status(self, monkeypatch, tmp_path):
        self._patch_subprocess(monkeypatch, {"failing.py": 2})
        logger = RecordingLogger()
        queue  = GpuQueue(gpus=[0], logger=logger, poll_interval_s=0.0)
        job    = GpuJob(name="bad", command=["python", "failing.py"], log_path=tmp_path / "bad.log")

        results = queue.run([job])

        assert results[0].status == "FAILED"
        assert results[0].returncode == 2
        assert len(logger.errors) == 1

    def test_run_multiple_jobs_more_than_gpus(self, monkeypatch, tmp_path):
        self._patch_subprocess(monkeypatch, {})
        queue = GpuQueue(gpus=[0], logger=RecordingLogger(), poll_interval_s=0.0)
        jobs  = [
            GpuJob(name=f"j{i}", command=["cmd", f"task{i}"], log_path=tmp_path / f"j{i}.log")
            for i in range(4)
        ]

        results = queue.run(jobs)

        names = {r.name for r in results}
        assert len(results) == 4
        assert names == {"j0", "j1", "j2", "j3"}

    def test_run_distributes_across_multiple_gpus(self, monkeypatch, tmp_path):
        self._patch_subprocess(monkeypatch, {})
        queue = GpuQueue(gpus=[0, 1], logger=RecordingLogger(), poll_interval_s=0.0)
        jobs  = [
            GpuJob(name=f"j{i}", command=["cmd"], log_path=tmp_path / f"j{i}.log")
            for i in range(2)
        ]

        results = queue.run(jobs)
        used    = {r.gpu for r in results}

        assert used == {0, 1}

    def test_run_empty_job_list_returns_empty(self, monkeypatch):
        self._patch_subprocess(monkeypatch, {})
        queue = GpuQueue(gpus=[0], logger=RecordingLogger(), poll_interval_s=0.0)

        results = queue.run([])

        assert results == []

    def test_run_returns_gpu_to_pool_after_completion(self, monkeypatch, tmp_path):
        launched = self._patch_subprocess(monkeypatch, {})
        queue    = GpuQueue(gpus=[5], logger=RecordingLogger(), poll_interval_s=0.0)
        jobs     = [
            GpuJob(name="first", command=["a"], log_path=tmp_path / "a.log"),
            GpuJob(name="second", command=["b"], log_path=tmp_path / "b.log"),
        ]

        results = queue.run(jobs)

        assert all(r.gpu == 5 for r in results)
        assert len(results) == 2


@dataclass
class PathsConfig:
    log_base_dir: str


@dataclass
class StageConfig:
    gpus            : list
    poll_interval_s : float
    paths           : PathsConfig


class TestExperimentStage:
    def _make_config(self, tmp_path):
        return StageConfig(gpus=[0, 1], poll_interval_s=0.0, paths=PathsConfig(log_base_dir=str(tmp_path)))

    def test_init_builds_run_dir(self, tmp_path):
        config = self._make_config(tmp_path)
        logger = RecordingLogger()

        stage = ExperimentStage(config, run_tag="exp01", logger=logger, entry_script=Path("run.py"))

        assert stage.run_tag == "exp01"
        assert stage.run_dir == Path(str(tmp_path)) / "exp01"
        assert stage.entry_script == Path("run.py")
        assert stage.logger is logger

    def test_init_entry_script_defaults_to_none(self, tmp_path):
        stage = ExperimentStage(self._make_config(tmp_path), run_tag="t", logger=RecordingLogger())

        assert stage.entry_script is None

    def test_run_queue_returns_dicts(self, monkeypatch, tmp_path):
        def fake_popen(command, stdout=None, stderr=None):
            return FakeProcess(returncode=0, poll_after=0)

        import pipelines.shared.orchestration as orch
        monkeypatch.setattr(orch.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(orch.time, "sleep", lambda *_: None)

        stage = ExperimentStage(self._make_config(tmp_path), run_tag="t", logger=RecordingLogger())
        jobs  = [GpuJob(name="j", command=["cmd"], log_path=tmp_path / "j.log")]

        results = stage._run_queue(jobs)

        assert isinstance(results, list)
        assert isinstance(results[0], dict)
        assert results[0]["name"] == "j"
        assert results[0]["status"] == "DONE"

    def test_order_results_orders_by_names(self, tmp_path):
        stage   = ExperimentStage(self._make_config(tmp_path), run_tag="t", logger=RecordingLogger())
        results = [{"name": "b"}, {"name": "a"}, {"name": "c"}]

        ordered = stage._order_results(results, names=["a", "b", "c"])

        assert [r["name"] for r in ordered] == ["a", "b", "c"]

    def test_order_results_skips_missing_names(self, tmp_path):
        stage   = ExperimentStage(self._make_config(tmp_path), run_tag="t", logger=RecordingLogger())
        results = [{"name": "a"}]

        ordered = stage._order_results(results, names=["a", "missing"])

        assert [r["name"] for r in ordered] == ["a"]

    def test_order_results_empty(self, tmp_path):
        stage = ExperimentStage(self._make_config(tmp_path), run_tag="t", logger=RecordingLogger())

        assert stage._order_results([], names=["a"]) == []

    def test_write_results_saves_json(self, tmp_path):
        stage   = ExperimentStage(self._make_config(tmp_path), run_tag="t", logger=RecordingLogger())
        path    = tmp_path / "results.json"
        results = [{"name": "a", "status": "DONE"}]

        stage._write_results(results, path)

        assert FileIO.load_json(path) == results

    def test_log_failures_logs_each_with_log_hint(self, tmp_path):
        logger = RecordingLogger()
        stage  = ExperimentStage(self._make_config(tmp_path), run_tag="t", logger=logger)
        failed = [{"name": "x", "log_file": "/tmp/x.log"}, {"name": "y", "log_file": "/tmp/y.log"}]

        stage._log_failures(failed)

        assert len(logger.errors) == 2
        assert "FAILED  x" in logger.errors[0]
        assert "/tmp/x.log" in logger.errors[0]

    def test_log_failures_without_log_file_omits_hint(self, tmp_path):
        logger = RecordingLogger()
        stage  = ExperimentStage(self._make_config(tmp_path), run_tag="t", logger=logger)
        failed = [{"name": "x", "log_file": ""}]

        stage._log_failures(failed)

        assert len(logger.errors) == 1
        assert "see" not in logger.errors[0]

    def test_log_failures_custom_name_key(self, tmp_path):
        logger = RecordingLogger()
        stage  = ExperimentStage(self._make_config(tmp_path), run_tag="t", logger=logger)
        failed = [{"job": "z", "log_file": ""}]

        stage._log_failures(failed, name_key="job")

        assert "FAILED  z" in logger.errors[0]

    def test_log_failures_empty(self, tmp_path):
        logger = RecordingLogger()
        stage  = ExperimentStage(self._make_config(tmp_path), run_tag="t", logger=logger)

        stage._log_failures([])

        assert logger.errors == []


class StyledPlot(PlotBase):
    pass


class TestPlotBaseStyle:
    def test_apply_style_sets_rcparams(self):
        plot = StyledPlot()

        plot._apply_style()

        assert plt.rcParams["font.family"] == ["serif"]
        assert plt.rcParams["figure.dpi"] == StyledPlot.fig_dpi
        assert plt.rcParams["savefig.dpi"] == StyledPlot.save_dpi

    def test_scientific_rc_contains_expected_keys(self):
        rc = PlotBase.SCIENTIFIC_RC

        assert rc["font.size"] == 11
        assert rc["pdf.fonttype"] == 42
        assert rc["savefig.bbox"] == "tight"


class TestPlotBaseSave:
    def test_save_writes_figure_and_creates_parents(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = tmp_path / "figs" / "plot.png"

        result = PlotBase._save(fig, path)

        assert result == path
        assert path.is_file()
        assert path.stat().st_size > 0

    def test_save_closes_figure(self, tmp_path):
        fig, _ = plt.subplots()
        path   = tmp_path / "p.png"
        num    = fig.number

        PlotBase._save(fig, path)

        assert not plt.fignum_exists(num)


class TestPlotBaseSharedClim:
    def test_shared_clim_returns_percentiles(self):
        arr = np.linspace(0.0, 100.0, 101)

        lo, hi = PlotBase._shared_clim(arr, q_low=1.0, q_high=99.0)

        assert np.isclose(lo, 1.0)
        assert np.isclose(hi, 99.0)
        assert lo < hi

    def test_shared_clim_concatenates_multiple_arrays(self):
        a = np.array([0.0, 10.0])
        b = np.array([20.0, 30.0])

        lo, hi = PlotBase._shared_clim(a, b, q_low=0.0, q_high=100.0)

        assert np.isclose(lo, 0.0)
        assert np.isclose(hi, 30.0)

    def test_shared_clim_ignores_non_finite(self):
        arr = np.array([0.0, np.nan, np.inf, -np.inf, 50.0, 100.0])

        lo, hi = PlotBase._shared_clim(arr, q_low=0.0, q_high=100.0)

        assert np.isfinite(lo) and np.isfinite(hi)
        assert np.isclose(lo, 0.0)
        assert np.isclose(hi, 100.0)

    def test_shared_clim_all_non_finite_returns_default(self):
        arr = np.array([np.nan, np.inf, -np.inf])

        result = PlotBase._shared_clim(arr)

        assert result == (0.0, 1.0)

    def test_shared_clim_returns_python_floats(self):
        lo, hi = PlotBase._shared_clim(np.array([1.0, 2.0, 3.0]))

        assert isinstance(lo, float)
        assert isinstance(hi, float)


class TestPlotBaseCmapWithBad:
    def test_cmap_with_bad_sets_bad_color(self):
        cmap = PlotBase._cmap_with_bad("viridis", bad_color="0.5")

        rgba = cmap(np.nan)
        assert len(rgba) == 4
        assert np.allclose(rgba[:3], 0.5)

    def test_cmap_with_bad_returns_colormap(self):
        import matplotlib.colors as mcolors

        cmap = PlotBase._cmap_with_bad("plasma")

        assert isinstance(cmap, mcolors.Colormap)


class TestPlotBaseNormalize01:
    def test_normalize_01_maps_to_unit_range(self):
        arr = np.array([2.0, 4.0, 6.0])

        out = PlotBase._normalize_01(arr)

        assert np.isclose(out.min(), 0.0)
        assert np.isclose(out.max(), 1.0)
        assert out.dtype == np.float32

    def test_normalize_01_constant_array_returns_zeros(self):
        arr = np.full(5, 3.0)

        out = PlotBase._normalize_01(arr)

        assert np.all(out == 0.0)
        assert out.dtype == np.float32

    def test_normalize_01_handles_nan_via_nanmin_nanmax(self):
        arr = np.array([0.0, np.nan, 10.0])

        out = PlotBase._normalize_01(arr)

        assert np.isclose(out[0], 0.0)
        assert np.isclose(out[2], 1.0)

    def test_normalize_01_preserves_shape(self):
        arr = np.arange(12.0).reshape(3, 4)

        out = PlotBase._normalize_01(arr)

        assert out.shape == (3, 4)


class TestPlotBaseTriplePanel:
    def test_triple_panel_populates_axes(self):
        fig, axes = plt.subplots(1, 3)
        data      = np.arange(9.0).reshape(3, 3)
        err       = np.ones((3, 3))
        panels    = [
            (data, "Reference", "viridis", 0.0, 8.0),
            (data, "Prediction", "viridis", 0.0, 8.0),
            (err,  "Error", "magma", 0.0, 1.0),
        ]

        PlotBase._triple_panel(
            fig,
            axes,
            panels    = panels,
            x_label   = "range",
            int_label = "intensity",
            extent    = [0, 3, 0, 3],
            origin    = "lower",
        )

        assert axes[0].get_title() == "Reference"
        assert axes[1].get_title() == "Prediction"
        assert axes[2].get_title() == "Error"
        assert axes[0].get_xlabel() == "range"
        assert len(axes[0].images) == 1
        plt.close(fig)

    def test_triple_panel_error_panel_uses_error_label(self):
        fig, axes = plt.subplots(1, 2)
        ref       = np.zeros((2, 2))
        err       = np.ones((2, 2))
        panels    = [
            (ref, "Reference", "viridis", 0.0, 1.0),
            (err, "Error", "magma", 0.0, 1.0),
        ]

        PlotBase._triple_panel(
            fig,
            axes,
            panels    = panels,
            x_label   = "x",
            int_label = "value",
            extent    = [0, 2, 0, 2],
            origin    = "upper",
        )

        assert len(axes[0].images) == 1
        assert len(axes[1].images) == 1
        plt.close(fig)

    def test_triple_panel_respects_origin(self):
        fig, axes = plt.subplots(1, 1)
        data      = np.arange(4.0).reshape(2, 2)
        panels    = [(data, "Only", "viridis", 0.0, 3.0)]

        PlotBase._triple_panel(
            fig,
            [axes],
            panels    = panels,
            x_label   = "x",
            int_label = "v",
            extent    = [0, 2, 0, 2],
            origin    = "lower",
        )

        assert axes.images[0].origin == "lower"
        plt.close(fig)
