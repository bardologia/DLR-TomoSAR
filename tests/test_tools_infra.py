from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.markdown import MarkdownDoc, MarkdownTable
from tools.tracker import NullTracker, Tracker
from tools.logger import Logger, LiveMonitor, NullLogger, get_console
from tools.resource_monitor import ResourceMonitor


class FakeWriter:
    def __init__(self):
        self.scalars      = []
        self.histograms   = []
        self.images       = []
        self.image_groups = []
        self.figures      = []
        self.texts        = []
        self.pr_curves    = []
        self.hparams      = []
        self.graphs       = []
        self.flushed      = 0
        self.closed       = 0

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_histogram(self, tag, values, step, bins="auto"):
        self.histograms.append((tag, values, step, bins))

    def add_image(self, tag, img, step, dataformats="CHW"):
        self.images.append((tag, img, step, dataformats))

    def add_images(self, tag, imgs, step, dataformats="NCHW"):
        self.image_groups.append((tag, imgs, step, dataformats))

    def add_figure(self, tag, fig, step, close=True):
        self.figures.append((tag, fig, step, close))

    def add_text(self, tag, text, step):
        self.texts.append((tag, text, step))

    def add_pr_curve(self, tag, labels, predictions, step):
        self.pr_curves.append((tag, labels, predictions, step))

    def add_hparams(self, hparams, metrics):
        self.hparams.append((hparams, metrics))

    def add_graph(self, model, input_to_model):
        self.graphs.append((model, input_to_model))

    def flush(self):
        self.flushed += 1

    def close(self):
        self.closed += 1


class TestMarkdownTable:
    def test_columns_coerced_to_str(self):
        table = MarkdownTable([1, 2, 3])
        assert table.columns == ["1", "2", "3"]

    def test_default_align_is_all_left(self):
        table = MarkdownTable(["a", "b"])
        assert table.align == ["left", "left"]

    def test_custom_align_preserved(self):
        table = MarkdownTable(["a", "b"], align=["right", "center"])
        assert table.align == ["right", "center"]

    def test_is_empty_true_when_no_rows(self):
        assert MarkdownTable(["a"]).is_empty()

    def test_is_empty_false_after_add_row(self):
        table = MarkdownTable(["a"])
        table.add_row("x")
        assert not table.is_empty()

    def test_add_row_returns_self(self):
        table = MarkdownTable(["a"])
        assert table.add_row("x") is table

    def test_add_row_pads_missing_cells_with_empty(self):
        table = MarkdownTable(["a", "b", "c"])
        table.add_row("x")
        assert table.rows[0] == ["x", MarkdownTable.EMPTY, MarkdownTable.EMPTY]

    def test_add_row_none_cell_becomes_empty(self):
        table = MarkdownTable(["a", "b"])
        table.add_row("x", None)
        assert table.rows[0] == ["x", MarkdownTable.EMPTY]

    def test_add_row_coerces_cells_to_str(self):
        table = MarkdownTable(["a"])
        table.add_row(42)
        assert table.rows[0] == ["42"]

    def test_add_rows_appends_all(self):
        table = MarkdownTable(["a", "b"])
        ret = table.add_rows([("1", "2"), ("3", "4")])
        assert ret is table
        assert len(table.rows) == 2

    def test_render_produces_header_separator_and_rows(self):
        table = MarkdownTable(["a", "b"])
        table.add_row("1", "2")
        lines = table.render()
        assert len(lines) == 3
        assert lines[0].startswith("|")
        assert set(lines[1].replace("|", "").replace(":", "").replace("-", "").strip()) == set()

    def test_render_min_column_width_is_three(self):
        table = MarkdownTable(["a"])
        lines = table.render()
        assert "a  " in lines[0]

    def test_render_widths_expand_to_longest_cell(self):
        table = MarkdownTable(["a"])
        table.add_row("longvalue")
        lines = table.render()
        assert "longvalue" in lines[2]
        assert "a        " in lines[0]

    def test_separator_left_align(self):
        table = MarkdownTable(["aaaa"], align=["left"])
        sep = table._separator(4, "left")
        assert sep == "----"

    def test_separator_right_align_ends_with_colon(self):
        table = MarkdownTable(["aaaa"], align=["right"])
        sep = table._separator(4, "right")
        assert sep == "---:"

    def test_separator_center_align_wrapped_in_colons(self):
        table = MarkdownTable(["aaaa"], align=["center"])
        sep = table._separator(4, "center")
        assert sep == ":--:"

    def test_aligned_left(self):
        table = MarkdownTable(["a"])
        assert table._aligned("x", 5, "left") == "x    "

    def test_aligned_right(self):
        table = MarkdownTable(["a"])
        assert table._aligned("x", 5, "right") == "    x"

    def test_aligned_center(self):
        table = MarkdownTable(["a"])
        assert table._aligned("x", 5, "center") == "  x  "

    def test_render_empty_table_has_header_and_separator_only(self):
        table = MarkdownTable(["a", "b"])
        lines = table.render()
        assert len(lines) == 2


class TestMarkdownDoc:
    def test_no_title_starts_empty(self):
        doc = MarkdownDoc()
        assert doc.lines == []

    def test_title_creates_level_one_heading(self):
        doc = MarkdownDoc("My Title")
        assert doc.lines[0] == "# My Title"

    def test_heading_levels(self):
        doc = MarkdownDoc()
        doc.heading("Sub", level=2)
        assert "## Sub" in doc.lines

    def test_heading_inserts_blank_before_when_not_first(self):
        doc = MarkdownDoc()
        doc.paragraph("text")
        doc.heading("Next", level=2)
        idx = doc.lines.index("## Next")
        assert doc.lines[idx - 1] == ""

    def test_paragraph_adds_text_and_blank(self):
        doc = MarkdownDoc()
        doc.paragraph("hello")
        assert doc.lines == ["hello", ""]

    def test_paragraph_coerces_to_str(self):
        doc = MarkdownDoc()
        doc.paragraph(123)
        assert doc.lines[0] == "123"

    def test_raw_appends_single_line(self):
        doc = MarkdownDoc()
        doc.raw("raw text")
        assert doc.lines == ["raw text"]

    def test_blank_appends_empty_string(self):
        doc = MarkdownDoc()
        doc.blank()
        assert doc.lines == [""]

    def test_bold_kv_formats_key_value(self):
        doc = MarkdownDoc()
        doc.bold_kv("speed", 5)
        assert doc.lines == ["**speed:** `5`"]

    def test_image_formats_markdown_image(self):
        doc = MarkdownDoc()
        doc.image("alt text", "img.png")
        assert doc.lines[0] == "![alt text](img.png)"

    def test_kv_table_from_mapping_with_code_keys(self):
        doc = MarkdownDoc()
        doc.kv_table({"lr": 0.1})
        rendered = doc.render()
        assert "`lr`" in rendered
        assert "0.1" in rendered

    def test_kv_table_without_code_keys(self):
        doc = MarkdownDoc()
        doc.kv_table({"lr": 0.1}, code_keys=False)
        rendered = doc.render()
        assert "`lr`" not in rendered
        assert "lr" in rendered

    def test_kv_table_from_iterable_of_tuples(self):
        doc = MarkdownDoc()
        doc.kv_table([("a", 1), ("b", 2)])
        rendered = doc.render()
        assert "`a`" in rendered
        assert "`b`" in rendered

    def test_table_extends_lines_and_appends_blank(self):
        doc = MarkdownDoc()
        table = MarkdownTable(["a"])
        table.add_row("1")
        doc.table(table)
        assert doc.lines[-1] == ""

    def test_chaining_returns_self(self):
        doc = MarkdownDoc()
        assert doc.heading("h").paragraph("p").blank().raw("r") is doc

    def test_render_ends_with_single_newline(self):
        doc = MarkdownDoc("Title")
        out = doc.render()
        assert out.endswith("\n")
        assert not out.endswith("\n\n")

    def test_render_empty_doc(self):
        doc = MarkdownDoc()
        assert doc.render() == "\n"

    def test_save_writes_file_and_returns_path(self, tmp_path):
        doc = MarkdownDoc("Report")
        target = tmp_path / "sub" / "report.md"
        result = doc.save(target)
        assert result == target
        assert target.exists()
        assert target.read_text(encoding="utf-8") == doc.render()

    def test_save_creates_parent_dirs(self, tmp_path):
        doc = MarkdownDoc("X")
        target = tmp_path / "a" / "b" / "c.md"
        doc.save(target)
        assert target.parent.is_dir()


class TestTracker:
    def test_inactive_when_no_writer(self):
        tracker = Tracker(writer=None)
        assert not tracker.active

    def test_active_with_writer(self):
        tracker = Tracker(writer=FakeWriter())
        assert tracker.active

    def test_initial_step_is_zero(self):
        assert Tracker()._step == 0

    def test_set_step_coerces_to_int(self):
        tracker = Tracker()
        tracker.set_step(7.9)
        assert tracker._step == 7

    def test_advance_default_increments_by_one(self):
        tracker = Tracker()
        assert tracker.advance() == 1
        assert tracker._step == 1

    def test_advance_by_n(self):
        tracker = Tracker()
        tracker.set_step(10)
        assert tracker.advance(5) == 15

    def test_scope_builds_nested_tag(self):
        tracker = Tracker()
        with tracker.scope("outer"):
            with tracker.scope("inner"):
                assert tracker._tag("loss") == "outer/inner/loss"

    def test_scope_pops_on_exit(self):
        tracker = Tracker()
        with tracker.scope("outer"):
            pass
        assert tracker._scopes == []

    def test_scope_pops_on_exception(self):
        tracker = Tracker()
        with pytest.raises(RuntimeError):
            with tracker.scope("outer"):
                raise RuntimeError("boom")
        assert tracker._scopes == []

    def test_tag_without_scope_is_plain(self):
        tracker = Tracker()
        assert tracker._tag("loss") == "loss"

    def test_resolve_uses_current_step_when_none(self):
        tracker = Tracker()
        tracker.set_step(4)
        assert tracker._resolve(None) == 4

    def test_resolve_uses_explicit_step(self):
        tracker = Tracker()
        tracker.set_step(4)
        assert tracker._resolve(9) == 9

    def test_log_scalar_emits_float(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        tracker.set_step(3)
        tracker.log_scalar("loss", 1, step=None)
        assert writer.scalars == [("loss", 1.0, 3)]
        assert isinstance(writer.scalars[0][1], float)

    def test_log_scalar_no_writer_is_noop(self):
        tracker = Tracker(writer=None)
        tracker.log_scalar("loss", 1.0)

    def test_log_scalar_respects_scope_tag(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        with tracker.scope("train"):
            tracker.log_scalar("loss", 0.5, step=2)
        assert writer.scalars[0][0] == "train/loss"

    def test_log_metrics_emits_each_key(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        tracker.log_metrics("phase", {"a": 1.0, "b": 2.0}, step=0)
        tags = {s[0] for s in writer.scalars}
        assert tags == {"phase/a", "phase/b"}

    def test_log_metrics_skips_non_numeric(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        tracker.log_metrics("phase", {"good": 1.0, "bad": "text"}, step=0)
        tags = {s[0] for s in writer.scalars}
        assert tags == {"phase/good"}

    def test_log_histogram_converts_to_float32_array(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        tracker.log_histogram("h", [[1, 2], [3, 4]], step=1)
        tag, values, step, bins = writer.histograms[0]
        assert values.dtype == np.float32
        assert values.ndim == 1
        assert values.shape == (4,)

    def test_log_histogram_no_writer_is_noop(self):
        tracker = Tracker(writer=None)
        tracker.log_histogram("h", [1, 2, 3])

    def test_log_image_emits_array(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        img     = np.zeros((3, 4, 4))
        tracker.log_image("img", img, step=0)
        assert writer.images[0][0] == "img"
        assert writer.images[0][3] == "CHW"

    def test_log_images_default_dataformat(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        tracker.log_images("imgs", np.zeros((2, 3, 4, 4)), step=0)
        assert writer.image_groups[0][3] == "NCHW"

    def test_log_text_coerces_to_str(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        tracker.log_text("t", 123, step=0)
        assert writer.texts[0][1] == "123"

    def test_log_pr_curve_emits(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        tracker.log_pr_curve("pr", [0, 1], [0.2, 0.8], step=5)
        assert writer.pr_curves[0][0] == "pr"
        assert writer.pr_curves[0][3] == 5

    def test_log_pr_curve_no_writer_is_noop(self):
        tracker = Tracker(writer=None)
        tracker.log_pr_curve("pr", [0], [0.5])

    def test_log_hparams_cleans_and_emits(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        tracker.log_hparams({"opt": object(), "lr": 0.1}, {"acc": 0.9, "name": "skip"})
        hparams, metrics = writer.hparams[0]
        assert isinstance(hparams["opt"], str)
        assert hparams["lr"] == 0.1
        assert metrics == {"acc": 0.9}

    def test_log_hparams_no_writer_is_noop(self):
        tracker = Tracker(writer=None)
        tracker.log_hparams({"a": 1}, {"m": 1.0})

    def test_log_graph_swallows_writer_exception(self):
        class Boom(FakeWriter):
            def add_graph(self, model, input_to_model):
                raise RuntimeError("fail")

        tracker = Tracker(writer=Boom())
        tracker.log_graph(object(), object())

    def test_log_param_stats_emits_five_stats(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        tracker.log_param_stats("w", np.array([1.0, 2.0, 3.0]), step=0)
        tags = {s[0] for s in writer.scalars}
        assert tags == {"w/mean", "w/std", "w/min", "w/max", "w/norm"}

    def test_log_param_stats_empty_tensor_is_noop(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        tracker.log_param_stats("w", np.array([]), step=0)
        assert writer.scalars == []

    def test_log_param_stats_values_correct(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        tracker.log_param_stats("w", np.array([0.0, 4.0]), step=0)
        stats = {s[0].split("/")[1]: s[1] for s in writer.scalars}
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["min"] == pytest.approx(0.0)
        assert stats["max"] == pytest.approx(4.0)

    def test_log_memory_no_writer_is_noop(self):
        tracker = Tracker(writer=None)
        tracker.log_memory()

    def test_log_memory_emits_only_when_cuda_available(self):
        import torch

        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        tracker.log_memory()
        if torch.cuda.is_available():
            tags = {s[0] for s in writer.scalars}
            assert "system/gpu_mem_alloc_GB" in tags
        else:
            assert writer.scalars == []

    def test_flush_delegates_to_writer(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        tracker.flush()
        assert writer.flushed == 1

    def test_flush_no_writer_is_noop(self):
        Tracker(writer=None).flush()

    def test_close_delegates_to_writer(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        tracker.close()
        assert writer.closed == 1

    def test_close_no_writer_is_noop(self):
        Tracker(writer=None).close()

    def test_log_figure_no_writer_closes_figure(self):
        plt     = pytest.importorskip("matplotlib.pyplot")
        tracker = Tracker(writer=None)
        fig     = plt.figure()
        tracker.log_figure("f", fig, close=True)
        assert not plt.fignum_exists(fig.number)

    def test_log_figure_with_writer_emits(self):
        writer  = FakeWriter()
        tracker = Tracker(writer=writer)
        sentinel = object()
        tracker.log_figure("f", sentinel, step=1, close=False)
        assert writer.figures[0][0] == "f"
        assert writer.figures[0][1] is sentinel


class TestNullTracker:
    def test_is_inactive(self):
        assert not NullTracker().active

    def test_log_calls_are_noops(self):
        tracker = NullTracker()
        tracker.log_scalar("a", 1.0)
        tracker.log_metrics("p", {"x": 1.0})
        tracker.flush()
        tracker.close()

    def test_debug_flag_preserved(self):
        assert NullTracker(debug=True).debug is True


class TestNullLogger:
    def test_any_attribute_is_callable_noop(self):
        log = NullLogger()
        assert log.info("x") is None
        assert log.section("y") is None
        assert log.anything_at_all(1, 2, key="v") is None


class TestGetConsole:
    def test_returns_singleton(self):
        assert get_console() is get_console()


class TestLiveMonitor:
    def test_render_returns_panel(self):
        monitor = LiveMonitor(get_console(), title="T")
        panel   = monitor._render()
        assert panel.title == "[bold cyan]T[/bold cyan]"

    def test_update_without_live_stores_metrics(self):
        monitor = LiveMonitor(get_console())
        monitor.update(loss=0.5, step=3)
        assert monitor._metrics == {"loss": 0.5, "step": 3}

    def test_render_formats_small_float_with_six_decimals(self):
        monitor = LiveMonitor(get_console())
        monitor.update(value=0.123456789)
        panel = monitor._render()
        assert panel is not None

    def test_context_manager_enters_and_exits(self):
        console = get_console()
        with LiveMonitor(console, title="Ctx") as monitor:
            monitor.update(a=1.0)
            assert monitor._live is not None
        assert monitor._live is None


class TestLogger:
    def test_creates_log_directory_and_file(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger  = Logger(log_dir=str(log_dir), name="exp_create", level="INFO")
        try:
            assert log_dir.is_dir()
            assert (log_dir / "exp_create.log").exists()
        finally:
            logger.close()

    def test_no_log_dir_means_no_file_handler(self, tmp_path):
        logger = Logger(log_dir="", name="exp_nofile")
        try:
            assert logger._file_handler is None
        finally:
            logger.close()

    def test_level_string_parsed_case_insensitively(self, tmp_path):
        logger = Logger(log_dir=str(tmp_path), name="exp_level", level="debug")
        try:
            assert logger.logger.level == logging.DEBUG
        finally:
            logger.close()

    def test_unknown_level_defaults_to_info(self, tmp_path):
        logger = Logger(log_dir=str(tmp_path), name="exp_bad_level", level="NONSENSE")
        try:
            assert logger.logger.level == logging.INFO
        finally:
            logger.close()

    def test_info_message_written_to_file(self, tmp_path):
        logger = Logger(log_dir=str(tmp_path), name="exp_write", level="INFO")
        try:
            logger.info("hello world")
        finally:
            logger.close()
        content = (tmp_path / "exp_write.log").read_text(encoding="utf-8")
        assert "hello world" in content

    def test_section_written_to_file_uppercase(self, tmp_path):
        logger = Logger(log_dir=str(tmp_path), name="exp_section", level="INFO")
        try:
            logger.section("setup")
        finally:
            logger.close()
        content = (tmp_path / "exp_section.log").read_text(encoding="utf-8")
        assert "SETUP" in content

    def test_subsection_and_ok_written(self, tmp_path):
        logger = Logger(log_dir=str(tmp_path), name="exp_sub", level="INFO")
        try:
            logger.subsection("step one")
            logger.ok("done")
        finally:
            logger.close()
        content = (tmp_path / "exp_sub.log").read_text(encoding="utf-8")
        assert "step one" in content
        assert "done" in content

    def test_fmt_float_uses_six_significant_digits(self):
        assert Logger._fmt(1.23456789) == "1.23457"

    def test_fmt_non_float_is_str(self):
        assert Logger._fmt("abc") == "abc"
        assert Logger._fmt(42) == "42"

    def test_close_removes_all_handlers(self, tmp_path):
        logger = Logger(log_dir=str(tmp_path), name="exp_close", level="INFO")
        logger.close()
        assert logger.logger.handlers == []
        assert logger._file_handler is None

    def test_context_manager_closes_on_exit(self, tmp_path):
        with Logger(log_dir=str(tmp_path), name="exp_ctx", level="INFO") as logger:
            logger.info("inside")
        assert logger.logger.handlers == []

    def test_reinit_same_name_removes_old_handlers(self, tmp_path):
        first = Logger(log_dir=str(tmp_path), name="exp_reinit", level="INFO")
        second = Logger(log_dir=str(tmp_path), name="exp_reinit", level="INFO")
        try:
            assert len(second.logger.handlers) == 2
        finally:
            first.close()
            second.close()

    def test_timer_logs_elapsed(self, tmp_path):
        logger = Logger(log_dir=str(tmp_path), name="exp_timer", level="INFO")
        try:
            with logger.timer("work"):
                pass
        finally:
            logger.close()
        content = (tmp_path / "exp_timer.log").read_text(encoding="utf-8")
        assert "work completed in" in content

    def test_kv_table_runs_without_error(self, tmp_path):
        logger = Logger(log_dir=str(tmp_path), name="exp_kv", level="INFO")
        try:
            logger.kv_table({"lr": 0.1, "name": "run"}, title="Config")
        finally:
            logger.close()

    def test_metrics_table_runs_without_error(self, tmp_path):
        logger = Logger(log_dir=str(tmp_path), name="exp_metrics", level="INFO")
        try:
            logger.metrics_table([{"epoch": 1, "loss": 0.5}], columns=["epoch", "loss"])
        finally:
            logger.close()

    def test_track_yields_progress(self, tmp_path):
        logger = Logger(log_dir=str(tmp_path), name="exp_track", level="INFO")
        try:
            with logger.track(transient=True) as progress:
                task = progress.add_task("doing", total=2)
                progress.advance(task)
        finally:
            logger.close()

    def test_progress_bar_is_alias_for_track(self):
        assert Logger.progress_bar is Logger.track

    def test_live_monitor_yields_monitor(self, tmp_path):
        logger = Logger(log_dir=str(tmp_path), name="exp_live", level="INFO")
        try:
            with logger.live_monitor(title="M") as monitor:
                monitor.update(loss=1.0)
                assert isinstance(monitor, LiveMonitor)
        finally:
            logger.close()

    def test_render_and_rule_and_panel_run(self, tmp_path):
        logger = Logger(log_dir=str(tmp_path), name="exp_render", level="INFO")
        try:
            logger.render("plain text")
            logger.rule("title")
            logger.panel("body", title="t")
        finally:
            logger.close()


class TestResourceMonitorConfig:
    @staticmethod
    def _cfg(**overrides):
        defaults = dict(
            enabled=True,
            poll_interval_sec=0.01,
            log_to_tensorboard=False,
            warn_ram_pct=999.0,
            warn_vram_pct=999.0,
            warn_swap_pct=999.0,
            warn_shm_pct=999.0,
            warn_cooldown_sec=0.0,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_load_config_reads_values(self):
        monitor = ResourceMonitor(self._cfg(poll_interval_sec=2.5), logger=None)
        try:
            assert monitor.interval == 2.5
            assert monitor.enabled is True
            assert monitor.log_to_tb is False
        finally:
            monitor._shutdown_nvml()

    def test_load_config_defaults_when_missing(self):
        monitor = ResourceMonitor(SimpleNamespace(), logger=None)
        try:
            assert monitor.enabled is True
            assert monitor.interval == 5.0
            assert monitor.warn_ram_pct == 90.0
        finally:
            monitor._shutdown_nvml()

    def test_default_step_getter_returns_zero(self):
        monitor = ResourceMonitor(self._cfg(), logger=None)
        try:
            assert monitor.step_getter() == 0
        finally:
            monitor._shutdown_nvml()

    def test_custom_step_getter_used(self):
        monitor = ResourceMonitor(self._cfg(), logger=None, step_getter=lambda: 17)
        try:
            assert monitor.step_getter() == 17
        finally:
            monitor._shutdown_nvml()

    def test_peak_tracking_initialised_to_zero(self):
        monitor = ResourceMonitor(self._cfg(), logger=None)
        try:
            assert all(v == 0.0 for v in monitor.peak.values())
        finally:
            monitor._shutdown_nvml()


class TestResourceMonitorSampling:
    @staticmethod
    def _cfg(**overrides):
        defaults = dict(
            enabled=True,
            poll_interval_sec=0.01,
            log_to_tensorboard=False,
            warn_ram_pct=999.0,
            warn_vram_pct=999.0,
            warn_swap_pct=999.0,
            warn_shm_pct=999.0,
            warn_cooldown_sec=0.0,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_bytes_to_gb_conversion(self):
        assert ResourceMonitor._bytes_to_gb(1024 ** 3) == pytest.approx(1.0)

    def test_sample_returns_core_ram_keys(self):
        monitor = ResourceMonitor(self._cfg(), logger=None)
        try:
            metrics = monitor.sample()
            for key in ("ram_used_gb", "ram_total_gb", "ram_pct", "swap_pct", "shm_pct"):
                assert key in metrics
            assert metrics["ram_total_gb"] > 0.0
        finally:
            monitor._shutdown_nvml()

    def test_sample_includes_vram_keys_even_without_gpu(self):
        monitor = ResourceMonitor(self._cfg(), logger=None)
        try:
            metrics = monitor.sample()
            assert "vram_used_gb" in metrics
            assert "vram_pct" in metrics
        finally:
            monitor._shutdown_nvml()

    def test_sample_updates_peak(self):
        monitor = ResourceMonitor(self._cfg(), logger=None)
        try:
            metrics = monitor.sample()
            assert monitor.peak["ram_used_gb"] == pytest.approx(metrics["ram_used_gb"])
            assert monitor.peak["ram_pct"] == pytest.approx(metrics["ram_pct"])
        finally:
            monitor._shutdown_nvml()

    def test_get_shm_usage_returns_pair(self):
        monitor = ResourceMonitor(self._cfg(), logger=None)
        try:
            used, pct = monitor._get_shm_usage()
            assert isinstance(used, float)
            assert isinstance(pct, float)
        finally:
            monitor._shutdown_nvml()

    def test_update_peak_metrics_keeps_maximum(self):
        monitor = ResourceMonitor(self._cfg(), logger=None)
        try:
            monitor._update_peak_metrics({"ram_pct": 50.0})
            monitor._update_peak_metrics({"ram_pct": 30.0})
            assert monitor.peak["ram_pct"] == 50.0
            monitor._update_peak_metrics({"ram_pct": 80.0})
            assert monitor.peak["ram_pct"] == 80.0
        finally:
            monitor._shutdown_nvml()


class TestResourceMonitorWarnings:
    class RecordingLogger:
        def __init__(self):
            self.warnings   = []
            self.sections   = []
            self.subsections = []

        def warning(self, msg):
            self.warnings.append(msg)

        def section(self, title):
            self.sections.append(title)

        def subsection(self, title):
            self.subsections.append(title)

    @staticmethod
    def _cfg(**overrides):
        defaults = dict(
            enabled=True,
            poll_interval_sec=0.01,
            log_to_tensorboard=False,
            warn_ram_pct=90.0,
            warn_vram_pct=90.0,
            warn_swap_pct=50.0,
            warn_shm_pct=80.0,
            warn_cooldown_sec=0.0,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_maybe_warn_emits_when_cooldown_elapsed(self):
        logger  = self.RecordingLogger()
        monitor = ResourceMonitor(self._cfg(warn_cooldown_sec=0.0), logger=logger)
        try:
            monitor._maybe_warn("ram", "too much ram")
            assert any("too much ram" in w for w in logger.warnings)
        finally:
            monitor._shutdown_nvml()

    def test_maybe_warn_suppressed_within_cooldown(self):
        logger  = self.RecordingLogger()
        monitor = ResourceMonitor(self._cfg(warn_cooldown_sec=1000.0), logger=logger)
        try:
            monitor._maybe_warn("ram", "first")
            monitor._maybe_warn("ram", "second")
            assert len(logger.warnings) == 1
        finally:
            monitor._shutdown_nvml()

    def test_maybe_warn_no_logger_is_safe(self):
        monitor = ResourceMonitor(self._cfg(), logger=None)
        try:
            monitor._maybe_warn("ram", "no logger")
        finally:
            monitor._shutdown_nvml()

    def test_check_warnings_triggers_on_high_ram(self):
        logger  = self.RecordingLogger()
        monitor = ResourceMonitor(self._cfg(warn_ram_pct=0.0), logger=logger)
        try:
            metrics = {
                "ram_pct": 95.0, "ram_used_gb": 10.0, "ram_total_gb": 16.0,
                "shm_used_gb": 1.0, "shm_pct": 5.0, "swap_pct": 0.0, "proc_rss_gb": 2.0,
            }
            monitor._check_warnings(metrics, gpu_used=0.0, gpu_total=0.0)
            assert any("RAM usage" in w for w in logger.warnings)
        finally:
            monitor._shutdown_nvml()

    def test_check_warnings_triggers_on_high_vram(self):
        logger  = self.RecordingLogger()
        monitor = ResourceMonitor(self._cfg(warn_vram_pct=50.0), logger=logger)
        try:
            metrics = {
                "ram_pct": 0.0, "ram_used_gb": 1.0, "ram_total_gb": 16.0,
                "shm_used_gb": 0.0, "shm_pct": 0.0, "swap_pct": 0.0, "proc_rss_gb": 1.0,
            }
            monitor._check_warnings(metrics, gpu_used=8.0, gpu_total=10.0)
            assert any("VRAM usage" in w for w in logger.warnings)
        finally:
            monitor._shutdown_nvml()

    def test_check_warnings_no_vram_warning_without_gpu(self):
        logger  = self.RecordingLogger()
        monitor = ResourceMonitor(self._cfg(warn_vram_pct=0.0), logger=logger)
        try:
            metrics = {
                "ram_pct": 0.0, "ram_used_gb": 1.0, "ram_total_gb": 16.0,
                "shm_used_gb": 0.0, "shm_pct": 0.0, "swap_pct": 0.0, "proc_rss_gb": 1.0,
            }
            monitor._check_warnings(metrics, gpu_used=0.0, gpu_total=0.0)
            assert not any("VRAM usage" in w for w in logger.warnings)
        finally:
            monitor._shutdown_nvml()


class TestResourceMonitorLifecycle:
    class RecordingLogger:
        def __init__(self):
            self.sections    = []
            self.subsections = []
            self.warnings    = []

        def section(self, title):
            self.sections.append(title)

        def subsection(self, title):
            self.subsections.append(title)

        def warning(self, msg):
            self.warnings.append(msg)

    @staticmethod
    def _cfg(**overrides):
        defaults = dict(
            enabled=True,
            poll_interval_sec=0.01,
            log_to_tensorboard=True,
            warn_ram_pct=999.0,
            warn_vram_pct=999.0,
            warn_swap_pct=999.0,
            warn_shm_pct=999.0,
            warn_cooldown_sec=0.0,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_disabled_monitor_does_not_start_thread(self):
        logger  = self.RecordingLogger()
        monitor = ResourceMonitor(self._cfg(enabled=False), logger=logger)
        try:
            monitor.start()
            assert monitor._thread is None
            assert any("disabled" in s for s in monitor.logger.subsections)
        finally:
            monitor._shutdown_nvml()

    def test_start_and_stop_runs_background_thread(self):
        logger  = self.RecordingLogger()
        tracker = NullTracker()
        monitor = ResourceMonitor(self._cfg(), logger=logger, tracker=tracker)
        with monitor:
            assert monitor._thread is not None
            deadline = __import__("time").time() + 5.0
            while monitor._sample_idx < 1 and __import__("time").time() < deadline:
                __import__("time").sleep(0.02)
            assert monitor._sample_idx >= 1
        assert monitor._thread is None
        assert any("Resource Monitor" in s for s in logger.sections)

    def test_publish_logs_to_tracker(self):
        recorded = []

        class CapturingTracker(NullTracker):
            def log_metrics(self, prefix, values, step=None):
                recorded.append((prefix, dict(values), step))

        monitor = ResourceMonitor(self._cfg(), logger=None, tracker=CapturingTracker(), step_getter=lambda: 3)
        try:
            monitor._publish({"ram_pct": 1.0})
            assert recorded == [("system/resources", {"ram_pct": 1.0}, 3)]
        finally:
            monitor._shutdown_nvml()

    def test_publish_skips_when_log_to_tb_false(self):
        recorded = []

        class CapturingTracker(NullTracker):
            def log_metrics(self, prefix, values, step=None):
                recorded.append(prefix)

        monitor = ResourceMonitor(self._cfg(log_to_tensorboard=False), logger=None, tracker=CapturingTracker())
        try:
            monitor._publish({"ram_pct": 1.0})
            assert recorded == []
        finally:
            monitor._shutdown_nvml()

    def test_stop_logs_peak_metrics(self):
        logger  = self.RecordingLogger()
        monitor = ResourceMonitor(self._cfg(), logger=logger)
        monitor.peak["ram_pct"] = 42.0
        monitor.start()
        monitor.stop()
        assert any("Peaks" in s for s in logger.sections)
        assert any("peak" in s for s in logger.subsections)

    def test_double_start_is_idempotent(self):
        monitor = ResourceMonitor(self._cfg(), logger=None, tracker=NullTracker())
        with monitor:
            thread_one = monitor._thread
            monitor.start()
            assert monitor._thread is thread_one
