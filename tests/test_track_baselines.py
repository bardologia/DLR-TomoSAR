from __future__ import annotations

import json

import numpy as np
import pytest

from configuration.training_config import GeometryConfig
from tools.track_baselines import BaselineExtractor, TrackBaselines, TrackFileResolver, TrackReader


def _track(horizontal, vertical, n_samples=100):
    raw    = np.zeros((5, n_samples))
    raw[2] = horizontal
    raw[3] = vertical
    return raw


def _fake_reader(tracks: dict):
    return lambda path: tracks[path]


class TestBaselineExtraction:
    def test_reference_track_is_zero(self):
        tracks    = {"a.rat": _track(5.0, 10.0), "b.rat": _track(8.0, 25.0)}
        extractor = BaselineExtractor({"PS03": "a.rat", "PS07": "b.rat"}, reader=_fake_reader(tracks))
        table     = extractor.extract()

        assert table.vertical[0]   == pytest.approx(0.0)
        assert table.horizontal[0] == pytest.approx(0.0)

    def test_baselines_relative_to_first_track(self):
        tracks    = {"a.rat": _track(5.0, 10.0), "b.rat": _track(8.0, 25.0), "c.rat": _track(2.0, 40.0)}
        extractor = BaselineExtractor({"m": "a.rat", "s1": "b.rat", "s2": "c.rat"}, reader=_fake_reader(tracks))
        table     = extractor.extract()

        assert table.vertical   == pytest.approx([0.0, 15.0, 30.0])
        assert table.horizontal == pytest.approx([0.0, 3.0, -3.0])

    def test_order_preserved(self):
        tracks    = {"a.rat": _track(0.0, 0.0), "b.rat": _track(0.0, 10.0), "c.rat": _track(0.0, 20.0)}
        extractor = BaselineExtractor({"m": "a.rat", "s2": "c.rat", "s1": "b.rat"}, reader=_fake_reader(tracks))
        table     = extractor.extract()

        assert table.labels   == ["m", "s2", "s1"]
        assert table.vertical == pytest.approx([0.0, 20.0, 10.0])

    def test_azimuth_window_applied(self):
        ramp      = np.arange(100, dtype=float)
        tracks    = {"a.rat": _track(0.0, 0.0), "b.rat": _track(0.0, ramp)}
        extractor = BaselineExtractor({"m": "a.rat", "s": "b.rat"}, azimuth_window=(10, 20), reader=_fake_reader(tracks))
        table     = extractor.extract()

        assert table.vertical[1]     == pytest.approx(np.mean(ramp[10:20]))
        assert table.azimuth_window  == (10, 20)

    def test_window_clamped_to_track_length(self):
        tracks    = {"a.rat": _track(0.0, 0.0, n_samples=50), "b.rat": _track(0.0, 7.0, n_samples=50)}
        extractor = BaselineExtractor({"m": "a.rat", "s": "b.rat"}, azimuth_window=(10, 5000), reader=_fake_reader(tracks))
        table     = extractor.extract()

        assert table.vertical[1] == pytest.approx(7.0)

    def test_window_beyond_track_raises(self):
        tracks    = {"a.rat": _track(0.0, 0.0, n_samples=50)}
        extractor = BaselineExtractor({"m": "a.rat"}, azimuth_window=(60, 100), reader=_fake_reader(tracks))

        with pytest.raises(ValueError):
            extractor.extract()

    def test_nan_samples_ignored(self):
        vertical      = np.full(100, 12.0)
        vertical[:10] = np.nan
        tracks        = {"a.rat": _track(0.0, 0.0), "b.rat": _track(0.0, vertical)}
        extractor     = BaselineExtractor({"m": "a.rat", "s": "b.rat"}, reader=_fake_reader(tracks))
        table         = extractor.extract()

        assert table.vertical[1] == pytest.approx(12.0)

    def test_std_reported_per_track(self):
        vertical  = np.concatenate([np.zeros(50), np.ones(50)])
        tracks    = {"a.rat": _track(0.0, vertical)}
        extractor = BaselineExtractor({"m": "a.rat"}, reader=_fake_reader(tracks))
        table     = extractor.extract()

        assert table.vertical_std[0] == pytest.approx(0.5)


class TestBaselineComponents:
    def _table(self):
        return TrackBaselines(labels=["m", "s"], vertical=[0.0, 4.0], horizontal=[0.0, 3.0], vertical_std=[0.0, 0.0], horizontal_std=[0.0, 0.0])

    def test_vertical_default(self):
        assert self._table().baselines() == (0.0, 4.0)

    def test_horizontal_component(self):
        assert self._table().baselines("horizontal") == (0.0, 3.0)

    def test_magnitude_component(self):
        assert self._table().baselines("magnitude") == pytest.approx((0.0, 5.0))

    def test_unknown_component_raises(self):
        with pytest.raises(ValueError):
            self._table().baselines("diagonal")


class TestPersistence:
    def _table(self):
        return TrackBaselines(
            labels         = ["PS03", "PS07"],
            vertical       = [0.0, 14.7],
            horizontal     = [0.0, -1.2],
            vertical_std   = [0.3, 0.4],
            horizontal_std = [0.1, 0.2],
            track_files    = ["a.rat", "b.rat"],
            azimuth_window = (1000, 16000),
        )

    def test_round_trip(self, tmp_path):
        path   = self._table().save(tmp_path / "meta" / TrackBaselines.FILENAME)
        loaded = TrackBaselines.load(path)

        assert loaded == self._table()

    def test_payload_is_plain_json(self, tmp_path):
        path    = self._table().save(tmp_path / TrackBaselines.FILENAME)
        payload = json.loads(path.read_text())

        assert payload["reference"] == "PS03"
        assert payload["vertical"]  == [0.0, 14.7]

    def test_describe_keys(self):
        description = self._table().describe()

        assert description["Tracks"]    == 2
        assert description["Reference"] == "PS03"


class TestTrackReader:
    def test_rejects_one_dimensional_data(self):
        reader = TrackReader(lambda path: np.zeros(10))

        with pytest.raises(ValueError):
            reader.read("a.rat")

    def test_rejects_too_few_rows(self):
        reader = TrackReader(lambda path: np.zeros((2, 10)))

        with pytest.raises(ValueError):
            reader.read("a.rat")

    def test_accepts_valid_track(self):
        reader = TrackReader(lambda path: np.zeros((5, 10)))

        assert reader.read("a.rat").shape == (5, 10)


class TestTrackFileResolver:
    def _make_pass(self, tmp_path, pass_name, track_dir, filename):
        directory = tmp_path / pass_name / track_dir / "INF" / "INF-TRACK"
        directory.mkdir(parents=True)
        (directory / filename).touch()
        return tmp_path / pass_name / track_dir

    def test_resolves_track_file(self, tmp_path):
        pass_dir = self._make_pass(tmp_path, "PS03", "T01L", "track_sar_resa_17sartom0103_Lhh_t01L.rat")
        resolved = TrackFileResolver().resolve(pass_dir)

        assert resolved.name == "track_sar_resa_17sartom0103_Lhh_t01L.rat"

    def test_missing_track_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TrackFileResolver().resolve(tmp_path / "PS03" / "T01L")

    def test_label_uses_pass_name_above_track_directory(self, tmp_path):
        assert TrackFileResolver().label(tmp_path / "PS07" / "T01L") == "PS07"

    def test_label_falls_back_to_directory_name(self, tmp_path):
        assert TrackFileResolver().label(tmp_path / "PS07") == "PS07"

    def test_resolve_passes_keeps_order(self, tmp_path):
        first  = self._make_pass(tmp_path, "PS03", "T01L", "track_a.rat")
        second = self._make_pass(tmp_path, "PS07", "T01L", "track_b.rat")
        mapped = TrackFileResolver().resolve_passes([first, second])

        assert list(mapped.keys()) == ["PS03", "PS07"]


class TestGeometryResolution:
    def _write_baselines(self, dataset_dir):
        table = TrackBaselines(labels=["m", "s"], vertical=[0.0, 14.7], horizontal=[0.0, -1.2], vertical_std=[0.0, 0.0], horizontal_std=[0.0, 0.0])
        return table.save(GeometryConfig().baselines_file(dataset_dir))

    def test_auto_uses_dataset_file(self, tmp_path):
        path     = self._write_baselines(tmp_path)
        resolved = GeometryConfig().resolved(tmp_path)

        assert resolved.baselines        == pytest.approx((0.0, 14.7))
        assert resolved.baselines_origin == str(path)

    def test_auto_without_file_keeps_manual(self, tmp_path):
        config   = GeometryConfig(baselines=(0.0, 10.0))
        resolved = config.resolved(tmp_path)

        assert resolved.baselines        == (0.0, 10.0)
        assert resolved.baselines_origin == "config"

    def test_manual_ignores_dataset_file(self, tmp_path):
        self._write_baselines(tmp_path)
        config   = GeometryConfig(baselines=(0.0, 10.0), baselines_source="manual")
        resolved = config.resolved(tmp_path)

        assert resolved.baselines == (0.0, 10.0)

    def test_dataset_source_requires_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GeometryConfig(baselines_source="dataset").resolved(tmp_path)

    def test_explicit_kz_bypasses_resolution(self, tmp_path):
        self._write_baselines(tmp_path)
        config   = GeometryConfig(kz_values=(0.1, 0.2))
        resolved = config.resolved(tmp_path)

        assert resolved.kz_values == (0.1, 0.2)
        assert resolved.baselines == GeometryConfig().baselines

    def test_horizontal_component_selected(self, tmp_path):
        self._write_baselines(tmp_path)
        resolved = GeometryConfig(baseline_component="horizontal").resolved(tmp_path)

        assert resolved.baselines == pytest.approx((0.0, -1.2))

    def test_unknown_source_raises(self, tmp_path):
        with pytest.raises(ValueError):
            GeometryConfig(baselines_source="guess").resolved(tmp_path)
