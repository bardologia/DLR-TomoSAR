from __future__ import annotations

import json

import numpy as np
import pytest

from configuration.training_config import GeometryConfig
from tools.track_baselines import BaselineExtractor, BaselineValidator, DuplicatePassError, TrackBaselines, TrackFileResolver, TrackProfiles, TrackReader


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

    def test_absolute_means_preserved(self):
        tracks    = {"a.rat": _track(5.0, 10.0), "b.rat": _track(8.0, 25.0)}
        extractor = BaselineExtractor({"m": "a.rat", "s": "b.rat"}, reader=_fake_reader(tracks))
        table     = extractor.extract()

        assert table.vertical_absolute   == pytest.approx([10.0, 25.0])
        assert table.horizontal_absolute == pytest.approx([5.0, 8.0])


class TestProfileExtraction:
    def test_profiles_match_windowed_rows(self):
        ramp            = np.arange(100, dtype=float)
        tracks          = {"a.rat": _track(0.0, 0.0), "b.rat": _track(ramp, 2.0 * ramp)}
        extractor       = BaselineExtractor({"m": "a.rat", "s": "b.rat"}, azimuth_window=(10, 20), reader=_fake_reader(tracks), validator=BaselineValidator(std_threshold=1000.0))
        table, profiles = extractor.extract_with_profiles()

        assert profiles.labels           == ["m", "s"]
        assert profiles.horizontal.shape == (2, 10)
        assert profiles.horizontal[1]    == pytest.approx(ramp[10:20])
        assert profiles.vertical[1]      == pytest.approx(2.0 * ramp[10:20])
        assert profiles.azimuth_start    == 10

    def test_azimuth_axis_offset_by_window(self):
        tracks      = {"a.rat": _track(0.0, 0.0)}
        extractor   = BaselineExtractor({"m": "a.rat"}, azimuth_window=(30, 40), reader=_fake_reader(tracks))
        _, profiles = extractor.extract_with_profiles()

        assert profiles.azimuth_axis.tolist() == list(range(30, 40))

    def test_profiles_truncated_to_common_length(self):
        tracks          = {"a.rat": _track(0.0, 0.0, n_samples=50), "b.rat": _track(1.0, 2.0, n_samples=80)}
        extractor       = BaselineExtractor({"m": "a.rat", "s": "b.rat"}, reader=_fake_reader(tracks))
        table, profiles = extractor.extract_with_profiles()

        assert profiles.n_samples == 50
        assert table.vertical[1]  == pytest.approx(2.0)

    def test_relative_profiles_subtract_reference(self):
        ramp        = np.arange(100, dtype=float)
        tracks      = {"a.rat": _track(0.0, ramp), "b.rat": _track(0.0, ramp + 7.0)}
        extractor   = BaselineExtractor({"m": "a.rat", "s": "b.rat"}, reader=_fake_reader(tracks), validator=BaselineValidator(std_threshold=1000.0))
        _, profiles = extractor.extract_with_profiles()
        relative    = profiles.relative_to_reference("vertical")

        assert relative[0] == pytest.approx(np.zeros(100))
        assert relative[1] == pytest.approx(np.full(100, 7.0))

    def test_table_consistent_with_profiles(self):
        tracks          = {"a.rat": _track(5.0, 10.0), "b.rat": _track(8.0, 25.0)}
        extractor       = BaselineExtractor({"m": "a.rat", "s": "b.rat"}, reader=_fake_reader(tracks))
        table, profiles = extractor.extract_with_profiles()

        assert table.vertical_absolute   == pytest.approx(np.nanmean(profiles.vertical,   axis=1))
        assert table.horizontal_absolute == pytest.approx(np.nanmean(profiles.horizontal, axis=1))


class TestProfilePersistence:
    def _profiles(self):
        return TrackProfiles(
            labels        = ["PS03", "PS07"],
            horizontal    = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            vertical      = np.array([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]),
            azimuth_start = 1000,
            track_files   = ["a.rat", "b.rat"],
        )

    def test_round_trip(self, tmp_path):
        path   = self._profiles().save(tmp_path / "data" / TrackProfiles.FILENAME)
        loaded = TrackProfiles.load(path)

        assert loaded.labels        == ["PS03", "PS07"]
        assert loaded.azimuth_start == 1000
        assert loaded.track_files   == ["a.rat", "b.rat"]
        assert loaded.horizontal    == pytest.approx(self._profiles().horizontal)
        assert loaded.vertical      == pytest.approx(self._profiles().vertical)

    def test_profiles_file_resolves_under_data(self, tmp_path):
        assert TrackProfiles.profiles_file(tmp_path) == tmp_path / "data" / TrackProfiles.FILENAME


class TestProfileDeviation:
    def _profiles(self):
        return TrackProfiles(
            labels        = ["PS02", "PS04"],
            horizontal    = np.array([[2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 6.0, 6.0]]),
            vertical      = np.array([[5.0, 5.0, 5.0, 5.0], [1.0, 5.0, 1.0, 5.0]]),
            azimuth_start = 1000,
        )

    def test_planar_deviation_zero_for_constant_track(self):
        deviation = self._profiles().planar_deviation()

        assert deviation[0] == pytest.approx(np.zeros(4))

    def test_planar_deviation_combines_components(self):
        deviation = self._profiles().planar_deviation()

        assert deviation[1] == pytest.approx(np.full(4, np.hypot(3.0, 2.0)))

    def test_deviation_radii_are_rms(self):
        radii = self._profiles().deviation_radii()

        assert radii[0] == pytest.approx(0.0)
        assert radii[1] == pytest.approx(np.hypot(3.0, 2.0))

    def test_position_summary_values(self):
        summary = self._profiles().position_summary()

        assert summary["labels"]          == ["PS02", "PS04"]
        assert summary["horizontal_mean"] == pytest.approx([2.0, 3.0])
        assert summary["vertical_mean"]   == pytest.approx([5.0, 3.0])
        assert summary["horizontal_span"] == pytest.approx([0.0, 6.0])
        assert summary["vertical_span"]   == pytest.approx([0.0, 4.0])
        assert summary["deviation_rms"]   == pytest.approx([0.0, np.hypot(3.0, 2.0)])
        assert summary["deviation_max"]   == pytest.approx([0.0, np.hypot(3.0, 2.0)])
        assert summary["azimuth_start"]   == 1000
        assert summary["n_samples"]       == 4

    def test_position_summary_is_plain_json(self):
        import json

        json.dumps(self._profiles().position_summary())


class TestProfileSubset:
    def _profiles(self):
        return TrackProfiles(
            labels        = ["PS02", "PS04", "PS06", "PS08"],
            horizontal    = np.arange(4 * 5, dtype=float).reshape(4, 5),
            vertical      = np.arange(4 * 5, dtype=float).reshape(4, 5) + 100.0,
            azimuth_start = 1000,
            track_files   = ["a", "b", "c", "d"],
        )

    def test_none_returns_same_object(self):
        profiles = self._profiles()

        assert profiles.subset(None) is profiles

    def test_subset_keeps_reference_and_selected_rows(self):
        subset = self._profiles().subset(("PS04", "PS08"))

        assert subset.labels == ["PS02", "PS04", "PS08"]
        assert subset.horizontal == pytest.approx(self._profiles().horizontal[[0, 1, 3]])
        assert subset.vertical   == pytest.approx(self._profiles().vertical[[0, 1, 3]])
        assert subset.track_files == ["a", "b", "d"]
        assert subset.azimuth_start == 1000

    def test_selecting_reference_raises(self):
        with pytest.raises(ValueError, match="reference"):
            self._profiles().subset(("PS02",))

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError, match="Unknown secondary labels"):
            self._profiles().subset(("PS99",))


class TestSubset:
    def _table(self):
        return TrackBaselines(
            labels              = ["PS02", "PS04", "PS06", "PS08", "PS26"],
            vertical            = [0.0, 1.0, 2.0, 3.0, 4.0],
            horizontal          = [0.0, 10.0, 20.0, 30.0, 40.0],
            vertical_std        = [0.1, 0.2, 0.3, 0.4, 0.5],
            horizontal_std      = [0.5, 0.4, 0.3, 0.2, 0.1],
            vertical_absolute   = [100.0, 101.0, 102.0, 103.0, 104.0],
            horizontal_absolute = [200.0, 210.0, 220.0, 230.0, 240.0],
            track_files         = ["a", "b", "c", "d", "e"],
            azimuth_window      = (1000, 16000),
        )

    def test_none_returns_same_table(self):
        table = self._table()

        assert table.subset(None) is table

    def test_subset_keeps_reference_and_selected(self):
        subset = self._table().subset(("PS04", "PS08"))

        assert subset.labels     == ["PS02", "PS04", "PS08"]
        assert subset.vertical   == pytest.approx([0.0, 1.0, 3.0])
        assert subset.horizontal == pytest.approx([0.0, 10.0, 30.0])

    def test_subset_preserves_dataset_order(self):
        subset = self._table().subset(("PS26", "PS04"))

        assert subset.labels == ["PS02", "PS04", "PS26"]

    def test_subset_carries_all_fields(self):
        subset = self._table().subset(("PS06",))

        assert subset.vertical_std        == pytest.approx([0.1, 0.3])
        assert subset.horizontal_absolute == pytest.approx([200.0, 220.0])
        assert subset.track_files         == ["a", "c"]
        assert subset.azimuth_window      == (1000, 16000)

    def test_selecting_reference_raises(self):
        with pytest.raises(ValueError, match="reference"):
            self._table().subset(("PS02", "PS04"))

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError, match="Unknown secondary labels"):
            self._table().subset(("PS99",))

    def test_full_selection_matches_original(self):
        table  = self._table()
        subset = table.subset(("PS04", "PS06", "PS08", "PS26"))

        assert subset.labels   == table.labels
        assert subset.vertical == pytest.approx(table.vertical)


class TestBaselineComponents:
    def _table(self):
        return TrackBaselines(labels=["m", "s"], vertical=[0.0, 4.0], horizontal=[0.0, 3.0], vertical_std=[0.0, 0.0], horizontal_std=[0.0, 0.0])

    def test_vertical_default(self):
        assert self._table().baselines() == (0.0, 4.0)

    def test_horizontal_component(self):
        assert self._table().baselines("horizontal") == (0.0, 3.0)

    def test_magnitude_component(self):
        assert self._table().baselines("magnitude") == pytest.approx((0.0, 5.0))

    def test_perpendicular_at_nadir_is_horizontal(self):
        assert self._table().baselines("perpendicular", look_angle_deg=0.0) == pytest.approx((0.0, 3.0))

    def test_perpendicular_at_grazing_is_vertical(self):
        assert self._table().baselines("perpendicular", look_angle_deg=90.0) == pytest.approx((0.0, 4.0))

    def test_perpendicular_mixes_components(self):
        expected = 3.0 * np.cos(np.deg2rad(45.0)) + 4.0 * np.sin(np.deg2rad(45.0))
        assert self._table().baselines("perpendicular", look_angle_deg=45.0) == pytest.approx((0.0, expected))

    def test_perpendicular_requires_look_angle(self):
        with pytest.raises(ValueError):
            self._table().baselines("perpendicular")

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

    def test_absolute_means_round_trip(self, tmp_path):
        table  = self._table()
        table.vertical_absolute   = [102.3, 117.0]
        table.horizontal_absolute = [-44.1, -45.3]

        loaded = TrackBaselines.load(table.save(tmp_path / TrackBaselines.FILENAME))

        assert loaded.vertical_absolute   == pytest.approx([102.3, 117.0])
        assert loaded.horizontal_absolute == pytest.approx([-44.1, -45.3])

    def test_payload_without_absolutes_raises(self):
        payload = self._table().to_payload()
        payload.pop("vertical_absolute")

        with pytest.raises(KeyError):
            TrackBaselines.from_payload(payload)


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

    def test_prefers_resampled_track_over_loc(self, tmp_path):
        pass_dir = self._make_pass(tmp_path, "PS02", "T01L", "track_loc_17sartom0102_Lhh_t01L.rat")
        (pass_dir / "INF" / "INF-TRACK" / "track_sar_resa_17sartom0102_Lhh_t01L.rat").touch()
        resolved = TrackFileResolver().resolve(pass_dir)

        assert resolved.name == "track_sar_resa_17sartom0102_Lhh_t01L.rat"

    def test_falls_back_to_loc_when_alone(self, tmp_path):
        pass_dir = self._make_pass(tmp_path, "PS02", "T01L", "track_loc_17sartom0102_Lhh_t01L.rat")
        resolved = TrackFileResolver().resolve(pass_dir)

        assert resolved.name == "track_loc_17sartom0102_Lhh_t01L.rat"

    def test_missing_track_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TrackFileResolver().resolve(tmp_path / "PS03" / "T01L")

    def _make_flightline_pass(self, tmp_path, flight, pass_name, track_dir, filename):
        directory = tmp_path / flight / pass_name / track_dir / "INF" / "INF-TRACK"
        directory.mkdir(parents=True)
        (directory / filename).touch()
        return tmp_path / flight / pass_name / track_dir

    def test_label_qualifies_pass_with_flight_line(self, tmp_path):
        assert TrackFileResolver().label(tmp_path / "FL01" / "PS07" / "T01L") == "FL01_PS07"

    def test_label_qualifies_when_no_track_directory(self, tmp_path):
        assert TrackFileResolver().label(tmp_path / "FL02" / "PS07") == "FL02_PS07"

    def test_resolve_passes_keeps_order(self, tmp_path):
        first  = self._make_flightline_pass(tmp_path, "FL01", "PS03", "T01L", "track_a.rat")
        second = self._make_flightline_pass(tmp_path, "FL01", "PS07", "T01L", "track_b.rat")
        mapped = TrackFileResolver().resolve_passes([first, second])

        assert list(mapped.keys()) == ["FL01_PS03", "FL01_PS07"]

    def test_resolve_passes_disambiguates_cross_flightline(self, tmp_path):
        fl01   = self._make_flightline_pass(tmp_path, "FL01", "PS29", "T01L", "track_sar_resa_17sartom0129.rat")
        fl02   = self._make_flightline_pass(tmp_path, "FL02", "PS29", "T01L", "track_sar_resa_17sartom0229.rat")
        unique = self._make_flightline_pass(tmp_path, "FL01", "PS04", "T01L", "track_sar_resa_17sartom0104.rat")
        mapped = TrackFileResolver().resolve_passes([unique, fl01, fl02])

        assert list(mapped.keys()) == ["FL01_PS04", "FL01_PS29", "FL02_PS29"]

    def test_resolve_passes_identical_pass_raises(self, tmp_path):
        primary   = self._make_flightline_pass(tmp_path, "FL01", "PS02", "T01L", "track_a.rat")
        secondary = self._make_flightline_pass(tmp_path, "FL01", "PS04", "T01L", "track_b.rat")

        with pytest.raises(DuplicatePassError, match="Duplicate pass label FL01_PS02"):
            TrackFileResolver().resolve_passes([primary, secondary, primary])


class TestBaselineValidation:
    def test_mixed_products_raise(self):
        tracks = {
            "track_loc_17sartom0102_Lhh_t01L.rat"      : _track(5.0, 10.0),
            "track_sar_resa_17sartom0104_Lhh_t01L.rat" : _track(8.0, 25.0),
        }
        extractor = BaselineExtractor({"PS02": "track_loc_17sartom0102_Lhh_t01L.rat", "PS04": "track_sar_resa_17sartom0104_Lhh_t01L.rat"}, reader=_fake_reader(tracks))

        with pytest.raises(ValueError, match="Mixed track file products"):
            extractor.extract()

    def test_consistent_products_pass(self):
        tracks = {
            "track_sar_resa_17sartom0102_Lhh_t01L.rat" : _track(5.0, 10.0),
            "track_sar_resa_17sartom0104_Lhh_t01L.rat" : _track(8.0, 25.0),
        }
        extractor = BaselineExtractor({"PS02": "track_sar_resa_17sartom0102_Lhh_t01L.rat", "PS04": "track_sar_resa_17sartom0104_Lhh_t01L.rat"}, reader=_fake_reader(tracks))
        table     = extractor.extract()

        assert table.vertical == pytest.approx([0.0, 15.0])

    def test_large_std_raises(self):
        wild      = np.concatenate([np.full(50, -300.0), np.full(50, 300.0)])
        tracks    = {"a.rat": _track(wild, 0.0)}
        extractor = BaselineExtractor({"m": "a.rat"}, reader=_fake_reader(tracks))

        with pytest.raises(ValueError, match="exceeds threshold"):
            extractor.extract()

    def test_threshold_override_allows_large_std(self):
        wild      = np.concatenate([np.full(50, -300.0), np.full(50, 300.0)])
        tracks    = {"a.rat": _track(wild, 0.0)}
        extractor = BaselineExtractor({"m": "a.rat"}, reader=_fake_reader(tracks), validator=BaselineValidator(std_threshold=500.0))
        table     = extractor.extract()

        assert table.horizontal_std[0] == pytest.approx(300.0)


class TestGeometryResolution:
    def _write_baselines(self, dataset_dir):
        table = TrackBaselines(labels=["m", "s"], vertical=[0.0, 14.7], horizontal=[0.0, -1.2], vertical_std=[0.0, 0.0], horizontal_std=[0.0, 0.0])
        return table.save(GeometryConfig().baselines_file(dataset_dir))

    def test_auto_uses_dataset_file(self, tmp_path):
        path     = self._write_baselines(tmp_path)
        resolved = GeometryConfig().resolved(tmp_path)
        theta    = np.deg2rad(GeometryConfig().look_angle_deg)
        expected = -1.2 * np.cos(theta) + 14.7 * np.sin(theta)

        assert resolved.baselines        == pytest.approx((0.0, expected))
        assert resolved.baselines_origin == str(path)

    def test_vertical_component_selected(self, tmp_path):
        self._write_baselines(tmp_path)
        resolved = GeometryConfig(baseline_component="vertical").resolved(tmp_path)

        assert resolved.baselines == pytest.approx((0.0, 14.7))

    def test_look_angle_changes_perpendicular_baseline(self, tmp_path):
        self._write_baselines(tmp_path)
        nadir    = GeometryConfig(look_angle_deg=0.0).resolved(tmp_path)
        grazing  = GeometryConfig(look_angle_deg=90.0).resolved(tmp_path)

        assert nadir.baselines   == pytest.approx((0.0, -1.2))
        assert grazing.baselines == pytest.approx((0.0, 14.7))

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

    def test_secondary_labels_subset_baselines(self, tmp_path):
        table = TrackBaselines(labels=["m", "s1", "s2", "s3"], vertical=[0.0, 5.0, 10.0, 15.0], horizontal=[0.0, 0.0, 0.0, 0.0], vertical_std=[0.0] * 4, horizontal_std=[0.0] * 4)
        table.save(GeometryConfig().baselines_file(tmp_path))

        resolved = GeometryConfig(baseline_component="vertical").resolved(tmp_path, secondary_labels=("s1", "s3"))

        assert resolved.baselines == pytest.approx((0.0, 5.0, 15.0))

    def test_secondary_labels_ignored_for_manual(self, tmp_path):
        config   = GeometryConfig(baselines=(0.0, 10.0), baselines_source="manual")
        resolved = config.resolved(tmp_path, secondary_labels=("s1",))

        assert resolved.baselines == (0.0, 10.0)
