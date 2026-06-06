from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from configuration.representation import Representation, _CHANNELS_PER_PASS
from tools.config_cli import ConfigCli, Detacher


@dataclass
class Inner:
    learning_rate : float = 0.01
    epochs        : int   = 10


@dataclass
class Outer:
    name         : str         = "run"
    enabled      : bool        = True
    out_dir      : Path        = Path("results")
    tags         : list        = field(default_factory=lambda: ["a", "b"])
    shape        : tuple       = (3, 4)
    options      : dict        = field(default_factory=lambda: {"k": 1})
    free         : object      = None
    threshold    : float       = 1.5
    inner        : Inner       = field(default_factory=Inner)


def make_config() -> Outer:
    return Outer()


class TestLeaves:
    def test_leaves_flattens_nested_dataclass_with_dotted_paths(self):
        leaves = dict(ConfigCli._leaves(make_config()))

        assert leaves["name"]               == "run"
        assert leaves["inner.learning_rate"] == 0.01
        assert leaves["inner.epochs"]        == 10

    def test_leaves_does_not_emit_nested_dataclass_itself(self):
        paths = [p for p, _ in ConfigCli._leaves(make_config())]

        assert "inner" not in paths
        assert "inner.epochs" in paths

    def test_leaves_on_flat_dataclass(self):
        leaves = dict(ConfigCli._leaves(Inner()))

        assert leaves == {"learning_rate": 0.01, "epochs": 10}


class TestApplyCoercion:
    def test_apply_returns_same_config_object(self):
        config = make_config()
        cli    = ConfigCli(config)

        returned = cli.apply([])

        assert returned is config

    def test_apply_no_argv_leaves_defaults_unchanged(self):
        config = make_config()
        cli    = ConfigCli(config)

        cli.apply([])

        assert config.name == "run"
        assert config.inner.epochs == 10
        assert cli.overrides == {}

    def test_apply_int_override(self):
        config = make_config()
        cli    = ConfigCli(config)

        cli.apply(["--inner.epochs", "42"])

        assert config.inner.epochs == 42
        assert isinstance(config.inner.epochs, int)
        assert cli.overrides["inner.epochs"] == 42

    def test_apply_float_override(self):
        config = make_config()
        cli    = ConfigCli(config)

        cli.apply(["--threshold", "2.75"])

        assert config.threshold == pytest.approx(2.75)
        assert isinstance(config.threshold, float)

    def test_apply_string_override(self):
        config = make_config()
        cli    = ConfigCli(config)

        cli.apply(["--name", "experiment"])

        assert config.name == "experiment"

    @pytest.mark.parametrize("raw", ["true", "1", "yes", "on", "TRUE", "On"])
    def test_apply_bool_true_variants(self, raw):
        config = make_config()
        config.enabled = False
        cli = ConfigCli(config)

        cli.apply(["--enabled", raw])

        assert config.enabled is True

    @pytest.mark.parametrize("raw", ["false", "0", "no", "off", "FALSE", "Off"])
    def test_apply_bool_false_variants(self, raw):
        config = make_config()
        cli    = ConfigCli(config)

        cli.apply(["--enabled", raw])

        assert config.enabled is False

    def test_apply_path_override_produces_path(self):
        config = make_config()
        cli    = ConfigCli(config)

        cli.apply(["--out_dir", "/tmp/run"])

        assert config.out_dir == Path("/tmp/run")
        assert isinstance(config.out_dir, Path)

    def test_apply_list_override(self):
        config = make_config()
        cli    = ConfigCli(config)

        cli.apply(["--tags", "['x', 'y', 'z']"])

        assert config.tags == ["x", "y", "z"]

    def test_apply_dict_override(self):
        config = make_config()
        cli    = ConfigCli(config)

        cli.apply(["--options", "{'a': 2, 'b': 3}"])

        assert config.options == {"a": 2, "b": 3}

    def test_apply_tuple_override_from_list_literal(self):
        config = make_config()
        cli    = ConfigCli(config)

        cli.apply(["--shape", "[5, 6, 7]"])

        assert config.shape == (5, 6, 7)
        assert isinstance(config.shape, tuple)

    def test_apply_tuple_override_from_scalar_wraps(self):
        config = make_config()
        cli    = ConfigCli(config)

        cli.apply(["--shape", "9"])

        assert config.shape == (9,)

    def test_apply_dashed_alias_maps_to_same_dest(self):
        config = make_config()
        cli    = ConfigCli(config)

        cli.apply(["--inner.learning-rate", "0.5"])

        assert config.inner.learning_rate == pytest.approx(0.5)

    def test_apply_unknown_argument_is_ignored(self):
        config = make_config()
        cli    = ConfigCli(config)

        cli.apply(["--does-not-exist", "value", "--name", "kept"])

        assert config.name == "kept"

    def test_apply_multiple_overrides_recorded(self):
        config = make_config()
        cli    = ConfigCli(config)

        cli.apply(["--name", "multi", "--inner.epochs", "7"])

        assert cli.overrides == {"name": "multi", "inner.epochs": 7}


class TestCoerceDirect:
    def test_coerce_bool_invalid_raises(self):
        cli = ConfigCli(make_config())

        with pytest.raises(ValueError):
            cli._coerce("maybe", True)

    def test_coerce_none_evaluates_literal(self):
        cli = ConfigCli(make_config())

        assert cli._coerce("123", None) == 123
        assert cli._coerce("[1, 2]", None) == [1, 2]

    def test_coerce_none_falls_back_to_raw_string(self):
        cli = ConfigCli(make_config())

        assert cli._coerce("plain text", None) == "plain text"

    def test_coerce_none_override_via_apply(self):
        config = make_config()
        cli    = ConfigCli(config)

        cli.apply(["--free", "3.14"])

        assert config.free == pytest.approx(3.14)

    def test_coerce_int_from_string(self):
        cli = ConfigCli(make_config())

        assert cli._coerce("8", 0) == 8

    def test_coerce_default_returns_raw_for_unknown_type(self):
        cli = ConfigCli(make_config())

        class Custom:
            pass

        sentinel = Custom()
        assert cli._coerce("verbatim", sentinel) == "verbatim"


class TestSetPathAndOverrides:
    def test_set_path_nested(self):
        config = make_config()

        ConfigCli.set_path(config, "inner.epochs", 99)

        assert config.inner.epochs == 99

    def test_set_path_top_level(self):
        config = make_config()

        ConfigCli.set_path(config, "name", "set")

        assert config.name == "set"

    def test_apply_overrides_classmethod(self):
        config    = make_config()
        overrides = {"name": "ov", "inner.learning_rate": 0.2}

        returned = ConfigCli.apply_overrides(config, overrides)

        assert returned is config
        assert config.name == "ov"
        assert config.inner.learning_rate == pytest.approx(0.2)

    def test_apply_overrides_empty(self):
        config = make_config()

        ConfigCli.apply_overrides(config, {})

        assert config.name == "run"


class TestToMapping:
    def test_to_mapping_serializes_path_as_string(self):
        mapping = ConfigCli.to_mapping(make_config())

        assert mapping["out_dir"] == "results"
        assert isinstance(mapping["out_dir"], str)

    def test_to_mapping_serializes_tuple_as_list(self):
        mapping = ConfigCli.to_mapping(make_config())

        assert mapping["shape"] == [3, 4]
        assert isinstance(mapping["shape"], list)

    def test_to_mapping_keeps_supported_scalars(self):
        mapping = ConfigCli.to_mapping(make_config())

        assert mapping["name"] == "run"
        assert mapping["enabled"] is True
        assert mapping["inner.epochs"] == 10

    def test_to_mapping_keeps_none(self):
        mapping = ConfigCli.to_mapping(make_config())

        assert "free" in mapping
        assert mapping["free"] is None

    def test_to_mapping_is_json_serializable(self):
        mapping = ConfigCli.to_mapping(make_config())

        text = json.dumps(mapping)

        assert isinstance(text, str)


class TestSaveAndLoadResolved:
    def test_save_resolved_writes_file_and_returns_path(self, tmp_path):
        config = make_config()
        target = tmp_path / "nested" / "resolved.json"

        returned = ConfigCli.save_resolved(config, target)

        assert returned == target
        assert target.exists()

    def test_save_resolved_content_matches_mapping(self, tmp_path):
        config = make_config()
        target = tmp_path / "resolved.json"

        ConfigCli.save_resolved(config, target)
        with open(target, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded == ConfigCli.to_mapping(config)

    def test_load_resolved_round_trip(self, tmp_path):
        original = make_config()
        original.name = "saved"
        original.inner.epochs = 21
        original.out_dir = Path("/data/out")
        original.shape = (8, 9)
        target = tmp_path / "resolved.json"
        ConfigCli.save_resolved(original, target)

        fresh = make_config()
        ConfigCli.load_resolved(fresh, target)

        assert fresh.name == "saved"
        assert fresh.inner.epochs == 21
        assert fresh.out_dir == Path("/data/out")
        assert isinstance(fresh.out_dir, Path)
        assert fresh.shape == (8, 9)
        assert isinstance(fresh.shape, tuple)

    def test_load_resolved_missing_file_returns_config_unchanged(self, tmp_path):
        config = make_config()
        missing = tmp_path / "absent.json"

        returned = ConfigCli.load_resolved(config, missing)

        assert returned is config
        assert config.name == "run"

    def test_load_resolved_ignores_unknown_keys(self, tmp_path):
        target = tmp_path / "partial.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump({"name": "only", "unknown_key": 5}, f)

        config = make_config()
        ConfigCli.load_resolved(config, target)

        assert config.name == "only"
        assert not hasattr(config, "unknown_key")

    def test_load_resolved_partial_keys_leave_others_default(self, tmp_path):
        target = tmp_path / "partial.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump({"inner.epochs": 55}, f)

        config = make_config()
        ConfigCli.load_resolved(config, target)

        assert config.inner.epochs == 55
        assert config.name == "run"


class TestToArgv:
    def test_to_argv_bool_true(self):
        argv = ConfigCli.to_argv({"enabled": True})

        assert argv == ["--enabled", "true"]

    def test_to_argv_bool_false(self):
        argv = ConfigCli.to_argv({"enabled": False})

        assert argv == ["--enabled", "false"]

    def test_to_argv_tuple_rendered_as_list(self):
        argv = ConfigCli.to_argv({"shape": (3, 4)})

        assert argv == ["--shape", "[3, 4]"]

    def test_to_argv_scalar(self):
        argv = ConfigCli.to_argv({"inner.epochs": 12})

        assert argv == ["--inner.epochs", "12"]

    def test_to_argv_empty(self):
        assert ConfigCli.to_argv({}) == []

    def test_to_argv_round_trips_through_apply(self):
        overrides = {"name": "rt", "inner.epochs": 3, "enabled": False, "shape": (1, 2)}
        argv      = ConfigCli.to_argv(overrides)

        config = make_config()
        ConfigCli(config).apply(argv)

        assert config.name == "rt"
        assert config.inner.epochs == 3
        assert config.enabled is False
        assert config.shape == (1, 2)

    def test_to_argv_multiple_preserves_pairs(self):
        argv = ConfigCli.to_argv({"a": 1, "b": 2})

        assert argv == ["--a", "1", "--b", "2"]


class TestHelpAndDetachFlags:
    def test_help_config_raises_systemexit_zero(self, capsys):
        cli = ConfigCli(make_config())

        with pytest.raises(SystemExit) as exc:
            cli.apply(["--help-config"])

        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "Configuration overrides" in out
        assert "--inner.epochs" in out

    def test_print_config_help_lists_execution_flags(self, capsys):
        cli = ConfigCli(make_config())

        cli._print_config_help()

        out = capsys.readouterr().out
        assert "--detach" in out
        assert "Execution flags" in out

    def test_detach_flag_triggers_detacher_ensure(self, monkeypatch):
        import tools.config_cli as mod

        called = {"ensure": False}

        class FakeDetacher:
            def ensure(self):
                called["ensure"] = True

        monkeypatch.setattr(mod, "Detacher", FakeDetacher)

        cli = ConfigCli(make_config())
        cli.apply(["--detach"])

        assert called["ensure"] is True

    def test_no_detach_flag_does_not_trigger_ensure(self, monkeypatch):
        import tools.config_cli as mod

        called = {"ensure": False}

        class FakeDetacher:
            def ensure(self):
                called["ensure"] = True

        monkeypatch.setattr(mod, "Detacher", FakeDetacher)

        cli = ConfigCli(make_config())
        cli.apply(["--name", "x"])

        assert called["ensure"] is False


class TestDetacher:
    def test_requested_detects_flags(self):
        det = Detacher()

        assert det.requested(["--detach"]) is True
        assert det.requested(["--nohup"]) is True
        assert det.requested(["--other"]) is False
        assert det.requested([]) is False

    def test_active_reads_env_flag(self, monkeypatch):
        det = Detacher()

        monkeypatch.setenv(Detacher.ENV_FLAG, "1")
        assert det.active() is True

        monkeypatch.setenv(Detacher.ENV_FLAG, "0")
        assert det.active() is False

        monkeypatch.delenv(Detacher.ENV_FLAG, raising=False)
        assert det.active() is False

    def test_ensure_active_ignores_sighup_and_returns(self, monkeypatch):
        import signal

        monkeypatch.setenv(Detacher.ENV_FLAG, "1")
        recorded = {}

        def fake_signal(sig, handler):
            recorded["sig"]     = sig
            recorded["handler"] = handler

        monkeypatch.setattr(signal, "signal", fake_signal)

        Detacher().ensure()

        assert recorded["sig"] == signal.SIGHUP
        assert recorded["handler"] == signal.SIG_IGN

    def test_ensure_not_requested_returns_without_relaunch(self, monkeypatch):
        monkeypatch.delenv(Detacher.ENV_FLAG, raising=False)
        det = Detacher()
        monkeypatch.setattr(det, "requested", lambda argv=None: False)

        assert det.ensure() is None

    def test_ensure_requested_relaunches_via_subprocess(self, tmp_path, monkeypatch):
        import tools.config_cli as mod

        monkeypatch.delenv(Detacher.ENV_FLAG, raising=False)
        det = Detacher(log_dir=str(tmp_path / "logs"))
        monkeypatch.setattr(det, "requested", lambda argv=None: True)

        captured = {}

        class FakeProcess:
            pid = 4321

        def fake_popen(cmd, **kwargs):
            captured["cmd"]    = cmd
            captured["env"]    = kwargs["env"]
            captured["kwargs"] = kwargs
            return FakeProcess()

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

        with pytest.raises(SystemExit) as exc:
            det.ensure()

        assert exc.value.code == 0
        assert captured["env"][Detacher.ENV_FLAG] == "1"
        assert captured["env"]["PYTHONUNBUFFERED"] == "1"
        assert captured["kwargs"]["start_new_session"] is True
        assert (tmp_path / "logs").exists()


class TestRepresentationChannelsPerPass:
    @pytest.mark.parametrize("rep", list(Representation))
    def test_channels_per_pass_matches_table(self, rep):
        assert rep.channels_per_pass == _CHANNELS_PER_PASS[rep.value]

    def test_channels_per_pass_values(self):
        assert Representation.REAL_IMAG.channels_per_pass     == 2
        assert Representation.MAG_REAL_IMAG.channels_per_pass == 3
        assert Representation.MAG_ANGLE.channels_per_pass     == 2
        assert Representation.MAG_RI_ANGLE.channels_per_pass  == 4
        assert Representation.ANGLE_ONLY.channels_per_pass    == 1
        assert Representation.MAG_ONLY.channels_per_pass      == 1

    @pytest.mark.parametrize("rep", list(Representation))
    def test_slot_kinds_length_equals_channels_per_pass(self, rep):
        assert len(rep.slot_kinds) == rep.channels_per_pass

    def test_enum_values_match_table_keys(self):
        assert {r.value for r in Representation} == set(_CHANNELS_PER_PASS.keys())


class TestRepresentationChannels:
    @staticmethod
    def make_data(n_samples=2, n_passes=3, h=4, w=5, seed=0):
        rng  = np.random.default_rng(seed)
        real = rng.standard_normal((n_samples, n_passes, h, w))
        imag = rng.standard_normal((n_samples, n_passes, h, w))
        return (real + 1j * imag).astype(np.complex64)

    def test_real_imag_channels(self):
        data     = self.make_data()
        channels = Representation.REAL_IMAG._channels(data)

        assert len(channels) == 2
        assert np.allclose(channels[0], data.real)
        assert np.allclose(channels[1], data.imag)

    def test_mag_only_channels(self):
        data     = self.make_data()
        channels = Representation.MAG_ONLY._channels(data)

        assert len(channels) == 1
        assert np.allclose(channels[0], np.abs(data))

    def test_angle_only_channels(self):
        data     = self.make_data()
        channels = Representation.ANGLE_ONLY._channels(data)

        assert len(channels) == 1
        assert np.allclose(channels[0], np.angle(data))

    def test_mag_angle_channels(self):
        data     = self.make_data()
        channels = Representation.MAG_ANGLE._channels(data)

        assert len(channels) == 2
        assert np.allclose(channels[0], np.abs(data))
        assert np.allclose(channels[1], np.angle(data))

    def test_mag_real_imag_normalizes_by_magnitude(self):
        data     = self.make_data()
        channels = Representation.MAG_REAL_IMAG._channels(data)
        mag      = np.abs(data)

        assert len(channels) == 3
        assert np.allclose(channels[0], mag)
        assert np.allclose(channels[1], data.real / mag)
        assert np.allclose(channels[2], data.imag / mag)

    def test_mag_real_imag_safe_divide_on_zero(self):
        data = np.zeros((1, 1, 2, 2), dtype=np.complex64)

        channels = Representation.MAG_REAL_IMAG._channels(data)

        assert np.all(np.isfinite(channels[1]))
        assert np.all(channels[1] == 0.0)
        assert np.all(channels[2] == 0.0)

    def test_mag_ri_angle_channels(self):
        data     = self.make_data()
        channels = Representation.MAG_RI_ANGLE._channels(data)
        mag      = np.abs(data)

        assert len(channels) == 4
        assert np.allclose(channels[0], mag)
        assert np.allclose(channels[1], data.real / mag)
        assert np.allclose(channels[2], data.imag / mag)
        assert np.allclose(channels[3], np.angle(data))


class TestRepresentationConvert:
    @staticmethod
    def make_data(n_samples=2, n_passes=3, h=4, w=5, seed=1):
        rng  = np.random.default_rng(seed)
        real = rng.standard_normal((n_samples, n_passes, h, w))
        imag = rng.standard_normal((n_samples, n_passes, h, w))
        return (real + 1j * imag).astype(np.complex64)

    @pytest.mark.parametrize("rep", list(Representation))
    def test_convert_output_shape_and_dtype(self, rep):
        data = self.make_data()
        n_samples, n_passes, h, w = data.shape

        out = rep.convert(data)

        assert out.shape == (n_samples, n_passes * rep.channels_per_pass, h, w)
        assert out.dtype == np.float32

    def test_convert_interleaves_channels_per_pass(self):
        data = self.make_data()
        out  = Representation.REAL_IMAG.convert(data)

        assert np.allclose(out[:, 0::2], data.real.astype(np.float32))
        assert np.allclose(out[:, 1::2], data.imag.astype(np.float32))

    def test_convert_mag_only_single_channel_per_pass(self):
        data = self.make_data()
        out  = Representation.MAG_ONLY.convert(data)

        assert out.shape[1] == data.shape[1]
        assert np.allclose(out, np.abs(data).astype(np.float32))

    def test_convert_single_pass_single_pixel(self):
        data = np.array([[[[1 + 2j]]]], dtype=np.complex64)

        out = Representation.MAG_ANGLE.convert(data)

        assert out.shape == (1, 2, 1, 1)
        assert np.isclose(out[0, 0, 0, 0], np.abs(data[0, 0, 0, 0]))
        assert np.isclose(out[0, 1, 0, 0], np.angle(data[0, 0, 0, 0]))


class TestRepresentationConvertInto:
    @staticmethod
    def make_data(n_passes=3, h=4, w=5, seed=2):
        rng  = np.random.default_rng(seed)
        real = rng.standard_normal((n_passes, h, w))
        imag = rng.standard_normal((n_passes, h, w))
        return (real + 1j * imag).astype(np.complex64)

    @pytest.mark.parametrize("rep", list(Representation))
    def test_convert_into_fills_provided_buffer(self, rep):
        data = self.make_data()
        n_passes, h, w = data.shape
        out = np.zeros((n_passes * rep.channels_per_pass, h, w), dtype=np.float32)

        rep.convert_into(out, data)

        expected_channels = rep._channels(data)
        cpp               = rep.channels_per_pass
        for c, arr in enumerate(expected_channels):
            assert np.allclose(out[c::cpp], arr.astype(np.float32))

    def test_convert_into_real_imag_layout(self):
        data = self.make_data()
        out  = np.zeros((data.shape[0] * 2, data.shape[1], data.shape[2]), dtype=np.float32)

        Representation.REAL_IMAG.convert_into(out, data)

        assert np.allclose(out[0::2], data.real.astype(np.float32))
        assert np.allclose(out[1::2], data.imag.astype(np.float32))
