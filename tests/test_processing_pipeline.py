from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from configuration.processing_config import (
    ParallelConfiguration,
    PathConfiguration,
    ProcessingConfiguration,
    TomogramConfiguration,
)
from pipelines.processing_pipeline.artifacts      import ArtifactRegistry, MetadataManager
from pipelines.processing_pipeline.interferogram  import InterferogramBuilder
from pipelines.processing_pipeline.tomogram_worker import PyRatJob, run_pyrat
from tools.logger                                 import NullLogger
from tools.regions                                import CropRegion


def _make_config(tmp_path: Path, **overrides) -> ProcessingConfiguration:
    crop  = overrides.pop("crop", CropRegion(0, 100, 0, 50))
    paths = PathConfiguration(
        main_directory  = tmp_path,
        pyrat_directory = tmp_path / "pyrat",
    )
    config = ProcessingConfiguration(
        crop     = crop,
        paths    = paths,
        parallel = overrides.pop("parallel", ParallelConfiguration(tomogram_workers=1, pyrat_threads=1)),
        **overrides,
    )
    return config


class TestPyRatJob:
    def test_required_fields_are_assigned(self):
        job = PyRatJob(
            pyrat_root_path       = "/some/pyrat",
            crop_tuple            = (0, 10, 0, 5),
            suffix                = "0001",
            fusar_project_path    = "proj.csv",
            stack_identifier      = "stack",
            base_directory        = "/base",
            polarisation          = "hv",
            track_selection       = "*",
            height_range          = (-20.0, 80.0),
            filter_method         = "Boxcar",
            filter_arguments      = {"win": [20, 10]},
            beamforming_method    = "Capon",
            beamforming_arguments = [],
            output_directory      = "/out",
            apply_resampling      = False,
            apply_presumming      = True,
            pyrat_threads         = 4,
        )

        assert job.crop_tuple == (0, 10, 0, 5)
        assert job.suffix == "0001"
        assert job.apply_presumming is True
        assert job.apply_resampling is False
        assert job.pyrat_threads == 4

    def test_parent_sys_path_defaults_to_none(self):
        job = PyRatJob(
            pyrat_root_path       = "/some/pyrat",
            crop_tuple            = (0, 10, 0, 5),
            suffix                = "0001",
            fusar_project_path    = "proj.csv",
            stack_identifier      = "stack",
            base_directory        = "/base",
            polarisation          = "hv",
            track_selection       = "*",
            height_range          = (-20.0, 80.0),
            filter_method         = "Boxcar",
            filter_arguments      = {},
            beamforming_method    = "Capon",
            beamforming_arguments = [],
            output_directory      = "/out",
            apply_resampling      = False,
            apply_presumming      = False,
            pyrat_threads         = 1,
        )

        assert job.parent_sys_path is None


class TestRunPyRat:
    def _make_job(self, **overrides) -> PyRatJob:
        base = dict(
            pyrat_root_path       = "/nonexistent/pyrat_root_marker",
            crop_tuple            = (0, 10, 0, 5),
            suffix                = "0001",
            fusar_project_path    = "proj.csv",
            stack_identifier      = "stack",
            base_directory        = "/base",
            polarisation          = "hv",
            track_selection       = "*",
            height_range          = (-20.0, 80.0),
            filter_method         = "Boxcar",
            filter_arguments      = {},
            beamforming_method    = "Capon",
            beamforming_arguments = [],
            output_directory      = "/out",
            apply_resampling      = False,
            apply_presumming      = False,
            pyrat_threads         = 2,
        )
        base.update(overrides)
        return PyRatJob(**base)

    def _install_fake_pyrat(self, monkeypatch, recorder):
        fake_pyrat = types.ModuleType("pyrat")

        def pyrat_init(*args, **kwargs):
            recorder["init_kwargs"] = kwargs

        tomo_module = types.ModuleType("pyrat.tomo")

        def fusartomo(**kwargs):
            recorder["fusartomo_kwargs"] = kwargs

        tomo_module.fusartomo = fusartomo
        fake_pyrat.pyrat_init = pyrat_init
        fake_pyrat.tomo       = tomo_module

        monkeypatch.setitem(sys.modules, "pyrat", fake_pyrat)
        monkeypatch.setitem(sys.modules, "pyrat.tomo", tomo_module)

    def test_run_pyrat_invokes_fusartomo_and_returns_zero(self, monkeypatch):
        recorder = {}
        self._install_fake_pyrat(monkeypatch, recorder)
        monkeypatch.setattr(sys, "path", list(sys.path))

        job    = self._make_job()
        result = run_pyrat(job)

        assert result == 0
        assert recorder["fusartomo_kwargs"]["crop"] == job.crop_tuple
        assert recorder["fusartomo_kwargs"]["suffix"] == "0001"
        assert recorder["fusartomo_kwargs"]["range"] == list(job.height_range)
        assert recorder["init_kwargs"]["nthreads"] == job.pyrat_threads

    def test_run_pyrat_sets_qt_platform_env(self, monkeypatch):
        recorder = {}
        self._install_fake_pyrat(monkeypatch, recorder)
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
        monkeypatch.setattr(sys, "path", list(sys.path))

        import os

        run_pyrat(self._make_job())

        assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"

    def test_run_pyrat_inserts_pyrat_root_into_sys_path(self, monkeypatch):
        recorder = {}
        self._install_fake_pyrat(monkeypatch, recorder)
        monkeypatch.setattr(sys, "path", list(sys.path))

        job = self._make_job()
        run_pyrat(job)

        assert job.pyrat_root_path in sys.path

    def test_run_pyrat_restores_parent_sys_path(self, monkeypatch):
        recorder = {}
        self._install_fake_pyrat(monkeypatch, recorder)
        parent_path = ["/marker_a", "/marker_b"]
        monkeypatch.setattr(sys, "path", list(sys.path))

        job = self._make_job(parent_sys_path=parent_path)
        run_pyrat(job)

        assert sys.path[:2] == ["/marker_a", "/marker_b"] or job.pyrat_root_path in sys.path
        assert "/marker_b" in sys.path


class TestArtifactRegistry:
    def test_artifact_filenames_contains_all_keys(self, tmp_path):
        config   = _make_config(tmp_path)
        registry = ArtifactRegistry(config, NullLogger())

        names = registry.artifact_filenames()

        expected_keys = {
            "tomogram_full", "tomogram_reduced", "dem_full", "dem_reduced",
            "primary_reduced", "secondaries_reduced", "interferograms_reduced",
        }
        assert set(names.keys()) == expected_keys
        assert all(value.endswith(".npy") for value in names.values())

    def test_artifact_filenames_embed_tags(self, tmp_path):
        config   = _make_config(tmp_path)
        registry = ArtifactRegistry(config, NullLogger())
        names    = registry.artifact_filenames()

        assert config.parameter_tag in names["tomogram_full"]
        assert config.parameter_tag in names["dem_full"]
        assert config.tomogram_tag  in names["tomogram_reduced"]
        assert config.tomogram_tag  in names["primary_reduced"]

    def test_artifact_path_under_data_directory(self, tmp_path):
        config   = _make_config(tmp_path)
        registry = ArtifactRegistry(config, NullLogger())

        path = registry.artifact_path("tomogram_reduced")

        assert path.parent == config.paths.data_directory
        assert path.name == registry.artifact_filenames()["tomogram_reduced"]

    def test_artifact_path_unknown_type_raises_keyerror(self, tmp_path):
        config   = _make_config(tmp_path)
        registry = ArtifactRegistry(config, NullLogger())

        with pytest.raises(KeyError):
            registry.artifact_path("does_not_exist")

    def test_ensure_directory_structure_creates_directories(self, tmp_path):
        config   = _make_config(tmp_path)
        registry = ArtifactRegistry(config, NullLogger())

        registry.ensure_directory_structure()

        assert config.paths.data_directory.exists()
        assert config.paths.metadata_directory.exists()
        assert config.paths.temporary_directory.exists()

    def test_ensure_directory_structure_idempotent(self, tmp_path):
        config   = _make_config(tmp_path)
        registry = ArtifactRegistry(config, NullLogger())

        registry.ensure_directory_structure()
        registry.ensure_directory_structure()

        assert config.paths.data_directory.exists()


class TestMetadataManager:
    def test_build_tomogram_metadata_fields(self, tmp_path):
        config  = _make_config(tmp_path)
        manager = MetadataManager(config, NullLogger())
        cfg     = config.input_configs
        out     = tmp_path / "tomo.npy"

        meta = manager.build_tomogram_metadata("reduced", out, "stack_x", cfg)

        assert meta["tomo_reduced"] == str(out)
        assert meta["id"] == "stack_x"
        assert meta["polarisation"] == cfg.polarisation
        assert meta["filter"] == cfg.filter_method
        assert meta["method"] == cfg.beamforming_method
        assert meta["win"] == "[20, 10]"

    def test_build_tomogram_metadata_crop_formatting(self, tmp_path):
        config  = _make_config(tmp_path, crop=CropRegion(1, 2, 3, 4))
        manager = MetadataManager(config, NullLogger())

        meta = manager.build_tomogram_metadata("full", tmp_path / "f.npy", "sid", config.input_configs)

        assert meta["crop"] == "[1, 2, 3, 4]"

    def test_build_inputs_metadata_shapes_and_paths(self, tmp_path):
        config  = _make_config(tmp_path)
        manager = MetadataManager(config, NullLogger())

        meta = manager.build_inputs_metadata(
            primary_path         = tmp_path / "p.npy",
            secondaries_path     = tmp_path / "s.npy",
            interferograms_path  = tmp_path / "i.npy",
            primary_shape        = (100, 50),
            secondaries_shape    = (8, 100, 50),
            interferograms_shape = (8, 100, 50),
        )

        assert meta["primary_shape"] == "[100, 50]"
        assert meta["secondaries_shape"] == "[8, 100, 50]"
        assert meta["interferograms_shape"] == "[8, 100, 50]"
        assert meta["data_type"] == config.dataset_type
        assert meta["id"] == config.reduced_stack_identifier

    def test_save_stage_metadata_writes_file(self, tmp_path):
        config  = _make_config(tmp_path)
        manager = MetadataManager(config, NullLogger())

        meta_path = manager.save_stage_metadata("inputs", "tagX", {"a": "1", "b": "2"})

        assert meta_path.exists()
        assert meta_path.name == "meta_inputs_tagX.txt"
        content = meta_path.read_text(encoding="utf-8")
        assert "a: 1" in content
        assert "b: 2" in content

    def test_save_pipeline_configuration_writes_json(self, tmp_path):
        config  = _make_config(tmp_path)
        manager = MetadataManager(config, NullLogger())

        dump_path = manager.save_pipeline_configuration()

        assert dump_path.exists()
        assert dump_path.suffix == ".json"
        import json
        loaded = json.loads(dump_path.read_text(encoding="utf-8"))
        assert "dataset_type" in loaded

    def test_save_dataset_layout_writes_json(self, tmp_path):
        config  = _make_config(tmp_path)
        manager = MetadataManager(config, NullLogger())

        out_path = manager.save_dataset_layout()

        assert out_path.exists()
        assert out_path.name == "dataset.json"
        import json
        layout = json.loads(out_path.read_text(encoding="utf-8"))
        assert layout["dataset_type"] == config.dataset_type
        assert layout["global_crop"] == list(config.crop.as_tuple())
        assert set(layout["artifacts"].keys()) == set(ArtifactRegistry(config, NullLogger()).artifact_filenames().keys())


class TestInterferogramBuilder:
    def test_build_non_fsar_raises_not_implemented(self, tmp_path):
        config = _make_config(tmp_path, dataset_type="OTHER")
        builder = InterferogramBuilder(config, NullLogger())

        with pytest.raises(NotImplementedError):
            builder.build((0, 10, 0, 5))

    def test_init_adds_pyrat_directory_to_sys_path(self, tmp_path):
        config = _make_config(tmp_path)
        pyrat_root = str(config.paths.pyrat_directory)

        if pyrat_root in sys.path:
            sys.path.remove(pyrat_root)

        InterferogramBuilder(config, NullLogger())

        assert pyrat_root in sys.path
        sys.path.remove(pyrat_root)

    def test_run_saves_arrays_and_returns_shapes(self, tmp_path, monkeypatch):
        config  = _make_config(tmp_path)
        builder = InterferogramBuilder(config, NullLogger())

        primary        = np.zeros((4, 3), dtype=np.complex64)
        secondaries    = np.zeros((2, 4, 3), dtype=np.complex64)
        interferograms = np.zeros((2, 4, 3), dtype=np.complex64)

        monkeypatch.setattr(builder, "build", lambda crop_tuple: (primary, secondaries, interferograms))

        primary_path        = tmp_path / "out" / "p.npy"
        secondaries_path    = tmp_path / "out" / "s.npy"
        interferograms_path = tmp_path / "out" / "i.npy"

        shapes = builder.run((0, 4, 0, 3), primary_path, secondaries_path, interferograms_path)

        assert shapes == ((4, 3), (2, 4, 3), (2, 4, 3))
        assert primary_path.exists()
        assert secondaries_path.exists()
        assert interferograms_path.exists()
        assert np.load(str(primary_path)).shape == (4, 3)

    def test_run_creates_parent_directories(self, tmp_path, monkeypatch):
        config  = _make_config(tmp_path)
        builder = InterferogramBuilder(config, NullLogger())

        arr = np.zeros((2, 2), dtype=np.complex64)
        monkeypatch.setattr(builder, "build", lambda crop_tuple: (arr, arr, arr))

        nested = tmp_path / "a" / "b" / "c"
        builder.run((0, 2, 0, 2), nested / "p.npy", nested / "s.npy", nested / "i.npy")

        assert nested.exists()


class TestTomogramProcessor:
    def _processor(self, tmp_path, **config_overrides):
        pytest.importorskip("h5py")
        from pipelines.processing_pipeline.tomogram import TomogramProcessor

        config = _make_config(tmp_path, **config_overrides)
        return TomogramProcessor(config, NullLogger()), config

    def test_divide_crop_single_section_when_within_limit(self, tmp_path):
        processor, config = self._processor(tmp_path, crop=CropRegion(0, 500, 0, 50))
        tomo_cfg = TomogramConfiguration(max_crop_azimuth_width=1000)

        sections = processor._divide_crop(tomo_cfg)

        assert sections == [config.crop.as_tuple()]

    def test_divide_crop_subdivides_when_exceeding_limit(self, tmp_path):
        processor, _ = self._processor(tmp_path, crop=CropRegion(0, 2500, 0, 50))
        tomo_cfg = TomogramConfiguration(max_crop_azimuth_width=1000)

        sections = processor._divide_crop(tomo_cfg)

        assert len(sections) == 3
        assert sections[0] == (0, 1000, 0, 50)
        assert sections[1] == (1000, 2000, 0, 50)
        assert sections[2] == (2000, 2500, 0, 50)

    def test_create_temp_creates_directory_under_temporary(self, tmp_path):
        processor, config = self._processor(tmp_path)

        temp_dir = processor._create_temp()

        assert temp_dir.exists()
        assert temp_dir.parent == config.paths.temporary_directory
        assert temp_dir.name.startswith("tomo_")

    def test_cleanup_temp_removes_directory(self, tmp_path):
        processor, _ = self._processor(tmp_path)
        temp_dir = processor._create_temp()
        (temp_dir / "marker.txt").write_text("x", encoding="utf-8")

        processor._cleanup_temp(temp_dir)

        assert not temp_dir.exists()

    def test_cleanup_temp_missing_directory_is_safe(self, tmp_path):
        processor, _ = self._processor(tmp_path)
        missing = tmp_path / "never_created"

        processor._cleanup_temp(missing)

        assert not missing.exists()

    def test_save_writes_both_arrays(self, tmp_path):
        processor, _ = self._processor(tmp_path)
        tomogram = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        dem      = np.arange(6, dtype=np.float32).reshape(3, 2)

        tomo_path = tmp_path / "save" / "tomo.npy"
        dem_path  = tmp_path / "save" / "dem.npy"

        processor._save(tomo_path, dem_path, tomogram, dem)

        assert np.array_equal(np.load(str(tomo_path)), tomogram)
        assert np.array_equal(np.load(str(dem_path)), dem)

    def test_concatenate_merges_partial_artifacts(self, tmp_path):
        import h5py
        processor, _ = self._processor(tmp_path)

        temp_dir = tmp_path / "tomo_temp"
        partial_dir = temp_dir / "TOMO" / "TOMO-SR"
        partial_dir.mkdir(parents=True, exist_ok=True)

        dem_a = np.arange(6, dtype=np.float32).reshape(3, 2)
        dem_b = (np.arange(4, dtype=np.float32) + 100.0).reshape(2, 2)
        tomo_a = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
        tomo_b = (np.arange(2 * 2 * 2, dtype=np.float32) + 200.0).reshape(2, 2, 2)

        for idx, (dem, tomo) in enumerate([(dem_a, tomo_a), (dem_b, tomo_b)]):
            with h5py.File(str(partial_dir / f"part_{idx:04d}.h5"), "w") as f:
                f.create_dataset("DEM", data=dem)
                f.create_dataset("tomogram", data=tomo)

        combined_dem, combined_tomo = processor._concatenate(temp_dir)

        assert combined_dem.shape == (5, 2)
        assert combined_tomo.shape == (2, 5, 2)
        assert np.array_equal(combined_dem[:3], dem_a)
        assert np.array_equal(combined_dem[3:], dem_b)
        assert np.array_equal(combined_tomo[:, :3, :], tomo_a)
        assert np.array_equal(combined_tomo[:, 3:, :], tomo_b)


class TestProcessingPipeline:
    def _build_pipeline(self, tmp_path, monkeypatch):
        pytest.importorskip("h5py")
        from pipelines.processing_pipeline import pipeline as pipeline_module

        config = _make_config(tmp_path)

        recorder = {"tomograms": [], "inputs": []}

        class StubTomogramProcessor:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, tomogram_path, dem_path, stack_identifier, tomogram_config):
                recorder["tomograms"].append((tomogram_path, dem_path, stack_identifier))
                return tomogram_path, dem_path

        class StubInterferogramBuilder:
            def __init__(self, *args, **kwargs):
                self.track_baselines = None

            def run(self, crop_tuple, primary_path, secondaries_path, interferograms_path):
                recorder["inputs"].append((primary_path, secondaries_path, interferograms_path))
                return (4, 3), (2, 4, 3), (2, 4, 3)

        monkeypatch.setattr(pipeline_module, "TomogramProcessor", StubTomogramProcessor)
        monkeypatch.setattr(pipeline_module, "InterferogramBuilder", StubInterferogramBuilder)

        pipe = pipeline_module.ProcessingPipeline(config, logger=NullLogger())
        return pipe, config, recorder

    def test_resolve_variant_full(self, tmp_path, monkeypatch):
        pipe, config, _ = self._build_pipeline(tmp_path, monkeypatch)

        variant = pipe._resolve_variant("full")

        assert variant.stack_identifier == config.full_stack_identifier
        assert variant.identifier_tag == config.parameter_tag
        assert variant.tomogram_config is config.output_config

    def test_resolve_variant_reduced(self, tmp_path, monkeypatch):
        pipe, config, _ = self._build_pipeline(tmp_path, monkeypatch)

        variant = pipe._resolve_variant("reduced")

        assert variant.stack_identifier == config.reduced_stack_identifier
        assert variant.identifier_tag == config.tomogram_tag
        assert variant.tomogram_config is config.input_configs

    def test_stage_tomogram_returns_artifact_paths(self, tmp_path, monkeypatch):
        pipe, config, recorder = self._build_pipeline(tmp_path, monkeypatch)

        tomo_path, dem_path = pipe._stage_tomogram("reduced")

        assert tomo_path == pipe.artifact_registry.artifact_path("tomogram_reduced")
        assert dem_path == pipe.artifact_registry.artifact_path("dem_reduced")
        assert len(recorder["tomograms"]) == 1

    def test_stage_inputs_returns_three_paths(self, tmp_path, monkeypatch):
        pipe, config, recorder = self._build_pipeline(tmp_path, monkeypatch)

        primary, secondaries, interferograms = pipe._stage_inputs()

        assert primary == pipe.artifact_registry.artifact_path("primary_reduced")
        assert secondaries == pipe.artifact_registry.artifact_path("secondaries_reduced")
        assert interferograms == pipe.artifact_registry.artifact_path("interferograms_reduced")
        assert len(recorder["inputs"]) == 1

    def test_run_returns_full_artifact_mapping(self, tmp_path, monkeypatch):
        pipe, config, recorder = self._build_pipeline(tmp_path, monkeypatch)

        outputs = pipe.run()

        expected_keys = {
            "tomogram_full", "tomogram_reduced", "dem_full", "dem_reduced",
            "primary_reduced", "secondaries_reduced", "interferograms_reduced",
            "run_directory",
        }
        assert set(outputs.keys()) == expected_keys
        assert outputs["run_directory"] == config.paths.run_directory
        assert len(recorder["tomograms"]) == 2
        assert len(recorder["inputs"]) == 1

    def test_run_writes_metadata_and_layout(self, tmp_path, monkeypatch):
        pipe, config, _ = self._build_pipeline(tmp_path, monkeypatch)

        pipe.run()

        dataset_json = config.paths.data_directory / "dataset.json"
        assert dataset_json.exists()
        meta_files = list(config.paths.metadata_directory.glob("*.txt"))
        assert len(meta_files) >= 3
