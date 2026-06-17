from __future__ import annotations

import pytest
import torch

from configuration.dataset                  import DatasetConfig, InputConfig, PatchConfig, Representation, SplitRegions
from configuration.sar.gaussian_config      import GaussianConfig
from configuration.training.backbone        import BackboneTrainerConfig
from pipelines.backbone.training.pipeline   import TrainingPipeline
from tools.data.regions                     import CropRegion

from tests.backbone_training._helpers import geometry_config


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _dataset_config(test_data_dir, params_dir) -> DatasetConfig:
    input_config = InputConfig(
        use_primary        = True, primary_representation        = Representation.MAG_ONLY,
        use_secondaries    = True, secondaries_representation    = Representation.MAG_ONLY,
        use_interferograms = True, interferograms_representation = Representation.ANGLE_ONLY,
    )

    splits = SplitRegions(
        train = CropRegion(1000, 1024, 500, 524),
        val   = CropRegion(1024, 1040, 500, 524),
        test  = CropRegion(1040, 1056, 500, 524),
    )

    return DatasetConfig(
        preprocessing_run_directory = test_data_dir,
        parameters_path             = params_dir / "parameters.npy",
        split_regions               = splits,
        secondary_labels            = ("FL01_PS04", "FL01_PS06"),
        patch                       = PatchConfig(size=(8, 8), stride=8),
        input_config                = input_config,
        batch_size                  = 2,
        num_workers                 = 0,
        prefetch_factor             = 8,
        n_gaussians                 = 5,
    )


def _trainer_config(test_data_dir, tmp_path) -> BackboneTrainerConfig:
    gaussian = GaussianConfig.from_dataset(test_data_dir, n_gaussians=5)
    config   = BackboneTrainerConfig(gaussian=gaussian)

    config.io.logdir                  = str(tmp_path)
    config.io.writer                  = None
    config.training.epochs            = 1
    config.training.validation_frequency = 1
    config.resources.enabled          = False
    config.geometry                   = geometry_config()

    config.curriculum.warmup.use_param_l1    = True
    config.curriculum.warmup.weight_param_l1 = 1.0

    return config


@pytest.mark.real_data
@pytest.mark.slow
def test_training_pipeline_end_to_end_produces_checkpoint(test_data_dir, params_dir, tmp_path):
    dataset_config = _dataset_config(test_data_dir, params_dir)
    trainer_config = _trainer_config(test_data_dir, tmp_path)

    pipeline = TrainingPipeline(
        trainer_config = trainer_config,
        dataset_config = dataset_config,
        backbone_name  = "resunet",
        model_config   = None,
        seed           = 0,
        run_name       = "pipeline_smoke",
    )

    train_losses, val_losses, best_val = pipeline.run(probe_config=None)

    run_directory = pipeline.run_metadata.run_directory

    assert (run_directory / "best_model.pt").exists()
    assert (run_directory / "meta").is_dir()
    assert len(train_losses) == 1
    assert val_losses is not None


@pytest.mark.real_data
@pytest.mark.slow
def test_training_pipeline_saves_configs(test_data_dir, params_dir, tmp_path):
    dataset_config = _dataset_config(test_data_dir, params_dir)
    trainer_config = _trainer_config(test_data_dir, tmp_path)

    pipeline = TrainingPipeline(
        trainer_config = trainer_config,
        dataset_config = dataset_config,
        backbone_name  = "resunet",
        model_config   = None,
        seed           = 0,
        run_name       = "pipeline_cfg",
    )

    pipeline.run(probe_config=None)

    meta_dir = pipeline.run_metadata.run_directory / "meta"
    written  = list(meta_dir.glob("*.json")) + list(meta_dir.glob("*.yaml")) + list(meta_dir.glob("*.txt"))

    assert meta_dir.is_dir()
    assert len(written) > 0
