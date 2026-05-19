from __future__ import annotations

import os
os.environ["MKL_NUM_THREADS"]     = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"]     = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import sys
from dataclasses import dataclass, field
from pathlib     import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from autoencoder import (
    AutoencoderConfig,
    AutoencoderPipeline,
    BackboneType,
    DecoderConfig,
    EncoderConfig,
    IOConfig,
    LossConfig,
    TrainerConfig,
)
from pipelines.autoencoder_pipeline.config import AugmentationConfig, ContrastiveView, DataConfig, ReconLossName
from configuration.dataset_config       import (
    DatasetCreationConfiguration,
    InputConfig,
    PassDropConfig,
    PatchConfiguration,
    Representation,
    SplitRegions,
)
from tools.crop_region import CropRegion


@dataclass
class RunConfig:
    dataset_path : Path = Path("/ste/rnd/User/vice_vi/Dataset/base_dataset")

    batch_size  : int = 1024
    num_workers : int = 8     

    patch_size   : tuple[int, int] = (64, 64)
    patch_stride : int             = 32

    input_config : InputConfig = field(default_factory=lambda: InputConfig(
        use_primary=True,  primary_representation=Representation.MAG_ONLY,
        use_secondaries=False, secondaries_representation=Representation.MAG_ONLY,
        use_interferograms=True, interferograms_representation=Representation.ANGLE_ONLY,
    ))

    train_azimuth : tuple[int, int] = (1000,  9120)
    val_azimuth   : tuple[int, int] = (9120, 12400)
    test_azimuth  : tuple[int, int] = (12400, 16000)

    latent_dim   : int = 16
    epochs       : int = 100


def build_dataset_config(run: RunConfig) -> DatasetCreationConfiguration:
    with open(run.dataset_path / "data" / "dataset.json", "r", encoding="utf-8") as f:
        layout = json.load(f)
    global_crop = CropRegion(*layout["global_crop"])

    split_regions = SplitRegions(
        train = CropRegion(run.train_azimuth[0], run.train_azimuth[1], global_crop.range_start, global_crop.range_end),
        val   = CropRegion(run.val_azimuth[0],   run.val_azimuth[1],   global_crop.range_start, global_crop.range_end),
        test  = CropRegion(run.test_azimuth[0],  run.test_azimuth[1],  global_crop.range_start, global_crop.range_end),
    )
    split_regions.validate_against(global_crop)

    return DatasetCreationConfiguration(
        preprocessing_run_directory = run.dataset_path,
        split_regions               = split_regions,
        patch                       = PatchConfiguration(size=run.patch_size, stride=run.patch_stride, use_reflective_padding=True),
        input_config                = run.input_config,
        pass_drop_train             = PassDropConfig(drop_probs=0.0, min_kept_passes=1, seed=0),
        pass_drop_val               = PassDropConfig(drop_probs=0.0, min_kept_passes=1),
        pass_drop_test              = PassDropConfig(drop_probs=0.0, min_kept_passes=1),
        batch_size                  = run.batch_size,
        num_workers                 = run.num_workers,
        shuffle_train               = True,
        pin_memory                  = True,
    )


def build_ae_config(run: RunConfig) -> AutoencoderConfig:
    return AutoencoderConfig(
        latent_dim = run.latent_dim,

        encoder = EncoderConfig(
            backbone            = BackboneType.conv1d,
            channels            = [32, 64, 128, 256],
            kernel_size         = 5,
            stride              = 2,
            activation          = "gelu",
            normalization       = "batch",
            dropout             = 0.0,
            proj_hidden         = [256],
            proj_dim            = 64,
            use_projection_head = True,
        ),
        
        decoder = DecoderConfig(
            backbone          = BackboneType.conv1d,
            channels          = [256, 128, 64, 32],
            kernel_size       = 5,
            stride            = 2,
            activation        = "gelu",
            normalization     = "batch",
            output_activation = None,
        ),
        
        loss = LossConfig(
            reconstruction_weight   = 1.0,
            variance_weight         = 1.0,
            covariance_weight       = 0.04,
            contrastive_weight      = 0.5,
            reconstruction_loss     = ReconLossName.mse,
            variance_target_std     = 1.0,
            contrastive_temperature = 0.1,
            contrastive_view        = ContrastiveView.augmentation,
        ),
        
        data = DataConfig(
            normalize        = "per_profile_max",
            log_compress     = True,
            log_eps          = 1e-6,
            contrastive_view = ContrastiveView.augmentation,
            max_profiles     = 500000,  
            sampling_seed    = 42,
            augmentation     = AugmentationConfig(
                jitter_std     = 0.02,
                scale_range    = (0.9, 1.1),
                shift_max      = 2,
                mask_prob      = 0.1,
                mask_max_width = 4,
                seed           = 0,
            ),
        ),
        
        trainer = TrainerConfig(
            epochs              = run.epochs,
            learning_rate       = 1e-3,
            weight_decay        = 1e-5,
            optimizer           = "adamw",
            scheduler           = "cosine",
            grad_clip           = 1.0,
            use_amp             = True,
            save_every          = 10,
            val_every           = 1,
            early_stop_patience = 20,
            device              = "cuda" if torch.cuda.is_available() else "cpu",
        ),
        io = IOConfig(
            logdir = "/ste/rnd/User/vice_vi/DLR-TomoSAR/logs",
        ),
    )


def main() -> None:
    run = RunConfig()

    dataset_config = build_dataset_config(run)
    ae_config      = build_ae_config(run)

    pipeline = AutoencoderPipeline(
        ae_config      = ae_config,
        dataset_config = dataset_config,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
