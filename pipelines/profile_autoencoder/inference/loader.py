from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from configuration.inference.profile_autoencoder        import ProfileAeInferenceConfig
from configuration.sar.gaussian_config                  import GaussianConfig
from models.profile_autoencoder                         import get_profile_autoencoder
from pipelines.backbone.dataset.spatial                 import Layout
from pipelines.profile_autoencoder.dataset.datasets     import ProfileDataset
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats
from pipelines.profile_autoencoder.dataset.splitting    import ParameterCropper
from pipelines.shared.config_factory                    import ConfigFactory
from tools.data.io                                      import FileIO, ProfileAutoencoderConfigIO
from tools.data.regions                                 import CropRegion
from tools.monitoring.logger                            import Logger


@dataclass
class ProfileAeRun:
    model                       : object
    ae_name                     : str
    embedding_dim               : int
    x_axis                      : np.ndarray
    normalizer                  : ProfileNormalizer
    dataset                     : ProfileDataset
    loader                      : DataLoader
    split_name                  : str
    n_curves                    : int
    checkpoint_meta             : dict
    preprocessing_run_directory : Path
    split_region                : CropRegion


class ProfileAeRunLoader:
    def __init__(self, run_directory: Path, entry_config, logger: Logger) -> None:
        self.run_directory  = Path(run_directory)
        self.entry_config   = entry_config
        self.logger         = logger
        self.meta_directory = self.run_directory / "meta"

    def _read_run_summary(self) -> dict:
        summary = FileIO.load_json(self.meta_directory / "run_summary.json")

        if str(summary.get("model_name")) != "profile_ae":
            raise ValueError(f"Run '{self.run_directory}' is not a standalone profile-autoencoder run (model_name='{summary.get('model_name')}'). Use 'infer' for backbone and JEPA runs.")

        return summary

    def _dataset_layout(self):
        factory        = ConfigFactory(self.entry_config)
        dataset_config = factory.training_dataset_config()
        gaussian       = GaussianConfig.from_dataset(self.entry_config.paths.dataset_path, self.entry_config.n_gaussians)

        return dataset_config, gaussian

    def _build_dataset(self, config: ProfileAeInferenceConfig, dataset_config, gaussian, normalizer: ProfileNormalizer):
        layout  = Layout(dataset_config.preprocessing_run_directory, logger=self.logger, parameters_path=dataset_config.parameters_path)
        cropper = ParameterCropper(layout, dataset_config.split_regions, logger=self.logger)

        param_arrays = cropper.load_split(config.split)
        x_len        = cropper.profile_length()
        x_axis       = np.linspace(gaussian.x_min, gaussian.x_max, x_len, dtype=np.float32)

        dataset = ProfileDataset(
            param_arrays    = param_arrays,
            x_axis          = x_axis,
            n_gaussians     = gaussian.n_default_gaussians,
            split_name      = config.split,
            pixel_subsample = config.pixel_subsample,
            keep_empty_frac = config.keep_empty_frac,
            seed            = config.seed,
            normalizer      = normalizer,
            augmenter       = None,
            logger          = self.logger,
        )

        return dataset, x_axis, x_len

    def _build_model(self, x_len: int, device: str):
        ae_cfg, ae_name      = ProfileAutoencoderConfigIO.load(self.meta_directory)
        ae_cfg.profile_length = x_len

        model, _ = get_profile_autoencoder(ae_name, ae_cfg)

        return model.to(device), ae_name, ae_cfg

    def _load_checkpoint(self, ckpt_path: Path, device: str):
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for inference: {ckpt_path}")

        ckpt   = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        raw    = ckpt["x_axis"]
        x_axis = raw.cpu().numpy().astype(np.float32) if hasattr(raw, "cpu") else np.asarray(raw, dtype=np.float32)

        meta = {
            "epoch"         : int(ckpt["epoch"]),
            "best_val_loss" : float(ckpt["best_val_loss"]),
            "best_epoch"    : int(ckpt["best_epoch"]),
        }

        return ckpt, x_axis, meta

    def load(self, *, config: ProfileAeInferenceConfig, device: str) -> ProfileAeRun:
        self.logger.section("[Profile AE Inference: Load Run]")
        self.logger.subsection(f"Run Directory : {self.run_directory} \n")

        summary       = self._read_run_summary()
        embedding_dim = int(summary["out_channels"])

        dataset_config, gaussian = self._dataset_layout()
        normalizer               = ProfileNormalizer(ProfileStats.load(self.meta_directory, self.logger))

        dataset, x_axis, x_len = self._build_dataset(config, dataset_config, gaussian, normalizer)

        if x_len != int(summary["x_axis_length"]):
            raise ValueError(f"Profile length mismatch: dataset gives {x_len} bins but the run was trained on {summary['x_axis_length']}. The dataset path does not match the training dataset.")

        model, ae_name, _    = self._build_model(x_len, device)
        ckpt_path            = self.run_directory / config.checkpoint_name
        ckpt, ckpt_x_axis, ckpt_meta = self._load_checkpoint(ckpt_path, device)

        model.load_state_dict(ckpt["params"])
        model.eval()

        if ckpt_x_axis.shape[0] != x_axis.shape[0]:
            raise ValueError(f"Checkpoint x-axis length {ckpt_x_axis.shape[0]} does not match dataset profile length {x_axis.shape[0]}.")

        batch_size = config.batch_size if config.batch_size is not None else len(dataset)
        loader     = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = config.num_workers,
            pin_memory  = True,
            drop_last   = False,
        )

        region = dataset_config.split_regions.regions(config.split)[0]

        self.logger.section(f"[Profile AE Model]  : '{ae_name}'")
        self.logger.kv_table({
            "Checkpoint"     : ckpt_path,
            "Embedding dim"  : embedding_dim,
            "Profile length" : x_len,
            "Curves"         : len(dataset),
            "Split"          : config.split,
        })

        return ProfileAeRun(
            model                       = model,
            ae_name                     = ae_name,
            embedding_dim               = embedding_dim,
            x_axis                      = x_axis,
            normalizer                  = normalizer,
            dataset                     = dataset,
            loader                      = loader,
            split_name                  = config.split,
            n_curves                    = len(dataset),
            checkpoint_meta             = ckpt_meta,
            preprocessing_run_directory = Path(dataset_config.preprocessing_run_directory),
            split_region                = region,
        )
