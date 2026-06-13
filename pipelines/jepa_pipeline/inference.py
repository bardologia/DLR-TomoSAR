from __future__ import annotations

from dataclasses import replace
from pathlib     import Path
from typing      import Dict, List

import numpy as np
import torch
import torch.nn as nn

from models                                    import get_model
from models.profile_autoencoder               import ProfileAutoencoder
from pipelines.inference_pipeline.loader      import ModelWrapper, RunLoader
from pipelines.inference_pipeline.metrics     import Metrics, Result
from pipelines.inference_pipeline.pipeline    import InferencePipeline
from pipelines.inference_pipeline.predictor   import Predictor
from pipelines.autoencoder_pipeline.pipeline   import AutoencoderPipelineSupport
from pipelines.jepa_pipeline.predictor_trainer import JepaModule
from pipelines.shared.io                       import ModelConfigIO
from tools.gaussians                           import GaussianReconstructor

_IMAGE_SIZE_MODELS = {"swin_unet", "transunet", "unetr"}


class JepaInferenceModel(nn.Module):
    def __init__(self, jepa_module: JepaModule) -> None:
        super().__init__()
        self.jepa = jepa_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_hat   = self.jepa.backbone(x)
        curve_n = self.jepa.autoencoder.decode(self.jepa.autoencoder.normalize_embedding(z_hat))
        return self.jepa.autoencoder.denormalize_curve(curve_n)


class JepaRunLoader(RunLoader):
    def _build_model(self, model_name: str, in_channels: int, out_channels: int, image_size: int):
        ae_cfg          = AutoencoderPipelineSupport.load_autoencoder_config(self.meta_directory)
        model_config, _ = ModelConfigIO.load(self.meta_directory)

        overrides = {"in_channels": in_channels, "out_channels": ae_cfg.embedding_dim}
        if model_name in _IMAGE_SIZE_MODELS:
            overrides["image_size"] = image_size
        backbone, _ = get_model(model_name, config=model_config, **overrides)

        autoencoder = ProfileAutoencoder(ae_cfg)
        return JepaModule(backbone, autoencoder)

    def _wrap_model(self, model, device: str, norm_stats, x_axis, amp_max: float) -> ModelWrapper:
        adapter = JepaInferenceModel(model).to(device)
        adapter.eval()
        return ModelWrapper(model=adapter, device=device, params_per_gaussian=3, normalizer=None, x_axis=None, amp_max=None)


class JepaCurvePredictor(Predictor):
    def _forward_pass(self):
        run = self.run
        n_K = run.n_gaussians
        x   = np.asarray(run.x_axis, dtype=np.float32).reshape(1, 1, -1, 1, 1)

        all_indices  : list = []
        all_pred     : list = []
        all_gt       : list = []
        sample_count = 0

        with self.logger.track(transient=True) as prog:
            task = prog.add_task("[section]GPU Forward Pass[/section]", total=len(run.loader))
            for batch in run.loader:
                images, gt_params_b = batch[0], batch[1]

                pred_curves = run.model(images)

                gt_params = gt_params_b[:, :n_K * 3]
                gt_params = run.dataset.normalizer.denormalize_output(gt_params).cpu().numpy().astype(np.float32)

                B         = pred_curves.shape[0]
                H, W      = gt_params.shape[-2:]
                gt_gauss  = gt_params.reshape(B, n_K, 3, H, W)
                gt_curves = GaussianReconstructor.reconstruct_batch(gt_gauss, x)

                all_indices.append(list(range(sample_count, sample_count + B)))
                all_pred.append(pred_curves.astype(np.float32))
                all_gt.append(gt_curves.astype(np.float32))
                sample_count += B
                prog.advance(task)

        return all_indices, all_pred, all_gt

    def _stitch_curves(self, all_indices, all_pred, all_gt):
        n_elev  = self.run.x_axis_length
        pred_st = self._create_stitcher(n_elev, "pred_curves")
        gt_st   = self._create_stitcher(n_elev, "gt_curves")

        for batch_indices, pc, gc in zip(all_indices, all_pred, all_gt):
            for b, idx in enumerate(batch_indices):
                pred_st.add_patch(idx, pc[b].astype(self.cube_dtype))
                gt_st.add_patch(  idx, gc[b].astype(self.cube_dtype))

        return pred_st.finalize_cube(), gt_st.finalize_cube()

    def _finalize(self, pred_cube: np.ndarray, gt_cube: np.ndarray) -> Result:
        pixel_maps     = Metrics.curve_pixel_metrics(pred_cube, gt_cube)
        pixel_mse      = pixel_maps["mse"]
        pixel_mae      = pixel_maps["mae"]
        pixel_r2       = pixel_maps["r2"]
        pixel_cos      = pixel_maps["cos"]
        pixel_peak_idx = pixel_maps["peak"].astype(np.int32)

        if self.save_cubes:
            np.save(self.cube_dir / "pred_curves.npy", pred_cube)
            np.save(self.cube_dir / "gt_curves.npy",   gt_cube)
            np.save(self.cube_dir / "pixel_mse.npy",   pixel_mse)
            np.save(self.cube_dir / "pixel_mae.npy",   pixel_mae)
            np.save(self.cube_dir / "pixel_r2.npy",    pixel_r2)
            np.save(self.cube_dir / "pixel_cos.npy",   pixel_cos)
            np.save(self.cube_dir / "pixel_peak.npy",  pixel_peak_idx)
            for tmp in self.cube_dir.glob("_tmp_*.npy"):
                try:
                    tmp.unlink()
                except OSError:
                    pass

        self.logger.section("[Inference: Results]")
        self.logger.kv_table({
            "Curves cube    (denorm)": pred_cube.shape,
            "Mean pixel MSE (denorm)": f"{pixel_mse.mean():.4g}  (pred vs gt)",
            "Mean pixel R²  (denorm)": f"{pixel_r2.mean():.4g}  (pred vs gt)",
        })

        return Result(
            pred_curves        = pred_cube,
            gt_curves          = gt_cube,
            pixel_mse          = pixel_mse,
            pixel_mae          = pixel_mae,
            pixel_r2           = pixel_r2,
            pixel_cosine       = pixel_cos,
            pixel_peak_err_idx = pixel_peak_idx,
            cube_directory     = self.cube_dir,
            azimuth_offset     = self.run.split_region.azimuth_start,
            range_offset       = self.run.split_region.range_start,
        )

    def run_inference(self) -> Result:
        backend = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.section("[Inference: Predict]")
        self.logger.kv_table({
            "Backend":    f"PyTorch / {backend}",
            "Cube dir":   self.cube_dir,
            "Window":     self.window_kind,
            "Cube dtype": self.cube_dtype,
        })

        all_indices, all_pred, all_gt = self._forward_pass()
        pred_cube, gt_cube            = self._stitch_curves(all_indices, all_pred, all_gt)
        return self._finalize(pred_cube, gt_cube)


class JepaInferencePipeline(InferencePipeline):
    def __init__(self, config, run_directory: Path) -> None:
        super().__init__(replace(config, run_directory=Path(run_directory), output_subdir=None))

    def _load_run(self, cfg, logger):
        loader = JepaRunLoader(cfg.run_directory, logger=logger)
        return loader.load(
            split           = cfg.split,
            batch_size      = cfg.batch_size,
            num_workers     = cfg.num_workers,
            device          = cfg.device,
            checkpoint_name = cfg.checkpoint_name,
        )

    def _predict(self, cfg, meta, run, logger):
        predictor = JepaCurvePredictor(
            run         = run,
            logger      = logger,
            window_kind = cfg.stitch_window,
            cube_dtype  = cfg.cube_dtype,
            save_cubes  = cfg.save_cubes,
            meta        = meta,
            cpu_workers = cfg.cpu_workers,
        )
        return predictor.run_inference()

    def _evaluate_metrics(self, result, x_axis_np: np.ndarray, run, meta, indices: dict) -> dict:
        global_metrics = Metrics(result, x_axis_np, run.n_gaussians).compute(
            elev_indices  = indices["all_elev_idx"],
            range_indices = indices["all_range_idx"],
            az_indices    = indices["all_az_idx"],
            param_space   = False,
        )

        global_metrics["split"]        = run.split_name
        global_metrics["split_region"] = list(run.split_region.as_tuple())

        if run.track_baselines is not None:
            global_metrics["tracks"] = run.track_baselines.to_payload()

        if run.track_profiles is not None:
            global_metrics["track_positions"] = run.track_profiles.position_summary()

        Metrics.write_json(global_metrics, meta.metrics_path)
        return global_metrics

    def _compose_figures(self, composer, result, run, global_metrics: dict, x_axis_np: np.ndarray, indices: dict) -> Dict[str, List[Path]]:
        return composer.compose(
            result         = result,
            run            = run,
            global_metrics = global_metrics,
            x_axis_np      = x_axis_np,
            indices        = indices,
            param_space    = False,
        )
