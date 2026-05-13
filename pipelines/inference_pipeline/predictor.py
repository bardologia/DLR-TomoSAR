from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Dict

import numpy as np
import torch
import torch.nn.functional as F

from pipelines.inference_pipeline.loader    import LoadedRun
from pipelines.inference_pipeline.stitching import CubeStitcher, make_patch_window
from tools.logger                           import Logger


PARAMS_PER_GAUSSIAN = 3


def reconstruct_curves(params: torch.Tensor, x_axis: torch.Tensor, n_gaussians: int) -> torch.Tensor:
    B, _, H, W = params.shape
    x   = x_axis.to(params.device, dtype=params.dtype).reshape(1, -1, 1, 1)
    out = torch.zeros((B, x.shape[1], H, W), device=params.device, dtype=params.dtype)
    for k in range(n_gaussians):
        a   = params[:, 3 * k     : 3 * k + 1]
        mu  = params[:, 3 * k + 1 : 3 * k + 2]
        sig = params[:, 3 * k + 2 : 3 * k + 3]
        out = out + a * torch.exp(-((x - mu) ** 2) / (2.0 * sig * sig + 1e-8))
    return out


def prepare_gt_params_for_reconstruction(
    gt_params  : torch.Tensor,
    n_gaussians: int,
    normalizer,  # Optional[Normalizer] — None if no output normalisation
) -> torch.Tensor:

    if normalizer is not None and normalizer.stats.output_stats is not None:
        gt_params = normalizer.denormalize_output(gt_params)

    return gt_params


@dataclass
class PredictionResult:
    pred_curves        : np.ndarray  
    gt_curves          : np.ndarray   
    raw_curves         : np.ndarray   
    params_pred        : np.ndarray
    params_gt          : np.ndarray

    pixel_mse          : np.ndarray
    pixel_mae          : np.ndarray
    pixel_r2           : np.ndarray
    pixel_cosine       : np.ndarray
    pixel_peak_err_idx : np.ndarray

    pixel_mse_raw          : np.ndarray
    pixel_mae_raw          : np.ndarray
    pixel_r2_raw           : np.ndarray
    pixel_cosine_raw       : np.ndarray
    pixel_peak_err_idx_raw : np.ndarray

    cube_directory     : Path
    azimuth_offset     : int
    range_offset       : int


class Predictor:
    def __init__(
        self,
        run         : LoadedRun,
        logger      : Logger,
        *,
        window_kind : str,
        cube_dtype  : str,
        save_cubes  : bool,
        output_dir  : Path,
    ) -> None:
        self.run         = run
        self.logger      = logger
        self.window_kind = window_kind
        self.cube_dtype  = cube_dtype
        self.save_cubes  = save_cubes
        self.output_dir  = Path(output_dir)
        self.cube_dir    = self.output_dir / "cubes"
        self.cube_dir.mkdir(parents=True, exist_ok=True)

    def _new_stitcher(self, n_channels: int, name: str) -> CubeStitcher:
        memmap_path = str(self.cube_dir / f"_tmp_{name}.npy") if self.save_cubes else None
        return CubeStitcher(
            grid        = self.run.grid,
            n_channels  = n_channels,
            window_kind = self.window_kind,
            dtype       = self.cube_dtype,
            memmap_path = memmap_path,
        )

    @staticmethod
    def _peak_index_diff(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return (pred.argmax(dim=1) - gt.argmax(dim=1)).abs().to(torch.int32)

    @staticmethod
    def _per_pixel_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        diff   = pred - gt
        mse    = (diff * diff).mean(dim=1)
        mae    = diff.abs().mean(dim=1)
        ss_res = (diff * diff).sum(dim=1)
        gmean  = gt.mean(dim=1, keepdim=True)
        ss_tot = ((gt - gmean) * (gt - gmean)).sum(dim=1)
        r2     = 1.0 - ss_res / (ss_tot + 1e-8)
        cos    = F.cosine_similarity(pred, gt, dim=1)
        return {"mse": mse, "mae": mae, "r2": r2, "cos": cos}

    @torch.no_grad()
    def run_inference(self) -> PredictionResult:
        run       = self.run
        device    = next(run.model.parameters()).device
        n_elev    = run.x_axis_length
        n_K       = run.n_gaussians
        out_ch    = run.out_channels
        params_ch = out_ch

        self.logger.section("[Inference: Predict]")
        self.logger.subsection(f"Device     : {device}")
        self.logger.subsection(f"Cube dir   : {self.cube_dir}")
        self.logger.subsection(f"Window     : {self.window_kind}")
        self.logger.subsection(f"Cube dtype : {self.cube_dtype}\n")

        H, W = run.grid.spatial_size
        pred_curve_stitcher = self._new_stitcher(n_elev,                    "pred_curves")
        raw_curve_stitcher  = self._new_stitcher(n_elev,                    "raw_curves")
        gt_curve_stitcher   = self._new_stitcher(n_elev,                    "gt_curves")
        param_pred_stitcher = self._new_stitcher(params_ch,                 "params_pred")
        gt_param_stitcher   = self._new_stitcher(n_K * PARAMS_PER_GAUSSIAN, "params_gt")

        pixel_mse      = np.zeros((H, W), dtype=np.float32)
        pixel_mae      = np.zeros((H, W), dtype=np.float32)
        pixel_r2       = np.zeros((H, W), dtype=np.float32)
        pixel_cos      = np.zeros((H, W), dtype=np.float32)
        pixel_peak     = np.zeros((H, W), dtype=np.float32)

        pixel_mse_raw  = np.zeros((H, W), dtype=np.float32)
        pixel_mae_raw  = np.zeros((H, W), dtype=np.float32)
        pixel_r2_raw   = np.zeros((H, W), dtype=np.float32)
        pixel_cos_raw  = np.zeros((H, W), dtype=np.float32)
        pixel_peak_raw = np.zeros((H, W), dtype=np.float32)

        pixel_w    = np.zeros((H, W), dtype=np.float32)

        win2d  = make_patch_window(run.grid.patch_size, kind=self.window_kind)
        ph, pw = run.grid.patch_size

        run.model.eval()
        sample_count = 0
        with self.logger.track(transient=True) as _prog:
            _task = _prog.add_task("[section]Inference[/section]", total=len(run.loader))
            for batch_idx, batch in enumerate(run.loader):
                images, raw_curves_b, gt_params_b = batch[0], batch[1], batch[2]

                images       = images.to(device, non_blocking=True).float()
                raw_curves_b = raw_curves_b.to(device, non_blocking=True).float()
                gt_params_b  = gt_params_b.to(device, non_blocking=True).float()

                pred_params = run.model(images)
                pred_gauss  = pred_params[:, : (out_ch // PARAMS_PER_GAUSSIAN) * PARAMS_PER_GAUSSIAN]
                pred_curves = reconstruct_curves(pred_gauss, run.x_axis, n_K)

                gt_params_ready = prepare_gt_params_for_reconstruction(
                    gt_params_b[:, : n_K * PARAMS_PER_GAUSSIAN],
                    n_K,
                    run.dataset.norm_stats,
                )
                gt_curves_b = reconstruct_curves(gt_params_ready, run.x_axis, n_K)

                mets     = self._per_pixel_metrics(pred_curves, gt_curves_b)
                mets_raw = self._per_pixel_metrics(pred_curves, raw_curves_b)
                peak     = self._peak_index_diff(pred_curves, gt_curves_b)
                peak_raw = self._peak_index_diff(pred_curves, raw_curves_b)

                pred_curves_np = pred_curves.detach().cpu().numpy().astype(self.cube_dtype, copy=False)
                raw_curves_np  = raw_curves_b.detach().cpu().numpy().astype(self.cube_dtype, copy=False)
                gt_curves_np   = gt_curves_b.detach().cpu().numpy().astype(self.cube_dtype, copy=False)
                params_pred_np = pred_params.detach().cpu().numpy().astype(self.cube_dtype, copy=False)
                params_gt_np   = gt_params_ready.detach().cpu().numpy().astype(self.cube_dtype, copy=False)
                mse_np         = mets["mse"].detach().cpu().numpy()
                mae_np         = mets["mae"].detach().cpu().numpy()
                r2_np          = mets["r2"].detach().cpu().numpy()
                cos_np         = mets["cos"].detach().cpu().numpy()
                peak_np        = peak.detach().cpu().numpy().astype(np.float32)
                mse_raw_np     = mets_raw["mse"].detach().cpu().numpy()
                mae_raw_np     = mets_raw["mae"].detach().cpu().numpy()
                r2_raw_np      = mets_raw["r2"].detach().cpu().numpy()
                cos_raw_np     = mets_raw["cos"].detach().cpu().numpy()
                peak_raw_np    = peak_raw.detach().cpu().numpy().astype(np.float32)

                B        = images.shape[0]
                base_idx = sample_count
                for b in range(B):
                    idx = base_idx + b
                    pred_curve_stitcher.add(idx, pred_curves_np[b])
                    raw_curve_stitcher.add(idx, raw_curves_np[b])
                    gt_curve_stitcher.add(idx, gt_curves_np[b])
                    param_pred_stitcher.add(idx, params_pred_np[b])
                    gt_param_stitcher.add(idx, params_gt_np[b])

                    iv, ih   = divmod(idx, run.grid.n_h)
                    v0       = iv * run.grid.stride - run.grid.pad_top
                    h0       = ih * run.grid.stride - run.grid.pad_left
                    v0c, h0c = max(0, v0), max(0, h0)
                    v1c, h1c = min(H, v0 + ph), min(W, h0 + pw)
                    pv0, ph0 = v0c - v0, h0c - h0
                    pv1, ph1 = pv0 + (v1c - v0c), ph0 + (h1c - h0c)
                    w_local  = win2d[pv0:pv1, ph0:ph1]

                    pixel_mse [v0c:v1c, h0c:h1c] += w_local * mse_np [b, pv0:pv1, ph0:ph1]
                    pixel_mae [v0c:v1c, h0c:h1c] += w_local * mae_np [b, pv0:pv1, ph0:ph1]
                    pixel_r2  [v0c:v1c, h0c:h1c] += w_local * r2_np  [b, pv0:pv1, ph0:ph1]
                    pixel_cos [v0c:v1c, h0c:h1c] += w_local * cos_np [b, pv0:pv1, ph0:ph1]
                    pixel_peak[v0c:v1c, h0c:h1c] += w_local * peak_np[b, pv0:pv1, ph0:ph1]

                    pixel_mse_raw [v0c:v1c, h0c:h1c] += w_local * mse_raw_np [b, pv0:pv1, ph0:ph1]
                    pixel_mae_raw [v0c:v1c, h0c:h1c] += w_local * mae_raw_np [b, pv0:pv1, ph0:ph1]
                    pixel_r2_raw  [v0c:v1c, h0c:h1c] += w_local * r2_raw_np  [b, pv0:pv1, ph0:ph1]
                    pixel_cos_raw [v0c:v1c, h0c:h1c] += w_local * cos_raw_np [b, pv0:pv1, ph0:ph1]
                    pixel_peak_raw[v0c:v1c, h0c:h1c] += w_local * peak_raw_np[b, pv0:pv1, ph0:ph1]

                    pixel_w[v0c:v1c, h0c:h1c] += w_local

                sample_count += B
                _prog.advance(_task)

        pred_curves_cube = pred_curve_stitcher.finalize()
        raw_curves_cube  = raw_curve_stitcher.finalize()
        gt_curves_cube   = gt_curve_stitcher.finalize()
        params_pred_cube = param_pred_stitcher.finalize()
        params_gt_cube   = gt_param_stitcher.finalize()

        w_safe             = np.where(pixel_w > 0, pixel_w, 1.0)
        pixel_mse          = (pixel_mse     / w_safe).astype(np.float32)
        pixel_mae          = (pixel_mae     / w_safe).astype(np.float32)
        pixel_r2           = (pixel_r2      / w_safe).astype(np.float32)
        pixel_cos          = (pixel_cos     / w_safe).astype(np.float32)
        pixel_peak_idx     = np.rint(pixel_peak     / w_safe).astype(np.int32)
        pixel_mse_raw      = (pixel_mse_raw / w_safe).astype(np.float32)
        pixel_mae_raw      = (pixel_mae_raw / w_safe).astype(np.float32)
        pixel_r2_raw       = (pixel_r2_raw  / w_safe).astype(np.float32)
        pixel_cos_raw      = (pixel_cos_raw / w_safe).astype(np.float32)
        pixel_peak_idx_raw = np.rint(pixel_peak_raw / w_safe).astype(np.int32)

        if self.save_cubes:
            np.save(self.cube_dir / "pred_curves.npy",    pred_curves_cube)
            np.save(self.cube_dir / "raw_curves.npy",     raw_curves_cube)
            np.save(self.cube_dir / "gt_curves.npy",      gt_curves_cube)
            np.save(self.cube_dir / "params_pred.npy",    params_pred_cube)
            np.save(self.cube_dir / "params_gt.npy",      params_gt_cube)
            np.save(self.cube_dir / "pixel_mse.npy",      pixel_mse)
            np.save(self.cube_dir / "pixel_mae.npy",      pixel_mae)
            np.save(self.cube_dir / "pixel_r2.npy",       pixel_r2)
            np.save(self.cube_dir / "pixel_cos.npy",      pixel_cos)
            np.save(self.cube_dir / "pixel_peak.npy",     pixel_peak_idx)
            np.save(self.cube_dir / "pixel_mse_raw.npy",  pixel_mse_raw)
            np.save(self.cube_dir / "pixel_mae_raw.npy",  pixel_mae_raw)
            np.save(self.cube_dir / "pixel_r2_raw.npy",   pixel_r2_raw)
            np.save(self.cube_dir / "pixel_cos_raw.npy",  pixel_cos_raw)
            np.save(self.cube_dir / "pixel_peak_raw.npy", pixel_peak_idx_raw)
            for tmp in self.cube_dir.glob("_tmp_*.npy"):
                try:
                    tmp.unlink()
                except OSError:
                    pass

        self.logger.subsection(f"Curves cube        : {pred_curves_cube.shape}")
        self.logger.subsection(f"Params cube        : {params_pred_cube.shape}")
        self.logger.subsection(f"GT params cube     : {params_gt_cube.shape}")
        self.logger.subsection(f"Mean pixel MSE     : {pixel_mse.mean():.4g}  (pred vs gt)")
        self.logger.subsection(f"Mean pixel R²      : {pixel_r2.mean():.4g}  (pred vs gt)")
        self.logger.subsection(f"Mean pixel MSE raw : {pixel_mse_raw.mean():.4g}  (pred vs raw)")
        self.logger.subsection(f"Mean pixel R² raw  : {pixel_r2_raw.mean():.4g}  (pred vs raw)\n")

        return PredictionResult(
            pred_curves            = pred_curves_cube,
            gt_curves              = gt_curves_cube,
            raw_curves             = raw_curves_cube,
            params_pred            = params_pred_cube,
            params_gt              = params_gt_cube,
            pixel_mse              = pixel_mse,
            pixel_mae              = pixel_mae,
            pixel_r2               = pixel_r2,
            pixel_cosine           = pixel_cos,
            pixel_peak_err_idx     = pixel_peak_idx,
            pixel_mse_raw          = pixel_mse_raw,
            pixel_mae_raw          = pixel_mae_raw,
            pixel_r2_raw           = pixel_r2_raw,
            pixel_cosine_raw       = pixel_cos_raw,
            pixel_peak_err_idx_raw = pixel_peak_idx_raw,
            cube_directory         = self.cube_dir,
            azimuth_offset         = run.split_region.azimuth_start,
            range_offset           = run.split_region.range_start,
        )
