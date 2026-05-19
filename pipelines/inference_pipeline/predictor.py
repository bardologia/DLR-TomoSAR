from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Dict

import numpy as np

from pipelines.inference_pipeline.loader    import LoadedRun
from pipelines.inference_pipeline.stitching import CubeStitcher, make_patch_window
from tools.logger                           import Logger


@dataclass
class Result:
    pred_curves        : np.ndarray  
    gt_curves          : np.ndarray   
    params_pred        : np.ndarray
    params_gt          : np.ndarray

    pixel_mse          : np.ndarray
    pixel_mae          : np.ndarray
    pixel_r2           : np.ndarray
    pixel_cosine       : np.ndarray
    pixel_peak_err_idx : np.ndarray

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
    def _reconstruct_curves(params: np.ndarray, x_axis: np.ndarray, n_gaussians: int) -> np.ndarray:
        B, _, H, W = params.shape
        p   = np.asarray(params, dtype=np.float32)
        x   = np.asarray(x_axis, dtype=np.float32).reshape(1, -1, 1, 1)
        out = np.zeros((B, x.shape[1], H, W), dtype=np.float32)

        for k in range(n_gaussians):
            a   = p[:, 3 * k     : 3 * k + 1]
            mu  = p[:, 3 * k + 1 : 3 * k + 2]
            sig = p[:, 3 * k + 2 : 3 * k + 3]
            out = out + a * np.exp(-((x - mu) ** 2) / (2.0 * sig * sig + 1e-8))

        return out

    @staticmethod
    def _per_pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, np.ndarray]:
        diff    = pred - gt
        mse     = (diff * diff).mean(axis=1)
        mae     = np.abs(diff).mean(axis=1)
        ss_res  = (diff * diff).sum(axis=1)
        gmean   = gt.mean(axis=1, keepdims=True)
        ss_tot  = ((gt - gmean) * (gt - gmean)).sum(axis=1)
        r2      = 1.0 - ss_res / (ss_tot + 1e-8)
        dot     = (pred * gt).sum(axis=1)
        norm_p  = np.sqrt((pred * pred).sum(axis=1)) + 1e-8
        norm_g  = np.sqrt((gt   * gt  ).sum(axis=1)) + 1e-8
        cos     = dot / (norm_p * norm_g)

        return {"mse": mse, "mae": mae, "r2": r2, "cos": cos}

    def _run_batch_loop(
        self,
        pred_curve_stitcher : CubeStitcher,
        gt_curve_stitcher   : CubeStitcher,
        param_pred_stitcher : CubeStitcher,
        gt_param_stitcher   : CubeStitcher,
        pixel_mse  : np.ndarray,
        pixel_mae  : np.ndarray,
        pixel_r2   : np.ndarray,
        pixel_cos  : np.ndarray,
        pixel_peak : np.ndarray,
        pixel_w    : np.ndarray,
        win2d      : np.ndarray,
    ) -> None:
        run    = self.run
        n_K    = run.n_gaussians
        out_ch = run.out_channels
        H, W   = run.grid.spatial_size
        ph, pw = run.grid.patch_size

        sample_count = 0
        with self.logger.track(transient=True) as _prog:
            _task = _prog.add_task("[section]Inference[/section]", total=len(run.loader))
            for batch_idx, batch in enumerate(run.loader):
                images, gt_params_b = batch[0], batch[1]

                images      = np.asarray(images,      dtype=np.float32)
                gt_params_b = np.asarray(gt_params_b, dtype=np.float32)

                pred_params     = run.model(images)
                pred_gauss      = pred_params[:, : (out_ch // 3) * 3]
                pred_curves     = self._reconstruct_curves(pred_gauss, run.x_axis, n_K)

                gt_params_ready = gt_params_b[:, : n_K * 3]
                gt_params_ready = run.dataset.norm_stats.denormalize_output(gt_params_ready)
                gt_curves_b     = self._reconstruct_curves(gt_params_ready, run.x_axis, n_K)

                mets    = self._per_pixel_metrics(pred_curves, gt_curves_b)
                peak_np = np.abs(pred_curves.argmax(axis=1) - gt_curves_b.argmax(axis=1)).astype(np.float32)

                B        = images.shape[0]
                base_idx = sample_count
                for b in range(B):
                    idx = base_idx + b

                    pred_curve_stitcher.add(idx, pred_curves.astype(self.cube_dtype)[b])
                    gt_curve_stitcher.add(idx, gt_curves_b.astype(self.cube_dtype)[b])
                    param_pred_stitcher.add(idx, pred_params.astype(self.cube_dtype)[b])
                    gt_param_stitcher.add(idx, gt_params_ready.astype(self.cube_dtype)[b])

                    iv, ih   = divmod(idx, run.grid.n_h)
                    v0       = iv * run.grid.stride - run.grid.pad_top
                    h0       = ih * run.grid.stride - run.grid.pad_left
                    v0c, h0c = max(0, v0), max(0, h0)
                    v1c, h1c = min(H, v0 + ph), min(W, h0 + pw)
                    pv0, ph0 = v0c - v0, h0c - h0
                    pv1, ph1 = pv0 + (v1c - v0c), ph0 + (h1c - h0c)
                    w_local  = win2d[pv0:pv1, ph0:ph1]

                    pixel_mse [v0c:v1c, h0c:h1c] += w_local * mets["mse"][b, pv0:pv1, ph0:ph1]
                    pixel_mae [v0c:v1c, h0c:h1c] += w_local * mets["mae"][b, pv0:pv1, ph0:ph1]
                    pixel_r2  [v0c:v1c, h0c:h1c] += w_local * mets["r2"] [b, pv0:pv1, ph0:ph1]
                    pixel_cos [v0c:v1c, h0c:h1c] += w_local * mets["cos"][b, pv0:pv1, ph0:ph1]
                    pixel_peak[v0c:v1c, h0c:h1c] += w_local * peak_np    [b, pv0:pv1, ph0:ph1]
                    pixel_w   [v0c:v1c, h0c:h1c] += w_local

                sample_count += B
                _prog.advance(_task)

    def _finalize(
        self,
        pred_curve_stitcher : CubeStitcher,
        gt_curve_stitcher   : CubeStitcher,
        param_pred_stitcher : CubeStitcher,
        gt_param_stitcher   : CubeStitcher,
        pixel_mse  : np.ndarray,
        pixel_mae  : np.ndarray,
        pixel_r2   : np.ndarray,
        pixel_cos  : np.ndarray,
        pixel_peak : np.ndarray,
        pixel_w    : np.ndarray,
    ) -> Result:
        pred_curves_cube = pred_curve_stitcher.finalize()
        gt_curves_cube   = gt_curve_stitcher.finalize()
        params_pred_cube = param_pred_stitcher.finalize()
        params_gt_cube   = gt_param_stitcher.finalize()

        w_safe         = np.where(pixel_w > 0, pixel_w, 1.0)
        pixel_mse      = (pixel_mse         / w_safe).astype(np.float32)
        pixel_mae      = (pixel_mae         / w_safe).astype(np.float32)
        pixel_r2       = (pixel_r2          / w_safe).astype(np.float32)
        pixel_cos      = (pixel_cos         / w_safe).astype(np.float32)
        pixel_peak_idx = np.rint(pixel_peak / w_safe).astype(np.int32)

        if self.save_cubes:
            np.save(self.cube_dir / "pred_curves.npy", pred_curves_cube)
            np.save(self.cube_dir / "gt_curves.npy",   gt_curves_cube)
            np.save(self.cube_dir / "params_pred.npy", params_pred_cube)
            np.save(self.cube_dir / "params_gt.npy",   params_gt_cube)
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

        self.logger.subsection(f"Curves cube    : {pred_curves_cube.shape}")
        self.logger.subsection(f"Params cube    : {params_pred_cube.shape}")
        self.logger.subsection(f"GT params cube : {params_gt_cube.shape}")
        self.logger.subsection(f"Mean pixel MSE : {pixel_mse.mean():.4g}  (pred vs gt)")
        self.logger.subsection(f"Mean pixel R²  : {pixel_r2.mean():.4g}  (pred vs gt)\n")

        return Result(
            pred_curves        = pred_curves_cube,
            gt_curves          = gt_curves_cube,
            params_pred        = params_pred_cube,
            params_gt          = params_gt_cube,
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
        import torch
        run    = self.run
        n_elev = run.x_axis_length
        n_K    = run.n_gaussians
        out_ch = run.out_channels

        backend = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.section("[Inference: Predict]")
        self.logger.subsection(f"Backend    : PyTorch / {backend}")
        self.logger.subsection(f"Cube dir   : {self.cube_dir}")
        self.logger.subsection(f"Window     : {self.window_kind}")
        self.logger.subsection(f"Cube dtype : {self.cube_dtype}\n")

        H, W = run.grid.spatial_size
        pred_curve_stitcher = self._new_stitcher(n_elev,  "pred_curves")
        gt_curve_stitcher   = self._new_stitcher(n_elev,  "gt_curves")
        param_pred_stitcher = self._new_stitcher(out_ch,  "params_pred")
        gt_param_stitcher   = self._new_stitcher(n_K * 3, "params_gt")

        pixel_mse  = np.zeros((H, W), dtype=np.float32)
        pixel_mae  = np.zeros((H, W), dtype=np.float32)
        pixel_r2   = np.zeros((H, W), dtype=np.float32)
        pixel_cos  = np.zeros((H, W), dtype=np.float32)
        pixel_peak = np.zeros((H, W), dtype=np.float32)
        pixel_w    = np.zeros((H, W), dtype=np.float32)

        win2d = make_patch_window(run.grid.patch_size, kind=self.window_kind)

        self._run_batch_loop(
            pred_curve_stitcher, gt_curve_stitcher,
            param_pred_stitcher, gt_param_stitcher,
            pixel_mse, pixel_mae, pixel_r2, pixel_cos, pixel_peak, pixel_w,
            win2d,
        )

        return self._finalize(
            pred_curve_stitcher, gt_curve_stitcher,
            param_pred_stitcher, gt_param_stitcher,
            pixel_mse, pixel_mae, pixel_r2, pixel_cos, pixel_peak, pixel_w,
        )
