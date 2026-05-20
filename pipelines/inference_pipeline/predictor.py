from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses        import dataclass
from pathlib            import Path
from typing             import List, Tuple

import numpy as np

from pipelines.inference_pipeline.loader    import LoadedRun
from pipelines.inference_pipeline.metadata  import InferenceMetadata
from pipelines.inference_pipeline.stitching import CubeStitcher
from tools.logger                           import Logger


@dataclass
class Result:
    pred_curves        : np.ndarray   # (denorm) reconstructed Gaussian curves from predicted params
    gt_curves          : np.ndarray   # (denorm) reconstructed Gaussian curves from GT params
    params_pred        : np.ndarray   # (denorm) predicted Gaussian parameters
    params_gt          : np.ndarray   # (denorm) ground-truth Gaussian parameters

    pixel_mse          : np.ndarray
    pixel_mae          : np.ndarray
    pixel_r2           : np.ndarray
    pixel_cosine       : np.ndarray
    pixel_peak_err_idx : np.ndarray

    cube_directory     : Path
    azimuth_offset     : int
    range_offset       : int


def _cpu_worker(args: tuple) -> tuple:
    pred_params_chunk, gt_params_chunk, x_axis, n_gaussians, out_ch = args

    x    = x_axis.reshape(1, -1, 1, 1).astype(np.float32)
    B, _, H, W = pred_params_chunk.shape
    n_elev     = x.shape[1]

    def reconstruct(params: np.ndarray, n_K: int) -> np.ndarray:
        out = np.zeros((B, n_elev, H, W), dtype=np.float32)
        for k in range(n_K):
            a   = np.maximum(params[:, 3 * k     : 3 * k + 1], 0.0)
            mu  =            params[:, 3 * k + 1 : 3 * k + 2]
            sig =            params[:, 3 * k + 2 : 3 * k + 3]
            out = out + a * np.exp(-((x - mu) ** 2) / (2.0 * sig * sig + 1e-8))
        return out

    pred_gauss  = pred_params_chunk[:, :(out_ch // 3) * 3]
    pred_curves = reconstruct(pred_gauss,         n_gaussians)
    gt_curves   = reconstruct(gt_params_chunk,    n_gaussians)

    diff   = pred_curves - gt_curves
    mse    = (diff * diff).mean(axis=1)
    mae    = np.abs(diff).mean(axis=1)
    ss_res = (diff * diff).sum(axis=1)
    gmean  = gt_curves.mean(axis=1, keepdims=True)
    ss_tot = ((gt_curves - gmean) ** 2).sum(axis=1)
    r2     = 1.0 - ss_res / (ss_tot + 1e-8)
    dot    = (pred_curves * gt_curves).sum(axis=1)
    norm_p = np.sqrt((pred_curves * pred_curves).sum(axis=1)) + 1e-8
    norm_g = np.sqrt((gt_curves   * gt_curves  ).sum(axis=1)) + 1e-8
    cos    = dot / (norm_p * norm_g)
    peak   = np.abs(pred_curves.argmax(axis=1) - gt_curves.argmax(axis=1)).astype(np.float32)

    return (
        pred_curves.astype(np.float32),
        gt_curves.astype(np.float32),
        pred_params_chunk.astype(np.float32),
        gt_params_chunk.astype(np.float32),
        {"mse": mse, "mae": mae, "r2": r2, "cos": cos},
        peak,
    )


class Predictor:
    def __init__(
        self,
        run         : LoadedRun,
        logger      : Logger,
        *,
        window_kind : str,
        cube_dtype  : str,
        save_cubes  : bool,
        meta        : InferenceMetadata,
        cpu_workers : int | None = None,
    ) -> None:
        
        self.run         = run
        self.logger      = logger
        self.window_kind = window_kind
        self.cube_dtype  = cube_dtype
        self.save_cubes  = save_cubes
        self.cube_dir    = meta.cube_dir()
        self.cpu_workers = cpu_workers if cpu_workers is not None else min(8, os.cpu_count() or 1)

    def _create_stitcher(self, n_channels: int, name: str) -> CubeStitcher:
        memmap_path = str(self.cube_dir / f"_tmp_{name}.npy") if self.save_cubes else None
        return CubeStitcher(
            grid        = self.run.grid,
            n_channels  = n_channels,
            window_kind = self.window_kind,
            dtype       = self.cube_dtype,
            memmap_path = memmap_path,
        )

    def _forward_pass(self) -> Tuple[List[List[int]], List[np.ndarray], List[np.ndarray]]:
        run    = self.run
        n_K    = run.n_gaussians

        all_indices     : List[List[int]]   = []
        all_pred_params : List[np.ndarray]  = []
        all_gt_params   : List[np.ndarray]  = []
        sample_count = 0

        with self.logger.track(transient=True) as prog:
            task = prog.add_task("[section]GPU Forward Pass[/section]", total=len(run.loader))
            for batch in run.loader:
                images, gt_params_b = batch[0], batch[1]
                images      = np.asarray(images,      dtype=np.float32)  # (norm)   model input
                gt_params_b = np.asarray(gt_params_b, dtype=np.float32)  # (norm)   from dataset

                pred_params     = run.model(images)  # (denorm) wrapper applies denorm + constraints

                gt_params_ready = gt_params_b[:, :n_K * 3]                              # (norm)
                gt_params_ready = run.dataset.norm_stats.denormalize_output(gt_params_ready)  # (denorm)

                B = images.shape[0]
                all_indices.append(list(range(sample_count, sample_count + B)))
                all_pred_params.append(pred_params)                                      # (denorm)
                all_gt_params.append(np.asarray(gt_params_ready, dtype=np.float32))      # (denorm)
                sample_count += B
                prog.advance(task)

        return all_indices, all_pred_params, all_gt_params

    def _compute_metrics(self, all_pred_params : List[np.ndarray], all_gt_params : List[np.ndarray]) -> List[tuple]:
        run    = self.run
        n_K    = run.n_gaussians
        out_ch = run.out_channels

        tasks = [(pred, gt, run.x_axis, n_K, out_ch) for pred, gt in zip(all_pred_params, all_gt_params)]

        results: List[tuple | None] = [None] * len(tasks)

        with self.logger.track(transient=True) as prog:
            task_id = prog.add_task("[section]CPU Metrics[/section]", total=len(tasks))
            with ProcessPoolExecutor(max_workers=self.cpu_workers) as pool:
                futures = {pool.submit(_cpu_worker, t): i for i, t in enumerate(tasks)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    results[idx] = fut.result()
                    prog.advance(task_id)

        return results  

    def _stitch_results(self, all_indices : List[List[int]], cpu_results : List[tuple]) -> Result:
        run    = self.run
        n_K    = run.n_gaussians
        out_ch = run.out_channels
        n_elev = run.x_axis_length
        H, W   = run.grid.spatial_size
        ph, pw = run.grid.patch_size

        pred_curve_stitcher = self._create_stitcher(n_elev,  "pred_curves")
        gt_curve_stitcher   = self._create_stitcher(n_elev,  "gt_curves")
        param_pred_stitcher = self._create_stitcher(out_ch,  "params_pred")
        gt_param_stitcher   = self._create_stitcher(n_K * 3, "params_gt")

        pixel_mse  = np.zeros((H, W), dtype=np.float32)
        pixel_mae  = np.zeros((H, W), dtype=np.float32)
        pixel_r2   = np.zeros((H, W), dtype=np.float32)
        pixel_cos  = np.zeros((H, W), dtype=np.float32)
        pixel_peak = np.zeros((H, W), dtype=np.float32)
        pixel_w    = np.zeros((H, W), dtype=np.float32)

        win2d = CubeStitcher.make_patch_window(run.grid.patch_size, kind=self.window_kind)

        for batch_indices, (pred_curves, gt_curves, pred_params, gt_params, mets, peak_np) in zip(all_indices, cpu_results):
            for b, idx in enumerate(batch_indices):
                pred_curve_stitcher.add_patch(idx, pred_curves[b].astype(self.cube_dtype))
                gt_curve_stitcher.add_patch(  idx, gt_curves  [b].astype(self.cube_dtype))
                param_pred_stitcher.add_patch(idx, pred_params [b].astype(self.cube_dtype))
                gt_param_stitcher.add_patch(  idx, gt_params   [b].astype(self.cube_dtype))

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

        return self._finalize_results(
            pred_curve_stitcher, gt_curve_stitcher,
            param_pred_stitcher, gt_param_stitcher,
            pixel_mse, pixel_mae, pixel_r2, pixel_cos, pixel_peak, pixel_w,
        )

    def _finalize_results(
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
     
        pred_curves_cube = pred_curve_stitcher.finalize_cube()
        gt_curves_cube   = gt_curve_stitcher.finalize_cube()
        params_pred_cube = param_pred_stitcher.finalize_cube()
        params_gt_cube   = gt_param_stitcher.finalize_cube()

        n_K = self.run.n_gaussians
        for k in range(n_K):
            a_gt    = params_gt_cube[3 * k]
            mask_gt = a_gt < 1e-7
            params_gt_cube  [3 * k + 1][mask_gt] = np.nan
            params_gt_cube  [3 * k + 2][mask_gt] = np.nan
            params_pred_cube[3 * k + 1][mask_gt] = np.nan
            params_pred_cube[3 * k + 2][mask_gt] = np.nan

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

        self.logger.subsection(f"Curves cube    (denorm) : {pred_curves_cube.shape}")
        self.logger.subsection(f"Params cube    (denorm) : {params_pred_cube.shape}")
        self.logger.subsection(f"GT params cube (denorm) : {params_gt_cube.shape}")
        self.logger.subsection(f"Mean pixel MSE (denorm) : {pixel_mse.mean():.4g}  (pred vs gt)")
        self.logger.subsection(f"Mean pixel R²  (denorm) : {pixel_r2.mean():.4g}  (pred vs gt)\n")

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
        backend = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.section("[Inference: Predict]")
        self.logger.subsection(f"Backend      : PyTorch / {backend}")
        self.logger.subsection(f"Cube dir     : {self.cube_dir}")
        self.logger.subsection(f"Window       : {self.window_kind}")
        self.logger.subsection(f"Cube dtype   : {self.cube_dtype}")
        self.logger.subsection(f"CPU workers  : {self.cpu_workers}\n")

        all_indices, all_pred_params, all_gt_params = self._forward_pass()
        cpu_results = self._compute_metrics(all_pred_params, all_gt_params)
        results     = self._stitch_results(all_indices, cpu_results)
        
        return results
