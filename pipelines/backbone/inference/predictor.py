from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing             import List, Optional, Tuple

import numpy as np
import torch

from pipelines.backbone.dataset.spatial   import GridInfo
from pipelines.backbone.inference.loader  import Run
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from pipelines.backbone.inference.metrics import Metrics, Result
from tools.data.gaussians                 import GaussianReconstructor
from tools.monitoring.logger              import Logger


class CubeStitcher:
    def __init__(
        self,
        grid           : GridInfo,
        n_channels     : int,
        window_kind    : str           = "hann",
        dtype          : str           = "float32",
        memmap_path    : Optional[str] = None,
    ) -> None:
        self.grid       = grid
        self.n_channels = int(n_channels)
        self.dtype      = np.dtype(dtype)
        self.window     = CubeStitcher.make_patch_window(grid.patch_size, kind=window_kind)

        H_pad, W_pad = grid.padded_size
        shape_pad    = (self.n_channels, H_pad, W_pad)

        if memmap_path is not None:
            self._accum = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=self.dtype, shape=shape_pad)
            self._accum[...] = 0
        else:
            self._accum = np.zeros(shape_pad, dtype=self.dtype)

        self._weight = np.zeros((H_pad, W_pad), dtype=np.float32)

    @staticmethod
    def make_patch_window(patch_size: Tuple[int, int], kind: str = "hann") -> np.ndarray:
        ph, pw = patch_size
        if kind == "uniform":
            return np.ones((ph, pw), dtype=np.float32)

        if kind == "hann":
            wv = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(ph) + 0.5) / ph)
            wh = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(pw) + 0.5) / pw)

        elif kind == "triangular":
            wv = 1.0 - np.abs((np.arange(ph) + 0.5) / ph * 2.0 - 1.0)
            wh = 1.0 - np.abs((np.arange(pw) + 0.5) / pw * 2.0 - 1.0)

        else:
            raise ValueError(f"Unknown window kind: {kind!r}")

        wv = np.clip(wv, 1e-3, None).astype(np.float32)
        wh = np.clip(wh, 1e-3, None).astype(np.float32)

        return np.outer(wv, wh)

    @property
    def number_of_patches(self) -> int:
        return self.grid.number_of_patches

    def add_patch(self, idx: int, patch: np.ndarray) -> None:
        ph, pw = self.grid.patch_size
        iv, ih = divmod(idx, self.grid.n_h)
        v0 = iv * self.grid.stride
        h0 = ih * self.grid.stride
        w  = self.window

        self._accum[:, v0:v0 + ph, h0:h0 + pw] += (patch * w[None, :, :]).astype(self.dtype, copy=False)
        self._weight[v0:v0 + ph, h0:h0 + pw]   += w

    def finalize_cube(self) -> np.ndarray:
        H, W         = self.grid.spatial_size
        pad_t, pad_l = self.grid.pad_top, self.grid.pad_left

        weight_safe = np.where(self._weight > 0, self._weight, 1.0)
        cube        = self._accum / weight_safe[None, :, :]
        cube        = cube[:, pad_t:pad_t + H, pad_l:pad_l + W]

        return np.ascontiguousarray(cube.astype(self.dtype, copy=False))


class Predictor:
    def __init__(
        self,
        run            : Run,
        logger         : Logger,
        *,
        window_kind    : str,
        cube_dtype     : str,
        save_cubes     : bool,
        meta           : InferenceMetadata,
        cpu_workers    : int | None = None,
    ) -> None:

        self.run         = run
        self.logger      = logger
        self.window_kind = window_kind
        self.cube_dtype  = cube_dtype
        self.save_cubes  = save_cubes
        self.cube_dir    = meta.cube_dir
        self.cpu_workers = cpu_workers if cpu_workers is not None else min(8, os.cpu_count() or 1)

    def _forward_pass(self) -> Tuple[List[List[int]], List[np.ndarray], List[np.ndarray]]:
        run = self.run
        n_K = run.n_gaussians

        all_indices     : List[List[int]]  = []
        all_pred_params : List[np.ndarray] = []
        all_gt_params   : List[np.ndarray] = []
        sample_count = 0

        with self.logger.track(transient=True) as prog:
            task = prog.add_task("[section]GPU Forward Pass[/section]", total=len(run.loader))
            for batch in run.loader:
                images, gt_params_b = batch[0], batch[1]

                pred_params = run.model(images)

                gt_params_ready = gt_params_b[:, :n_K * 3]
                gt_params_ready = run.dataset.normalizer.denormalize_output(gt_params_ready)

                B = images.shape[0]
                all_indices.append(list(range(sample_count, sample_count + B)))
                all_pred_params.append(pred_params)
                all_gt_params.append(gt_params_ready.cpu().numpy().astype(np.float32))
                sample_count += B
                prog.advance(task)

        return all_indices, all_pred_params, all_gt_params

    def _compute_metrics(self, all_pred_params : List[np.ndarray], all_gt_params : List[np.ndarray]) -> List[tuple]:
        run = self.run
        n_K = run.n_gaussians

        tasks = [(pred, gt, run.x_axis, n_K) for pred, gt in zip(all_pred_params, all_gt_params)]
        results: List[tuple | None] = [None] * len(tasks)

        with self.logger.track(transient=True) as prog:
            task_id = prog.add_task("[section]CPU Metrics[/section]", total=len(tasks))
            with ProcessPoolExecutor(max_workers=self.cpu_workers) as pool:
                futures = {pool.submit(Predictor._cpu_worker, t): i for i, t in enumerate(tasks)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    results[idx] = fut.result()
                    prog.advance(task_id)

        return results

    @staticmethod
    def _cpu_worker(args: tuple) -> tuple:
        pred_params_chunk, gt_params_chunk, x_axis, n_gaussians = args

        x          = x_axis.reshape(1, 1, -1, 1, 1).astype(np.float32)
        B, _, H, W = pred_params_chunk.shape

        n_K        = n_gaussians
        pred_gauss = pred_params_chunk[:, :n_K * 3].reshape(B, n_K, 3, H, W).astype(np.float32)
        gt_gauss   = gt_params_chunk[:,   :n_K * 3].reshape(B, n_K, 3, H, W).astype(np.float32)

        sort_key   = np.where(gt_gauss[:, :, 0] < 1e-3, np.inf, gt_gauss[:, :, 1])
        sort_idx   = np.argsort(sort_key, axis=1)
        sort_idx_e = sort_idx[:, :, None, :, :].repeat(3, axis=2)

        gt_gauss_matched   = np.take_along_axis(gt_gauss, sort_idx_e, axis=1)

        pred_gauss_flat = pred_gauss.reshape(      B, n_K * 3, H, W)
        gt_gauss_flat   = gt_gauss_matched.reshape(B, n_K * 3, H, W)

        pred_curves = GaussianReconstructor.reconstruct_batch(pred_gauss,       x)
        gt_curves   = GaussianReconstructor.reconstruct_batch(gt_gauss_matched, x)

        return (
            pred_curves,
            gt_curves,
            pred_gauss_flat,
            gt_gauss_flat,
        )

    def _stitch_results(self, all_indices : List[List[int]], cpu_results : List[tuple]) -> Result:
        run    = self.run
        n_K    = run.n_gaussians
        out_ch = run.out_channels
        n_elev = run.x_axis_length

        pred_curve_stitcher = self._create_stitcher(n_elev,  "pred_curves")
        gt_curve_stitcher   = self._create_stitcher(n_elev,  "gt_curves")
        param_pred_stitcher = self._create_stitcher(out_ch,  "params_pred")
        gt_param_stitcher   = self._create_stitcher(n_K * 3, "params_gt")

        for batch_indices, (pred_curves, gt_curves, pred_params, gt_params) in zip(all_indices, cpu_results):
            for b, idx in enumerate(batch_indices):
                pred_curve_stitcher.add_patch(idx, pred_curves[b].astype(self.cube_dtype))
                gt_curve_stitcher.add_patch(  idx, gt_curves  [b].astype(self.cube_dtype))
                param_pred_stitcher.add_patch(idx, pred_params [b].astype(self.cube_dtype))
                gt_param_stitcher.add_patch(  idx, gt_params   [b].astype(self.cube_dtype))

        return self._finalize_results(
            pred_curve_stitcher, gt_curve_stitcher,
            param_pred_stitcher, gt_param_stitcher,
        )

    def _create_stitcher(self, n_channels: int, name: str) -> CubeStitcher:
        memmap_path = str(self.cube_dir / f"_tmp_{name}.npy") if self.save_cubes else None
        return CubeStitcher(
            grid        = self.run.grid,
            n_channels  = n_channels,
            window_kind = self.window_kind,
            dtype       = self.cube_dtype,
            memmap_path = memmap_path,
        )

    def _finalize_results(
        self,
        pred_curve_stitcher : CubeStitcher,
        gt_curve_stitcher   : CubeStitcher,
        param_pred_stitcher : CubeStitcher,
        gt_param_stitcher   : CubeStitcher,
    ) -> Result:

        pred_curves_cube = pred_curve_stitcher.finalize_cube()
        gt_curves_cube   = gt_curve_stitcher.finalize_cube()
        params_pred_cube = param_pred_stitcher.finalize_cube()
        params_gt_cube   = gt_param_stitcher.finalize_cube()

        n_K = self.run.n_gaussians
        for k in range(n_K):
            a_gt    = params_gt_cube[3 * k]
            mask_gt = a_gt < 1e-7
            params_gt_cube[3 * k + 1][mask_gt] = np.nan
            params_gt_cube[3 * k + 2][mask_gt] = np.nan

        pixel_maps     = Metrics.curve_pixel_metrics(pred_curves_cube, gt_curves_cube)
        pixel_mse      = pixel_maps["mse"]
        pixel_mae      = pixel_maps["mae"]
        pixel_r2       = pixel_maps["r2"]
        pixel_cos      = pixel_maps["cos"]
        pixel_peak_idx = pixel_maps["peak"].astype(np.int32)

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
                tmp.unlink(missing_ok=True)

        self.logger.section("[Inference: Results]")
        self.logger.kv_table({
            "Curves cube    (denorm)": pred_curves_cube.shape,
            "Params cube    (denorm)": params_pred_cube.shape,
            "GT params cube (denorm)": params_gt_cube.shape,
            "Mean pixel MSE (denorm)": f"{pixel_mse.mean():.4g}  (pred vs gt)",
            "Mean pixel R²  (denorm)": f"{pixel_r2.mean():.4g}  (pred vs gt)",
        })

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
        backend = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.section("[Inference: Predict]")
        self.logger.kv_table({
            "Backend":     f"PyTorch / {backend}",
            "Cube dir":    self.cube_dir,
            "Window":      self.window_kind,
            "Cube dtype":  self.cube_dtype,
            "CPU workers": self.cpu_workers,
        })

        all_indices, all_pred_params, all_gt_params = self._forward_pass()
        cpu_results = self._compute_metrics(all_pred_params, all_gt_params)
        results     = self._stitch_results(all_indices, cpu_results)

        return results
