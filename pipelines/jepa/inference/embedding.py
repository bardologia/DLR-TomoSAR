from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from pipelines.backbone.inference.predictor import CubeStitcher
from tools.data.gaussians                   import GaussianReconstructor
from tools.monitoring.logger                import Logger


class JepaEmbeddingMaps:
    def __init__(self, embedding_error: np.ndarray, embedding_cosine: np.ndarray, decode_mse: np.ndarray, chain_mse: np.ndarray) -> None:
        self.embedding_error  = embedding_error
        self.embedding_cosine = embedding_cosine
        self.decode_mse       = decode_mse
        self.chain_mse        = chain_mse


class JepaEmbeddingEvaluator:
    def __init__(self, run, logger: Logger, window_kind: str = "uniform") -> None:
        self._run        = run
        self.logger      = logger
        self.adapter     = run.model.module
        self.device      = torch.device(run.model.device)
        self.window_kind = window_kind

    def _gt_curves(self, gt_params_batch: torch.Tensor, x: np.ndarray) -> torch.Tensor:
        run = self._run
        n_K = run.n_gaussians

        gt_params = gt_params_batch[:, :n_K * 3]
        gt_phys   = run.dataset.normalizer.denormalize_output(gt_params).cpu().numpy().astype(np.float32)

        B, _, H, W = gt_phys.shape
        gt_gauss   = gt_phys.reshape(B, n_K, 3, H, W)
        gt_curves  = GaussianReconstructor.reconstruct_batch(gt_gauss, x)

        return torch.from_numpy(gt_curves.astype(np.float32)).to(self.device)

    def _batch_fields(self, images: torch.Tensor, gt_curves: torch.Tensor) -> Dict[str, torch.Tensor]:
        jepa        = self.adapter.jepa
        autoencoder = jepa.profile_autoencoder
        normalizer  = self.adapter.profile_normalizer

        z_hat_n    = autoencoder.normalize_embedding(jepa(images))
        gt_curve_n = normalizer.normalize(gt_curves)
        z_star_n   = autoencoder.encode(gt_curve_n)

        decode_hat  = autoencoder.decode(z_hat_n)
        decode_star = autoencoder.decode(z_star_n)

        return {
            "embedding_sq"  : (z_hat_n - z_star_n) ** 2,
            "embedding_cos" : F.cosine_similarity(z_hat_n, z_star_n, dim=1).unsqueeze(1),
            "decode_sq"     : (decode_star - gt_curve_n) ** 2,
            "chain_sq"      : (decode_hat  - gt_curve_n) ** 2,
            "z_star_sq"     : z_star_n ** 2,
        }

    def _accumulate(self) -> tuple[Dict[str, float], np.ndarray, JepaEmbeddingMaps]:
        run   = self._run
        x     = np.asarray(run.x_axis, dtype=np.float32).reshape(1, 1, -1, 1, 1)
        grid  = run.grid

        error_st  = CubeStitcher(grid=grid, n_channels=1, window_kind=self.window_kind, dtype="float32")
        cosine_st = CubeStitcher(grid=grid, n_channels=1, window_kind=self.window_kind, dtype="float32")
        decode_st = CubeStitcher(grid=grid, n_channels=1, window_kind=self.window_kind, dtype="float32")
        chain_st  = CubeStitcher(grid=grid, n_channels=1, window_kind=self.window_kind, dtype="float32")

        sums        : Dict[str, float] = {}
        per_dim_sum : np.ndarray | None = None
        sample_count = 0

        with torch.no_grad():
            with self.logger.track(transient=True) as prog:
                task = prog.add_task("[section]Embedding Diagnostics[/section]", total=len(run.loader))
                for batch in run.loader:
                    images    = batch[0].to(self.device)
                    gt_curves = self._gt_curves(batch[1], x)

                    fields = self._batch_fields(images, gt_curves)

                    embedding_map = fields["embedding_sq"].mean(dim=1, keepdim=True).cpu().numpy().astype(np.float32)
                    decode_map    = fields["decode_sq"].mean(dim=1, keepdim=True).cpu().numpy().astype(np.float32)
                    chain_map     = fields["chain_sq"].mean(dim=1, keepdim=True).cpu().numpy().astype(np.float32)
                    cosine_map    = fields["embedding_cos"].cpu().numpy().astype(np.float32)

                    for b in range(embedding_map.shape[0]):
                        error_st.add_patch( sample_count + b, embedding_map[b])
                        cosine_st.add_patch(sample_count + b, cosine_map[b])
                        decode_st.add_patch(sample_count + b, decode_map[b])
                        chain_st.add_patch( sample_count + b, chain_map[b])

                    dim_sum     = fields["embedding_sq"].sum(dim=(0, 2, 3)).cpu().numpy().astype(np.float64)
                    per_dim_sum = dim_sum if per_dim_sum is None else per_dim_sum + dim_sum

                    sums["embedding_sq"] = sums.get("embedding_sq", 0.0) + float(fields["embedding_sq"].sum())
                    sums["embedding_n"]  = sums.get("embedding_n",  0.0) + float(fields["embedding_sq"].numel())
                    sums["z_star_sq"]    = sums.get("z_star_sq",    0.0) + float(fields["z_star_sq"].sum())
                    sums["cosine_sum"]   = sums.get("cosine_sum",   0.0) + float(fields["embedding_cos"].sum())
                    sums["cosine_n"]     = sums.get("cosine_n",     0.0) + float(fields["embedding_cos"].numel())
                    sums["decode_sq"]    = sums.get("decode_sq",    0.0) + float(fields["decode_sq"].sum())
                    sums["chain_sq"]     = sums.get("chain_sq",     0.0) + float(fields["chain_sq"].sum())
                    sums["curve_n"]      = sums.get("curve_n",      0.0) + float(fields["chain_sq"].numel())

                    sample_count += embedding_map.shape[0]
                    prog.advance(task)

        if not sums or sums["embedding_n"] == 0.0:
            raise ValueError("JEPA embedding evaluation saw no samples; the inference loader is empty.")

        maps = JepaEmbeddingMaps(
            embedding_error  = error_st.finalize_cube()[0],
            embedding_cosine = cosine_st.finalize_cube()[0],
            decode_mse       = decode_st.finalize_cube()[0],
            chain_mse        = chain_st.finalize_cube()[0],
        )

        samples_per_dim = sums["embedding_n"] / per_dim_sum.size

        return sums, per_dim_sum / samples_per_dim, maps

    def _scalars(self, sums: Dict[str, float], per_dim_mse: np.ndarray, maps: JepaEmbeddingMaps) -> Dict[str, float]:
        embedding_mse = sums["embedding_sq"] / sums["embedding_n"]
        decode_mse    = sums["decode_sq"]    / sums["curve_n"]
        chain_mse     = sums["chain_sq"]     / sums["curve_n"]
        target_power  = sums["z_star_sq"]    / sums["embedding_n"]

        per_dim_rmse = np.sqrt(per_dim_mse)

        return {
            "jepa_embedding_mse"          : embedding_mse,
            "jepa_embedding_rmse"         : float(np.sqrt(embedding_mse)),
            "jepa_embedding_cosine"       : sums["cosine_sum"] / sums["cosine_n"],
            "jepa_embedding_nmse"         : embedding_mse / target_power if target_power > 0.0 else float("nan"),
            "jepa_embedding_dim_rmse_min" : float(per_dim_rmse.min()),
            "jepa_embedding_dim_rmse_max" : float(per_dim_rmse.max()),
            "jepa_embedding_dim_rmse_std" : float(per_dim_rmse.std()),
            "jepa_decode_mse_norm"        : decode_mse,
            "jepa_chain_mse_norm"         : chain_mse,
            "jepa_predictor_excess_mse"   : chain_mse - decode_mse,
            "jepa_decoder_floor_frac"     : decode_mse / chain_mse if chain_mse > 0.0 else float("nan"),
            "jepa_decode_amplification"   : (chain_mse - decode_mse) / embedding_mse if embedding_mse > 0.0 else float("nan"),
            "jepa_embedding_err_p95"      : float(np.percentile(maps.embedding_error[np.isfinite(maps.embedding_error)], 95.0)),
            "jepa_embedding_cosine_p5"    : float(np.percentile(maps.embedding_cosine[np.isfinite(maps.embedding_cosine)], 5.0)),
        }

    def _correlate_with_error(self, maps: JepaEmbeddingMaps, pixel_mse: np.ndarray | None) -> Dict[str, float]:
        if pixel_mse is None:
            return {}

        embedding = maps.embedding_error.reshape(-1).astype(np.float64)
        curve     = np.asarray(pixel_mse, dtype=np.float64).reshape(-1)

        if embedding.size != curve.size:
            raise ValueError(f"Embedding-error map holds {embedding.size} pixels but the curve-error map holds {curve.size}; they were stitched on different grids.")

        both = np.isfinite(embedding) & np.isfinite(curve)
        if both.sum() <= 2:
            return {"jepa_embedding_error_corr": float("nan")}

        return {"jepa_embedding_error_corr": float(np.corrcoef(embedding[both], curve[both])[0, 1])}

    def _report(self, metrics: Dict[str, float]) -> None:
        self.logger.section("[Inference: JEPA Embedding Diagnostics]")
        self.logger.kv_table({
            "Embedding MSE"           : f"{metrics['jepa_embedding_mse']:.6g}",
            "Embedding RMSE"          : f"{metrics['jepa_embedding_rmse']:.6g} (per-component spread; calibrates the profile-AE latent_noise_std)",
            "Embedding NMSE"          : f"{metrics['jepa_embedding_nmse']:.6g} (error power over target power)",
            "Embedding cosine"        : f"{metrics['jepa_embedding_cosine']:.4f}",
            "Per-dim RMSE spread"     : f"min {metrics['jepa_embedding_dim_rmse_min']:.4g}, max {metrics['jepa_embedding_dim_rmse_max']:.4g}",
            "Decoder-only MSE (norm)" : f"{metrics['jepa_decode_mse_norm']:.6g}",
            "Full-chain MSE (norm)"   : f"{metrics['jepa_chain_mse_norm']:.6g}",
            "Predictor excess MSE"    : f"{metrics['jepa_predictor_excess_mse']:.6g} ({metrics['jepa_decoder_floor_frac'] * 100.0:.1f}% of the chain error is the decoder floor)",
            "Decode amplification"    : f"{metrics['jepa_decode_amplification']:.4g} (curve MSE added per unit of embedding MSE)",
        })

        if "jepa_embedding_error_corr" in metrics:
            self.logger.subsection(f"Embedding error correlates with per-pixel curve MSE at r = {metrics['jepa_embedding_error_corr']:.3f}")

    def run(self, pixel_mse: np.ndarray | None = None) -> tuple[Dict[str, float], JepaEmbeddingMaps]:
        sums, per_dim_mse, maps = self._accumulate()

        metrics = self._scalars(sums, per_dim_mse, maps)
        metrics.update(self._correlate_with_error(maps, pixel_mse))

        self._report(metrics)

        return metrics, maps
