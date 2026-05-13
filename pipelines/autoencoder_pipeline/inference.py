from __future__ import annotations

import json
from pathlib import Path
from typing  import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.logger import Logger

from .config        import AutoencoderConfig
from .model         import Autoencoder
from .normalization import ProfileNormalizer
from .plotter       import Plotter


class Inference:

    def __init__(
        self,
        model         : Autoencoder,
        ae_config     : AutoencoderConfig,
        loader        : DataLoader,
        run_directory : Path,
        logger        : Logger,
        plotter       : Plotter,
        split_name    : str        = "test",
        max_profiles  : int | None = None,
    ) -> None:
        self.model        = model
        self.ae_config    = ae_config
        self.loader       = loader
        self.run_dir      = Path(run_directory)
        self.logger       = logger
        self.plotter      = plotter
        self.split_name   = split_name
        self.max_profiles = max_profiles
        self.device       = next(model.parameters()).device

        self.embed_dir = Path(ae_config.io.embed_dir or self.run_dir / "embeddings")
        self.recon_dir = Path(ae_config.io.recon_dir or self.run_dir / "reconstructions")
        self.embed_dir.mkdir(parents=True, exist_ok=True)
        self.recon_dir.mkdir(parents=True, exist_ok=True)

        self.normalizer = ProfileNormalizer(ae_config.data)

    @torch.no_grad()
    def run(self) -> dict[str, Any]:
        self.logger.section(f"[Inference:{self.split_name}]")

        profiles, recons, latents, projections, errors, scales = self._encode_all()
        self.logger.subsection(f"Profiles encoded : {profiles.shape[0]:,}")
        self.logger.subsection(f"Latent dim       : {latents.shape[1]}")

        h5_path  = self._dump_h5(profiles, recons, latents, projections, errors, scales)
        recon_st = self._reconstruction_stats(profiles, recons, errors)
        embed_st = self._embedding_stats(latents, projections)
        self._write_json(self.embed_dir / f"{self.split_name}_embedding_stats.json", embed_st)
        self._write_json(self.recon_dir / f"{self.split_name}_reconstruction_stats.json", recon_st)

        gallery_paths = self._save_galleries(profiles, recons, errors, scales)
        spectrum_path = self.plotter.plot_embedding_spectrum(self.split_name, np.array(embed_st["covariance_eigenvalues"]))
        hist_path     = self.plotter.plot_error_histogram(self.split_name, errors)
        pca_path, umap_path = self._save_2d_projections(latents, errors)

        summary = {
            "split"                : self.split_name,
            "num_profiles"         : int(profiles.shape[0]),
            "profile_length"       : int(profiles.shape[1]),
            "latent_dim"           : int(latents.shape[1]),
            "projection_dim"       : int(projections.shape[1]) if projections is not None else None,
            "reconstruction_stats" : recon_st,
            "embedding_stats"      : embed_st,
            "h5_path"              : str(h5_path),
            "gallery_paths"        : {k: str(v) for k, v in gallery_paths.items()},
            "spectrum_path"        : str(spectrum_path),
            "error_histogram_path" : str(hist_path),
            "pca_path"             : str(pca_path) if pca_path is not None else None,
            "umap_path"            : str(umap_path) if umap_path is not None else None,
        }
        self._write_json(self.run_dir / f"{self.split_name}_inference_summary.json", summary)
        self.logger.subsection(f"[Summary] -> {self.split_name}_inference_summary.json")
        return summary

    def _encode_all(self):
        self.model.eval()
        profiles_l, recons_l, latents_l, proj_l, errors_l, scales_l = [], [], [], [], [], []
        seen = 0

        for profile_a, _, scale_a in self.loader:
            profile_a = profile_a.to(self.device, non_blocking=True)
            output    = self.model(profile_a)
            err       = (output.reconstruction - profile_a).pow(2).mean(dim=1)

            profiles_l.append(profile_a.detach().cpu().numpy())
            recons_l  .append(output.reconstruction.detach().cpu().numpy())
            latents_l .append(output.latent.detach().cpu().numpy())
            errors_l  .append(err.detach().cpu().numpy())
            scales_l  .append(scale_a.numpy() if isinstance(scale_a, torch.Tensor) else np.asarray(scale_a, dtype=np.float32))
            if output.projection is not None:
                proj_l.append(output.projection.detach().cpu().numpy())

            seen += profile_a.shape[0]
            if self.max_profiles is not None and seen >= self.max_profiles:
                break

        profiles    = np.concatenate(profiles_l, axis=0)
        recons      = np.concatenate(recons_l,   axis=0)
        latents     = np.concatenate(latents_l,  axis=0)
        errors      = np.concatenate(errors_l,   axis=0)
        scales      = np.concatenate(scales_l,   axis=0)
        projections = np.concatenate(proj_l, axis=0) if proj_l else None

        if self.max_profiles is not None:
            profiles, recons, latents, errors, scales = (
                profiles[:self.max_profiles], recons[:self.max_profiles],
                latents[:self.max_profiles],  errors[:self.max_profiles],
                scales[:self.max_profiles],
            )
            if projections is not None:
                projections = projections[:self.max_profiles]

        return profiles, recons, latents, projections, errors, scales

    def _dump_h5(self, profiles, recons, latents, projections, errors, scales) -> Path:
        path = self.embed_dir / f"{self.split_name}_latents.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("profiles",        data=profiles, compression="gzip", compression_opts=4)
            f.create_dataset("reconstructions", data=recons,   compression="gzip", compression_opts=4)
            f.create_dataset("latents",         data=latents,  compression="gzip", compression_opts=4)
            f.create_dataset("recon_mse",       data=errors,   compression="gzip", compression_opts=4)
            f.create_dataset("scales",          data=scales,   compression="gzip", compression_opts=4)
            if projections is not None:
                f.create_dataset("projections", data=projections, compression="gzip", compression_opts=4)
            f.attrs["split"]          = self.split_name
            f.attrs["num_profiles"]   = int(profiles.shape[0])
            f.attrs["profile_length"] = int(profiles.shape[1])
            f.attrs["latent_dim"]     = int(latents.shape[1])
        self.logger.subsection(f"[HDF5] -> {path}")
        return path

    @staticmethod
    def _reconstruction_stats(profiles: np.ndarray, recons: np.ndarray, errors: np.ndarray) -> dict[str, Any]:
        diff   = recons - profiles
        mae    = np.mean(np.abs(diff), axis=1)
        rmse   = np.sqrt(np.mean(diff ** 2, axis=1))
        peak   = np.maximum(np.abs(profiles).max(axis=1), 1e-12)
        psnr   = 20.0 * np.log10(peak) - 10.0 * np.log10(np.maximum(rmse ** 2, 1e-20))
        var_t  = profiles.var(axis=1) + 1e-12
        var_e  = diff.var(axis=1)
        r2     = 1.0 - (var_e / var_t)
        
        return {
            "mse_mean"    : float(errors.mean()),
            "mse_median"  : float(np.median(errors)),
            "mse_min"     : float(errors.min()),
            "mse_max"     : float(errors.max()),
            "mse_std"     : float(errors.std()),
            "mae_mean"    : float(mae.mean()),
            "mae_median"  : float(np.median(mae)),
            "rmse_mean"   : float(rmse.mean()),
            "psnr_mean"   : float(psnr.mean()),
            "psnr_median" : float(np.median(psnr)),
            "r2_mean"     : float(r2.mean()),
            "r2_median"   : float(np.median(r2)),
        }

    @staticmethod
    def _embedding_stats(latents: np.ndarray, projections: np.ndarray | None) -> dict[str, Any]:
        z       = latents
        z_cent  = z - z.mean(axis=0, keepdims=True)
        std     = z.std(axis=0)
        cov     = (z_cent.T @ z_cent) / max(1, z.shape[0] - 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 0.0, None)
        eigvals_sorted = np.sort(eigvals)[::-1]

        active_dims = int(np.sum(std > 1e-2))
        s_sum       = float(eigvals.sum())
        s_sq        = float((eigvals ** 2).sum())
        pr          = (s_sum ** 2) / (s_sq + 1e-12)

        ratios     = eigvals_sorted / (s_sum + 1e-12)
        ent        = -(ratios * np.log(ratios + 1e-12)).sum()
        eff_rank   = float(np.exp(ent))

        cum        = np.cumsum(ratios)
        n_for_90   = int(np.searchsorted(cum, 0.90) + 1) if len(cum) else 0
        n_for_95   = int(np.searchsorted(cum, 0.95) + 1) if len(cum) else 0

        stats = {
            "latent_dim"               : int(z.shape[1]),
            "mean_per_dim"             : z.mean(axis=0).tolist(),
            "std_per_dim"              : std.tolist(),
            "min_per_dim"              : z.min(axis=0).tolist(),
            "max_per_dim"              : z.max(axis=0).tolist(),
            "active_dimensions"        : active_dims,
            "covariance_eigenvalues"   : eigvals_sorted.tolist(),
            "explained_variance_ratio" : ratios.tolist(),
            "participation_ratio"      : pr,
            "effective_rank"           : eff_rank,
            "components_for_90pct"     : n_for_90,
            "components_for_95pct"     : n_for_95,
        }
        if projections is not None:
            stats["projection_std_per_dim"] = projections.std(axis=0).tolist()
        return stats

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, default=str)

    def _save_galleries(self, profiles : np.ndarray, recons : np.ndarray, errors : np.ndarray, scales   : np.ndarray) -> dict[str, Path]:
        order = np.argsort(errors)
        n     = profiles.shape[0]
        k     = min(16, n)

        selections = {
            "best"   : order[:k],
            "worst"  : order[-k:][::-1],
            "median" : order[max(0, n // 2 - k // 2): n // 2 - k // 2 + k],
            "random" : np.random.default_rng(0).choice(n, size=k, replace=False) if n >= k else np.arange(n),
        }
        out: dict[str, Path] = {}
        
        for tag, idx in selections.items():
            np.savez(self.recon_dir / f"{self.split_name}_{tag}.npz", profiles=profiles[idx], reconstructions=recons[idx], mse=errors[idx], indices=idx, scales=scales[idx])
            profiles_dn = self.normalizer.invert_numpy(profiles[idx], scales[idx])
            recons_dn   = self.normalizer.invert_numpy(recons[idx],   scales[idx])
            out[tag] = self.plotter.plot_reconstruction_gallery(self.split_name, tag, profiles_dn, recons_dn, errors[idx], idx,)
            self.logger.subsection(f"[Gallery:{self.split_name}/{tag}] -> {out[tag].name}")
        
        return out

    def _save_2d_projections(self, latents: np.ndarray, errors: np.ndarray):
        z = latents - latents.mean(axis=0, keepdims=True)
        try:
            u, s, _ = np.linalg.svd(z, full_matrices=False)
            pca_2d  = u[:, :2] * s[:2]
        except np.linalg.LinAlgError:
            pca_2d = np.zeros((z.shape[0], 2), dtype=np.float32)
        np.save(self.embed_dir / f"{self.split_name}_pca_2d.npy", pca_2d)
        pca_path = self.plotter.plot_embedding_scatter(self.split_name, pca_2d, "pca_2d", "PCA(2)", color_by=errors)

        umap_path = None
        try:
            import umap
            n_neighbors = min(15, max(2, latents.shape[0] - 1))
            reducer  = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=0)
            umap_2d  = reducer.fit_transform(latents)
            np.save(self.embed_dir / f"{self.split_name}_umap_2d.npy", umap_2d)
            umap_path = self.plotter.plot_embedding_scatter(self.split_name, umap_2d, "umap_2d", "UMAP(2)", color_by=errors)
        except Exception as exc:
            self.logger.subsection(f"[UMAP:{self.split_name}] skipped — {type(exc).__name__}: {exc}")

        return pca_path, umap_path
