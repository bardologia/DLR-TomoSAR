from __future__ import annotations

import numpy as np
import torch

from configuration.param_extraction_config import ExtractionConfig, FitMode, FitSettings
from pipelines.training_pipeline.loss       import PhysicsComponents
from tools.gaussians                        import GaussianMixture
from tools.tomo_geometry                    import TomoGeometry
from tools.track_baselines                  import TrackBaselines


class PhysicsQuantitiesCheck:
    LOSS_KINDS = {
        "total_power":      "masked relative L1 error",
        "moments":          "weighted normalised blend",
        "coherence_resyn":  "mean squared complex-coherence diff",
        "covariance_match": "normalised squared-Frobenius ratio",
        "capon_cycle":      "normalised spectrum MSE",
    }

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    def run(self) -> dict:
        capon_profiles, gauss_profiles, x_axis = self._load_profiles()

        x_t     = torch.tensor(x_axis, dtype=torch.float32, device=self.device)
        dx      = float(x_axis[1] - x_axis[0])
        capon_t = self._to_tensor(capon_profiles)
        gauss_t = self._to_tensor(gauss_profiles)

        geometry = TomoGeometry(self.config.geometry.resolved(self.config.dataset_path, secondary_labels=self.config.secondary_labels), x_t)
        self.logger.kv_table(geometry.describe(), title="Tomographic Geometry")

        losses    = self._loss_terms(gauss_t, capon_t, x_t, dx, geometry)
        moments   = self._moment_agreement(gauss_t, capon_t, x_t, dx)
        coherence = self._coherence_agreement(gauss_t, capon_t, dx, geometry)

        self.logger.metrics_table(losses,    ["Term", "Kind", "Gauss vs Capon"],                                       title="Physics Loss Terms on Ground-Truth Pairs")
        self.logger.metrics_table(moments,   ["Quantity", "Unit", "MAE", "Rel MAE [%]", "Pearson r", "Signed bias"],   title="Per-Pixel Moment Agreement")
        self.logger.metrics_table(coherence, ["Track", "Role", "kz [rad/m]", "Mean |dGamma|"],                         title="Per-Baseline Coherence Agreement")

        return {"losses": losses, "moments": moments, "coherence": coherence}

    def _load_profiles(self) -> tuple:
        cfg = self.config

        extraction = ExtractionConfig(
            processed_data_path = cfg.dataset_path,
            height_range        = cfg.height_range,
            output_prefix       = cfg.output_prefix,
            output_suffix       = cfg.output_suffix,
            fit_settings        = FitSettings(fit_config=FitMode.SigmaOnly(k_max=cfg.fit_k_max)),
        )

        tomogram_path   = extraction.discover_tomogram_path()
        parameters_path = extraction.parameters_npy_path
        height_range    = extraction.discover_height_range()

        self.logger.kv_table({
            "Tomogram":     tomogram_path,
            "Parameters":   parameters_path,
            "Height range": height_range,
            "Pixels":       cfg.n_pixels,
            "Seed":         cfg.seed,
        }, title="Inputs")

        tomogram   = np.load(tomogram_path,   mmap_mode="r")
        parameters = np.load(parameters_path, mmap_mode="r")

        n_bins = tomogram.shape[0]
        x_axis = np.linspace(height_range[0], height_range[1], n_bins)
        dx     = float(x_axis[1] - x_axis[0])

        capon_flat = np.asarray(tomogram,   dtype=np.float32).reshape(n_bins, -1).T
        param_flat = np.asarray(parameters, dtype=np.float32).reshape(parameters.shape[0], -1).T

        mass  = capon_flat.sum(axis=1) * dx
        valid = np.flatnonzero(mass > cfg.physics_floor)

        rng    = np.random.default_rng(cfg.seed)
        n_take = min(cfg.n_pixels, valid.size)
        picked = rng.choice(valid, size=n_take, replace=False)

        capon_profiles = capon_flat[picked]
        params_picked  = param_flat[picked]

        amps  = params_picked[:, 0::3]
        mus   = params_picked[:, 1::3]
        sigs  = params_picked[:, 2::3]

        gauss_profiles = GaussianMixture.evaluate_batch(x_axis, amps, mus, sigs)

        self.logger.subsection(f"Sampled {n_take} of {valid.size} valid pixels ({mass.size} total)")

        return capon_profiles, gauss_profiles, x_axis

    def _to_tensor(self, profiles: np.ndarray) -> torch.Tensor:
        t = torch.tensor(profiles, dtype=torch.float32, device=self.device)
        return t.T.reshape(1, t.shape[1], 1, t.shape[0])

    def _loss_terms(self, gauss_t, capon_t, x_t, dx, geometry) -> list:
        pc  = PhysicsComponents
        cfg = self.config

        values = {
            "total_power":      pc.total_power(          gauss_t, capon_t, dx, cfg.physics_floor),
            "moments":          pc.moments(              gauss_t, capon_t, x_t, dx, cfg.physics_floor, cfg.moments_weights),
            "coherence_resyn":  pc.coherence_resynthesis(gauss_t, capon_t, geometry.steering, dx, cfg.physics_floor),
            "covariance_match": pc.covariance_matching(  gauss_t, capon_t, geometry.outer, dx, cfg.physics_floor),
            "capon_cycle":      pc.capon_cycle(          gauss_t, capon_t, geometry.steering, geometry.outer, dx, cfg.capon_loading, cfg.physics_floor),
        }

        return [{"Term": name, "Kind": self.LOSS_KINDS[name], "Gauss vs Capon": f"{float(val):.6f}"} for name, val in values.items()]

    def _pearson(self, gv: np.ndarray, cv: np.ndarray) -> str:
        if gv.std() == 0.0 or cv.std() == 0.0:
            self.logger.warning("Pearson r undefined: zero variance in one moment array; reporting 'degenerate'")
            return "degenerate"

        r = float(np.corrcoef(gv, cv)[0, 1])

        if not np.isfinite(r):
            self.logger.warning("Pearson r undefined: corrcoef returned a non-finite value; reporting 'degenerate'")
            return "degenerate"

        return f"{r:.4f}"

    def _moment_agreement(self, gauss_t, capon_t, x_t, dx) -> list:
        floor       = self.config.physics_floor
        height_span = float(x_t.max() - x_t.min())

        g0, g1, g2 = PhysicsComponents.moment_sums(gauss_t, x_t, dx)
        c0, c1, c2 = PhysicsComponents.moment_sums(capon_t, x_t, dx)

        g_mean = g1 / g0.clamp(min=floor)
        c_mean = c1 / c0.clamp(min=floor)

        g_spread = torch.sqrt((g2 / g0.clamp(min=floor) - g_mean ** 2).clamp(min=0.0))
        c_spread = torch.sqrt((c2 / c0.clamp(min=floor) - c_mean ** 2).clamp(min=0.0))

        specs = [
            ("mass m0",         "power*m", g0,       c0,       float(np.abs(c0.flatten().cpu().numpy()).mean())),
            ("mean elevation",  "m",       g_mean,   c_mean,   height_span),
            ("vertical spread", "m",       g_spread, c_spread, float(c_spread.flatten().cpu().numpy().mean())),
        ]

        rows = []

        for name, unit, g, c, rel_scale in specs:
            gv = g.flatten().cpu().numpy()
            cv = c.flatten().cpu().numpy()

            mae    = float(np.abs(gv - cv).mean())
            signed = float((gv - cv).mean())
            rel    = 100.0 * mae / max(rel_scale, floor)
            pear   = self._pearson(gv, cv)

            rows.append({
                "Quantity":    name,
                "Unit":        unit,
                "MAE":         f"{mae:.4f}",
                "Rel MAE [%]": f"{rel:.2f}",
                "Pearson r":   pear,
                "Signed bias": f"{signed:+.4f}",
            })

        return rows

    def _track_labels(self, geometry) -> list | None:
        cfg = self.config.geometry

        if cfg.baselines_source == "manual" or len(cfg.kz_values) > 0:
            return None

        path = cfg.baselines_file(self.config.dataset_path)
        if not path.exists():
            return None

        return list(TrackBaselines.load(path).subset(self.config.secondary_labels).labels)

    def _coherence_agreement(self, gauss_t, capon_t, dx, geometry) -> list:
        floor  = self.config.physics_floor
        labels = self._track_labels(geometry)

        g0 = gauss_t.sum(dim=1) * dx
        c0 = capon_t.sum(dim=1) * dx

        gg = torch.einsum("nk,bkhw->bnhw", geometry.steering, gauss_t.to(geometry.steering.dtype)) * dx
        gc = torch.einsum("nk,bkhw->bnhw", geometry.steering, capon_t.to(geometry.steering.dtype)) * dx

        gg = gg / g0.clamp(min=floor).unsqueeze(1)
        gc = gc / c0.clamp(min=floor).unsqueeze(1)

        diff = (gg - gc).abs().mean(dim=(0, 2, 3))

        rows = []

        for n in range(geometry.n_tracks):
            track = labels[n] if labels is not None else f"index {n}"
            kz    = float(geometry.kz[n])
            role  = "kz=0 reference" if kz == 0.0 else "secondary"

            rows.append({
                "Track":         track,
                "Role":          role,
                "kz [rad/m]":    f"{kz:.4f}",
                "Mean |dGamma|": f"{float(diff[n]):.4f}",
            })

        return rows
