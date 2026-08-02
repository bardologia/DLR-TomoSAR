from __future__ import annotations

import io
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from project_paths            import ProjectPaths
from tools.reporting.plotting import PlotBase
from web_logger               import WebLogger


class ModelProbe:

    FAMILIES      = ("amp", "mu", "sigma")
    PERTURBATIONS = ("drop_channel", "scale_channel", "noise")

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:
        self.paths  = paths
        self.logger = logger
        self.lock   = threading.Lock()

        self.loaded = None
        self.status = {"state": "idle", "path": "", "progress": 0.0, "stage": "", "error": "", "info": None}

    def start_load(self, run_path: str, split: str = "test", device: str = "cpu") -> dict:
        with self.lock:
            if self.status["state"] == "loading":
                return {"ok": False, "error": "a model load is already running"}

            self.loaded = None
            self.status = {"state": "loading", "path": run_path, "progress": 0.0, "stage": "opening run", "error": "", "info": None}

        threading.Thread(target=self._load_worker, args=(run_path, split, device), name="ModelProbeLoad", daemon=True).start()
        return {"ok": True}

    def _set_load(self, progress: float, stage: str) -> None:
        with self.lock:
            self.status["progress"] = progress
            self.status["stage"]    = stage

    def _load_worker(self, run_path: str, split: str, device: str) -> None:
        try:
            from pipelines.backbone.inference.input_attribution import ChannelLabeler
            from pipelines.backbone.inference.loader            import RunLoader
            from pipelines.backbone.inference.probes            import PredictionCurves
            from tools.diagnostics.activation_recorder          import ActivationRecorder

            self._set_load(0.1, "loading checkpoint and dataset")

            run = RunLoader(Path(run_path), logger=self.logger).load(
                split           = split,
                batch_size      = 1,
                num_workers     = 0,
                device          = device,
                checkpoint_name = "best_model.pt",
            )

            self._set_load(0.8, "indexing layers")

            labels   = ChannelLabeler.build(run)
            layers   = ActivationRecorder(run.model.module).leaf_names()
            modules  = dict(run.model.module.named_modules())
            renderer = PredictionCurves(run.n_gaussians, run.x_axis)

            info = {
                "run"          : run_path,
                "backbone"     : run.backbone_name,
                "split"        : split,
                "in_channels"  : run.in_channels,
                "n_gaussians"  : run.n_gaussians,
                "azimuth_size" : run.split_region.azimuth_size,
                "range_size"   : run.split_region.range_size,
                "az_offset"    : run.split_region.azimuth_start,
                "rg_offset"    : run.split_region.range_start,
                "patch"        : list(run.dataset_config.patch.size),
                "channels"     : labels,
                "n_layers"     : len(layers),
            }

            with self.lock:
                self.loaded = {
                    "run"      : run,
                    "labels"   : labels,
                    "layers"   : layers,
                    "types"    : {name: type(modules[name]).__name__ for name in layers},
                    "renderer" : renderer,
                    "patch"    : tuple(run.dataset_config.patch.size),
                }
                self.status = {"state": "ready", "path": run_path, "progress": 1.0, "stage": "ready", "error": "", "info": info}

            self.logger.info(f"model probe loaded {run.backbone_name} from {run_path} ({len(layers)} layers)")

        except Exception as error:
            with self.lock:
                self.loaded = None
                self.status = {"state": "error", "path": run_path, "progress": 0.0, "stage": "", "error": f"{type(error).__name__}: {error}", "info": None}
            self.logger.error(f"model probe load failed: {error}")

    def load_status(self) -> dict:
        with self.lock:
            return dict(self.status)

    def layers(self) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}
            return {"ok": True, "layers": [{"name": name, "type": self.loaded["types"][name]} for name in self.loaded["layers"]]}

    def map_png(self) -> bytes | None:
        with self.lock:
            if self.loaded is None:
                return None
            primary = np.abs(self.loaded["run"].complex_inputs[0]).astype(np.float32)

        vmin, vmax = PlotBase._amplitude_clim(primary)

        buf = io.BytesIO()
        plt.imsave(buf, primary, cmap="gray", vmin=float(vmin), vmax=float(vmax), format="png")
        return buf.getvalue()

    def _window(self, az: int, rg: int) -> tuple[np.ndarray, int, int]:
        run    = self.loaded["run"]
        ph, pw = self.loaded["patch"]
        n_az   = run.split_region.azimuth_size
        n_rg   = run.split_region.range_size

        if not (0 <= az < n_az and 0 <= rg < n_rg):
            raise ValueError(f"pixel ({az}, {rg}) is outside the {n_az}x{n_rg} region")

        top  = int(np.clip(az - ph // 2, 0, n_az - ph))
        left = int(np.clip(rg - pw // 2, 0, n_rg - pw))

        complex_window = run.complex_inputs[:, top:top + ph, left:left + pw]
        dem            = run.dataset.dem
        dem_window     = dem[top:top + ph, left:left + pw] if dem is not None else None

        window = run.dataset.assemble_window(complex_window, dem_window)
        return window[None].astype(np.float32), az - top, rg - left

    def _slots(self, params_center: np.ndarray) -> list[dict]:
        from tools.loss.param_loss import ParamMatcher

        n_k   = self.loaded["run"].n_gaussians
        slots = []
        for k in range(n_k):
            amp = float(params_center[3 * k])
            slots.append({
                "slot"   : k,
                "amp"    : amp,
                "mu"     : float(params_center[3 * k + 1]),
                "sigma"  : float(params_center[3 * k + 2]),
                "active" : bool(amp > ParamMatcher.ACTIVE_AMP_THR),
            })
        return slots

    def _predict_window(self, window: np.ndarray, cy: int, cx: int) -> tuple[list[dict], list[float]]:
        run      = self.loaded["run"]
        renderer = self.loaded["renderer"]

        params = run.model(window)
        curve  = renderer.render(params)[0, :, cy, cx]

        return self._slots(params[0, :, cy, cx]), [float(v) for v in curve]

    def predict(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                az, rg           = int(body["az"]), int(body["rg"])
                window, cy, cx   = self._window(az, rg)
                slots, curve     = self._predict_window(window, cy, cx)

                run      = self.loaded["run"]
                renderer = self.loaded["renderer"]

                gt_curve  = None
                gt_slots  = None
                gt_params = run.dataset.gt_parameters
                if gt_params is not None:
                    center    = np.asarray(gt_params[:, az, rg], dtype=np.float64)
                    gt_slots  = self._slots(center)
                    gt_curve  = [float(v) for v in renderer.render(center.reshape(1, -1, 1, 1))[0, :, 0, 0]]

                raw_curve = [float(v) for v in run.full_curves[:, az, rg]] if run.full_curves is not None else None

                return {
                    "ok"        : True,
                    "az"        : az,
                    "rg"        : rg,
                    "x_axis"    : [float(v) for v in run.x_axis],
                    "slots"     : slots,
                    "curve"     : curve,
                    "gt_slots"  : gt_slots,
                    "gt_curve"  : gt_curve,
                    "raw_curve" : raw_curve,
                }
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def saliency(self, body: dict) -> dict:
        import torch

        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                az, rg = int(body["az"]), int(body["rg"])
                family = str(body.get("family", "mu"))
                if family not in self.FAMILIES:
                    return {"ok": False, "error": f"unknown family '{family}', expected one of {self.FAMILIES}"}

                window, cy, cx = self._window(az, rg)
                offset         = self.FAMILIES.index(family)

                x = torch.from_numpy(window).requires_grad_(True)
                y = self.loaded["run"].model.module(x)

                y[0, offset::3, cy, cx].abs().sum().backward()
                grad = x.grad.abs()[0].numpy()

                shares = grad.sum(axis=(1, 2))
                total  = float(shares.sum())
                if total <= 0.0:
                    return {"ok": False, "error": f"the '{family}' output at this pixel does not depend on the input"}

                spatial = grad.sum(axis=0)
                spatial = spatial / spatial.max()

                return {
                    "ok"       : True,
                    "family"   : family,
                    "channels" : self.loaded["labels"],
                    "shares"   : [float(v / total) for v in shares],
                    "map"      : [[float(v) for v in row] for row in spatial],
                    "center"   : [cy, cx],
                }
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def _perturb(self, window: np.ndarray, perturbation: dict) -> np.ndarray:
        kind      = perturbation.get("kind")
        perturbed = window.copy()

        if kind == "drop_channel":
            channel                  = int(perturbation["channel"])
            perturbed[:, channel]    = 0.0
        elif kind == "scale_channel":
            channel                  = int(perturbation["channel"])
            perturbed[:, channel]   *= float(perturbation.get("factor", 0.5))
        elif kind == "noise":
            sigma = float(perturbation.get("sigma", 0.5))
            rng   = np.random.default_rng(int(perturbation.get("seed", 0)))
            perturbed += rng.normal(0.0, sigma, size=perturbed.shape).astype(np.float32)
        else:
            raise ValueError(f"unknown perturbation '{kind}', expected one of {self.PERTURBATIONS}")

        return perturbed

    def whatif(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                az, rg         = int(body["az"]), int(body["rg"])
                window, cy, cx = self._window(az, rg)

                base_slots, base_curve = self._predict_window(window, cy, cx)

                perturbed                          = self._perturb(window, body.get("perturbation", {}))
                perturbed_slots, perturbed_curve   = self._predict_window(perturbed, cy, cx)

                base      = np.asarray(base_curve)
                shifted   = np.asarray(perturbed_curve)
                delta_mse = float(((base - shifted) ** 2).mean())

                return {
                    "ok"              : True,
                    "az"              : az,
                    "rg"              : rg,
                    "x_axis"          : [float(v) for v in self.loaded["run"].x_axis],
                    "base_slots"      : base_slots,
                    "base_curve"      : base_curve,
                    "perturbed_slots" : perturbed_slots,
                    "perturbed_curve" : perturbed_curve,
                    "delta_mse"       : delta_mse,
                }
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def features_png(self, az: int, rg: int, layer: str, max_channels: int = 16) -> bytes | None:
        import torch

        from tools.diagnostics.activation_recorder import ActivationRecorder

        with self.lock:
            if self.loaded is None or layer not in self.loaded["layers"]:
                return None

            window, _cy, _cx = self._window(az, rg)

            recorder = ActivationRecorder(self.loaded["run"].model.module)
            recorder.attach_store([layer])

            with torch.no_grad():
                self.loaded["run"].model.module(torch.from_numpy(window))

            recorder.detach()
            stored = recorder.stored().get(layer)

        if stored is None or stored.ndim < 4:
            return None

        maps   = stored[0].numpy()
        energy = np.abs(maps).sum(axis=(1, 2)) if maps.ndim == 3 else None
        order  = np.argsort(energy)[::-1][:max_channels] if energy is not None else range(min(max_channels, maps.shape[0]))

        n_cols = 4
        n_rows = int(np.ceil(len(list(order)) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.0))
        axes      = np.atleast_2d(axes)

        for index, channel in enumerate(order):
            ax = axes[index // n_cols][index % n_cols]
            ax.imshow(maps[channel], cmap="magma", aspect="auto")
            ax.set_title(f"ch {int(channel)}", fontsize=7)
            ax.axis("off")

        for index in range(len(list(order)), n_rows * n_cols):
            axes[index // n_cols][index % n_cols].axis("off")

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)

        return buf.getvalue()
