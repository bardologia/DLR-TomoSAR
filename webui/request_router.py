from __future__ import annotations

import http.client
import json
import math
import mimetypes
import queue
import threading
from pathlib      import Path
from urllib.parse import parse_qs, unquote, urlparse

from config_registry        import ConfigRegistry
from cube_explorer          import CubeExplorer
from dataset_browser        import DatasetBrowser
from equation_library       import EquationLibrary
from flow_library           import FlowLibrary
from gpu_watchdog            import GpuWatchdog
from launch_layout           import LaunchLayout, LayoutError
from backbone_model_library          import BackboneModelLibrary
from image_autoencoder_model_library  import ImageAutoencoderModelLibrary
from pipeline_library       import PipelineLibrary
from repomap_library        import RepoMapLibrary
from profile_autoencoder_model_library import ProfileAutoencoderModelLibrary
from jepa_model_library               import JepaModelLibrary
from notifier               import JobNotifier
from physics_loss_library   import PhysicsLossLibrary
from process_manager        import ProcessManager, ProcessNuke, ServerDetacher
from project_paths          import ProjectPaths
from resource_watchdog      import ResourceWatchdog
from results_browser        import ResultsBrowser
from run_leaderboard        import RunLeaderboard
from script_catalog         import ScriptCatalog
from script_config_resolver import ScriptConfigResolver
from system_monitor         import SystemMonitor
from contention_monitor      import ContentionMonitor
from tensorboard_manager    import TensorboardManager
from training_curves        import TrainingCurves
from web_logger             import WebLogger


class RequestRouter:

    PROJECT = {
        "name"        : "DLR-TomoSAR",
        "tagline"     : "Neural SAR tomography control console",
        "description" : "Supervised deep learning that replaces per-pixel iterative optimisation in SAR tomographic parameter estimation, inferring all 3K Gaussian-mixture parameters of the elevation spectrum in one forward pass.",
        "pipelines"   : ["Processing", "Parameter Extraction", "Dataset", "Training", "Inference", "Tuning"],
    }

    def __init__(self, paths: ProjectPaths, logger: WebLogger, catalog: ScriptCatalog, resolver: ScriptConfigResolver, layout: LaunchLayout, configs: ConfigRegistry, equations: EquationLibrary, physics_loss: PhysicsLossLibrary, flows: FlowLibrary, models: BackboneModelLibrary, profile_ae_models: ProfileAutoencoderModelLibrary, image_ae_models: ImageAutoencoderModelLibrary, jepa_models: JepaModelLibrary, pipelines: PipelineLibrary, repomap: RepoMapLibrary, processes: ProcessManager, notifier: JobNotifier, nuke: ProcessNuke, detacher: ServerDetacher, system: SystemMonitor, watchdog: ResourceWatchdog, contention: ContentionMonitor, gpu_guard: GpuWatchdog, tensorboard: TensorboardManager, results: ResultsBrowser, cubes: CubeExplorer, datasets: DatasetBrowser, leaderboard: RunLeaderboard, curves: TrainingCurves) -> None:
        self.paths       = paths
        self.logger      = logger
        self.catalog     = catalog
        self.resolver    = resolver
        self.layout      = layout
        self.configs     = configs
        self.equations   = equations
        self.physics_loss = physics_loss
        self.flows       = flows
        self.models      = models
        self.profile_ae_models = profile_ae_models
        self.image_ae_models   = image_ae_models
        self.jepa_models       = jepa_models
        self.pipelines   = pipelines
        self.repomap     = repomap
        self.processes   = processes
        self.notifier    = notifier
        self.nuke        = nuke
        self.detacher    = detacher
        self.system      = system
        self.watchdog    = watchdog
        self.contention  = contention
        self.gpu_guard   = gpu_guard
        self.tensorboard = tensorboard
        self.results     = results
        self.cubes       = cubes
        self.datasets    = datasets
        self.leaderboard = leaderboard
        self.curves      = curves

    def _route_get(self, handler, path: str) -> None:
        if path == "/" or path == "":
            self._serve_static(handler, "index.html")
            return
        if path.startswith("/static/"):
            self._serve_static(handler, path[len("/static/"):])
            return
        if path == "/resultsmedia":
            query = parse_qs(urlparse(handler.path).query)
            self._serve_results_media(handler, (query.get("path") or [""])[0])
            return
        if path == "/api/results/tree":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.results.tree((query.get("path") or [""])[0])
            self._send_json(handler, result, 200 if result.get("ok") else 404)
            return
        if path == "/api/results/folder":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.results.folder((query.get("root") or [""])[0], (query.get("rel") or [""])[0])
            self._send_json(handler, result, 200 if result.get("ok") else 404)
            return
        if path == "/api/results/catalog":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.results.catalog((query.get("datasets") or [""])[0], (query.get("logs") or [""])[0])
            self._send_json(handler, result)
            return
        if path == "/api/results/gallery":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.results.gallery((query.get("root") or [""])[0])
            self._send_json(handler, result, 200 if result.get("ok") else 404)
            return
        if path == "/api/fs/datasets":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.datasets.datasets((query.get("base") or [""])[0])
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/fs/runs":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.datasets.runs(query.get("base") or [])
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/fs/run_groups":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.datasets.run_groups(query.get("base") or [])
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/fs/param_trials":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.datasets.param_trials((query.get("base") or [""])[0])
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/fs/params":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.datasets.params((query.get("dataset") or [""])[0])
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/leaderboard":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.leaderboard.table((query.get("base") or [""])[0])
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/leaderboard/trials":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.leaderboard.trials((query.get("base") or [""])[0])
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/curves/runs":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.curves.runs((query.get("base") or [""])[0])
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/curves":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.curves.curves(query.get("run") or [], (query.get("tag") or [""])[0])
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/leaderboard/diff":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.leaderboard.diff(query.get("run") or [])
            self._send_json(handler, result, 200 if result.get("ok") else 404)
            return
        if path == "/api/cubes":
            query = parse_qs(urlparse(handler.path).query)
            self._send_json(handler, self.cubes.list_cubes((query.get("base") or [""])[0]))
            return
        if path == "/api/cubes/status":
            self._send_json(handler, self.cubes.load_status())
            return
        if path == "/api/cubes/primary":
            query = parse_qs(urlparse(handler.path).query)
            png   = self.cubes.primary_png((query.get("id") or [""])[0])
            self._send_png(handler, png)
            return
        if path == "/api/cubes/profiles":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.cubes.profiles(
                cube_id = (query.get("id") or [""])[0],
                az      = int((query.get("az") or ["0"])[0]),
                rg      = int((query.get("rg") or ["0"])[0]),
            )
            self._send_json(handler, result, 200 if result.get("ok") else 404)
            return
        if path == "/api/cubes/ssim":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.cubes.slice_ssim(
                cube_id = (query.get("id") or [""])[0],
                az      = int((query.get("az") or ["0"])[0]),
                rg      = int((query.get("rg") or ["0"])[0]),
                space   = (query.get("space") or ["physical"])[0],
            )
            self._send_json(handler, result, 200 if result.get("ok") else 404)
            return
        if path == "/api/cubes/plane":
            query = parse_qs(urlparse(handler.path).query)
            png   = self.cubes.plane_png(
                cube_id = (query.get("id") or [""])[0],
                source  = (query.get("source") or ["pred"])[0],
                frac    = float((query.get("frac") or ["0"])[0]),
                space   = (query.get("space") or ["physical"])[0],
                cmap    = (query.get("cmap") or ["jet"])[0],
            )
            self._send_png(handler, png)
            return
        if path == "/api/cubes/cbar":
            query = parse_qs(urlparse(handler.path).query)
            self._send_png(handler, self.cubes.cbar_png((query.get("cmap") or ["viridis"])[0]))
            return
        if path == "/api/cubes/metric_map":
            query = parse_qs(urlparse(handler.path).query)
            png   = self.cubes.metric_overlay_png(
                cube_id  = (query.get("id") or [""])[0],
                key      = (query.get("key") or [""])[0],
                vmin     = float((query.get("vmin") or ["0"])[0]),
                vmax     = float((query.get("vmax") or ["0"])[0]),
                keep_min = float((query.get("keep_min") or ["-inf"])[0]),
                keep_max = float((query.get("keep_max") or ["inf"])[0]),
                alpha    = float((query.get("alpha") or ["0.75"])[0]),
            )
            self._send_png(handler, png)
            return
        if path == "/api/cubes/metric_at":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.cubes.metric_value_at(
                cube_id = (query.get("id") or [""])[0],
                key     = (query.get("key") or [""])[0],
                az      = int((query.get("az") or ["0"])[0]),
                rg      = int((query.get("rg") or ["0"])[0]),
            )
            self._send_json(handler, result, 200 if result.get("ok") else 404)
            return
        if path == "/api/cubes/points":
            query = parse_qs(urlparse(handler.path).query)
            blob  = self.cubes.points_bin(
                cube_id    = (query.get("id") or [""])[0],
                source     = (query.get("source") or ["pred"])[0],
                amp_min    = float((query.get("amp_min") or ["0.001"])[0]),
                max_points = int((query.get("max") or ["60000"])[0]),
            )
            self._send_bytes(handler, blob)
            return
        if path == "/api/cubes/dem_points":
            query = parse_qs(urlparse(handler.path).query)
            blob  = self.cubes.dem_points_bin(
                cube_id = (query.get("id") or [""])[0],
                stride  = int((query.get("stride") or ["4"])[0]),
            )
            self._send_bytes(handler, blob)
            return
        if path == "/api/cubes/transect":
            query = parse_qs(urlparse(handler.path).query)
            png   = self.cubes.transect_png(
                cube_id = (query.get("id") or [""])[0],
                source  = (query.get("source") or ["pred"])[0],
                az0     = int((query.get("az0") or ["0"])[0]),
                rg0     = int((query.get("rg0") or ["0"])[0]),
                az1     = int((query.get("az1") or ["0"])[0]),
                rg1     = int((query.get("rg1") or ["0"])[0]),
                space   = (query.get("space") or ["physical"])[0],
                cmap    = (query.get("cmap") or ["jet"])[0],
            )
            self._send_png(handler, png)
            return
        if path == "/api/cubes/param_map":
            query = parse_qs(urlparse(handler.path).query)
            png   = self.cubes.param_map_png(
                cube_id = (query.get("id") or [""])[0],
                source  = (query.get("source") or ["pred"])[0],
                field   = (query.get("field") or ["amp"])[0],
                slot    = int((query.get("slot") or ["0"])[0]),
            )
            self._send_png(handler, png)
            return
        if path == "/api/cubes/param_cbar":
            query = parse_qs(urlparse(handler.path).query)
            png   = self.cubes.param_cbar_png(
                cube_id = (query.get("id") or [""])[0],
                source  = (query.get("source") or ["pred"])[0],
                field   = (query.get("field") or ["amp"])[0],
            )
            self._send_png(handler, png)
            return
        if path == "/api/cubes/params_at":
            query  = parse_qs(urlparse(handler.path).query)
            result = self.cubes.params_at(
                cube_id = (query.get("id") or [""])[0],
                az      = int((query.get("az") or ["0"])[0]),
                rg      = int((query.get("rg") or ["0"])[0]),
            )
            self._send_json(handler, result, 200 if result.get("ok") else 404)
            return
        if path == "/api/cubes/slice":
            query = parse_qs(urlparse(handler.path).query)
            png   = self.cubes.slice_png(
                cube_id = (query.get("id") or [""])[0],
                source  = (query.get("source") or ["pred"])[0],
                axis    = (query.get("axis") or ["range"])[0],
                az      = int((query.get("az") or ["0"])[0]),
                rg      = int((query.get("rg") or ["0"])[0]),
                space   = (query.get("space") or ["physical"])[0],
                cmap    = (query.get("cmap") or ["jet"])[0],
            )
            self._send_png(handler, png)
            return
        if path == "/api/project":
            self._send_json(handler, self._project_payload())
            return
        if path == "/api/equations":
            self._send_json(handler, {"groups": self.equations.collect()})
            return
        if path == "/api/physics-loss":
            self._send_json(handler, self.physics_loss.collect())
            return
        if path == "/api/flows":
            self._send_json(handler, {"flows": self.flows.collect()})
            return
        if path == "/api/backbones":
            self._send_json(handler, {"families": self.models.collect(), "heads": self.models.heads()})
            return
        if path.startswith("/api/backbones/") and path.endswith("/note"):
            key  = path[len("/api/backbones/"):-len("/note")]
            note = self.models.note(key)
            if note is None:
                self._send_json(handler, {"error": "not found"}, 404)
            else:
                self._send_json(handler, note)
            return
        if path == "/api/profile-autoencoders":
            self._send_json(handler, {"families": self.profile_ae_models.collect()})
            return
        if path.startswith("/api/profile-autoencoders/") and path.endswith("/note"):
            key  = path[len("/api/profile-autoencoders/"):-len("/note")]
            note = self.profile_ae_models.note(key)
            if note is None:
                self._send_json(handler, {"error": "not found"}, 404)
            else:
                self._send_json(handler, note)
            return
        if path == "/api/image-autoencoders":
            self._send_json(handler, {"families": self.image_ae_models.collect()})
            return
        if path.startswith("/api/image-autoencoders/") and path.endswith("/note"):
            key  = path[len("/api/image-autoencoders/"):-len("/note")]
            note = self.image_ae_models.note(key)
            if note is None:
                self._send_json(handler, {"error": "not found"}, 404)
            else:
                self._send_json(handler, note)
            return
        if path == "/api/jepa-variants":
            self._send_json(handler, {"families": self.jepa_models.collect()})
            return
        if path.startswith("/api/jepa-variants/") and path.endswith("/note"):
            key  = path[len("/api/jepa-variants/"):-len("/note")]
            note = self.jepa_models.note(key)
            if note is None:
                self._send_json(handler, {"error": "not found"}, 404)
            else:
                self._send_json(handler, note)
            return
        if path == "/api/pipelines":
            self._send_json(handler, {"pipelines": self.pipelines.collect()})
            return
        if path == "/api/repomap":
            self._send_json(handler, {"folders": self.repomap.collect()})
            return
        if path == "/api/scripts":
            self._send_json(handler, {"scripts": self.catalog.list_scripts()})
            return
        if path.startswith("/api/scripts/") and path.endswith("/config"):
            key = path[len("/api/scripts/"):-len("/config")]
            if not self.paths.has_script(key):
                self._send_json(handler, {"error": f"unknown script '{key}'"}, 404)
                return
            result = self.resolver.resolve(key, self._preferred_interpreter(key))
            if result.get("ok"):
                try:
                    result = {**result, "layout": self.layout.build(key, result["leaves"])}
                except LayoutError as exc:
                    result = {"ok": False, "error": str(exc)}
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path.startswith("/api/scripts/"):
            key = path[len("/api/scripts/"):]
            if not self.paths.has_script(key):
                self._send_json(handler, {"error": f"unknown script '{key}'"}, 404)
                return
            detail = self.catalog.get_script(key)
            if detail is None:
                self._send_json(handler, {"error": "not found"}, 404)
            else:
                self._send_json(handler, detail)
            return
        if path == "/api/configs":
            self._send_json(handler, {"groups": self.configs.collect()})
            return
        if path == "/api/jobs":
            self.processes.adopt_orphans()
            self._send_json(handler, {"jobs": self.processes.list_jobs()})
            return
        if path.startswith("/api/jobs/") and path.endswith("/gpus"):
            job_id = path[len("/api/jobs/"):-len("/gpus")]
            result = self.processes.gpu_pool(job_id)
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/tensorboard":
            self._send_json(handler, {"instances": self.tensorboard.list_instances()})
            return
        if path == "/api/tensorboard/logdirs":
            interpreter = self._preferred_interpreter("train_backbone")
            runs_root   = self._runs_root("train_backbone", interpreter)
            if not runs_root:
                self._send_json(handler, {"ok": False, "error": "could not resolve runs root"}, 400)
                return
            result = self.tensorboard.list_logdirs(runs_root)
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/gpu-guard/history":
            query  = parse_qs(urlparse(handler.path).query)
            limit  = int((query.get("limit") or ["100"])[0])
            self._send_json(handler, self.gpu_guard.history(limit))
            return
        if path == "/api/system":
            payload              = self.system.snapshot()
            payload["alerts"]    = self.watchdog.state()
            payload["impact"]    = self.contention.state()
            payload["gpu_guard"] = self.gpu_guard.state()
            payload["server"]    = self.detacher.state()
            payload["notify"]    = self.notifier.state()
            self._send_json(handler, payload)
            return
        if path.startswith("/api/jobs/") and path.endswith("/stream"):
            job_id = path[len("/api/jobs/"):-len("/stream")]
            self._stream_job(handler, job_id)
            return

        self._send_json(handler, {"error": "not found"}, 404)

    def _route_post(self, handler, path: str) -> None:
        body = self._read_json(handler)

        if path == "/api/run":
            key = body.get("script_key", "")
            if not self.paths.has_script(key):
                self._send_json(handler, {"error": f"unknown script '{key}'"}, 404)
                return
            interpreter = body.get("interpreter") or self._preferred_interpreter(key)
            overrides   = body.get("overrides", {})
            follow_up   = body.get("follow_up") or None
            detach      = bool(body.get("detach"))
            queue       = bool(body.get("queue"))

            if queue:
                result = self.processes.enqueue(key, interpreter, overrides, follow_up, detach)
            else:
                result = self.processes.launch(key, interpreter, overrides, follow_up, detach)

            if result.get("ok") and self.tensorboard.logdir_keys(key):
                threading.Thread(target=self._autostart_tensorboard, args=(key, overrides, interpreter), daemon=True).start()

            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path.startswith("/api/jobs/") and path.endswith("/stop"):
            job_id = path[len("/api/jobs/"):-len("/stop")]
            result = self.processes.stop(job_id)
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path.startswith("/api/jobs/") and path.endswith("/gpus"):
            job_id = path[len("/api/jobs/"):-len("/gpus")]
            result = self.processes.set_gpus(job_id, body.get("gpus"), park=bool(body.get("park")))
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/system/nuke":
            self.processes.clear_queue()
            result = self.nuke.nuke()
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/system/detach":
            result = self.detacher.detach()
            self._send_json(handler, result, 200 if result.get("ok") else 500)
            return

        if path == "/api/impact/arm":
            result = self.contention.arm(bool(body.get("armed")))
            self._send_json(handler, result)
            return

        if path == "/api/notify/config":
            result = self.notifier.configure(body or {})
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/notify/test":
            result = self.notifier.test()
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/cubes/load":
            result = self.cubes.start_load(body.get("id", ""))
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/cubes/attach":
            result = self.cubes.attach_second(body.get("id", ""), body.get("other", ""))
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/cubes/detach":
            result = self.cubes.detach_second(body.get("id", ""))
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/cubes/save_transect":
            result = self.cubes.save_transect(
                cube_id = body.get("id", ""),
                az0     = int(body.get("az0", 0)),
                rg0     = int(body.get("rg0", 0)),
                az1     = int(body.get("az1", 0)),
                rg1     = int(body.get("rg1", 0)),
                space   = body.get("space", "physical"),
                cmap    = body.get("cmap", "jet"),
            )
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/cubes/save_slices":
            result = self.cubes.save_slices(
                cube_id = body.get("id", ""),
                az      = int(body.get("az", 0)),
                rg      = int(body.get("rg", 0)),
                space   = body.get("space", "physical"),
                cmap    = body.get("cmap", "jet"),
            )
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/tensorboard/start":
            interpreter = body.get("interpreter") or self._preferred_interpreter(body.get("script_key", "train_backbone"))
            logdir      = body.get("logdir") or self._training_logdir(body.get("script_key", "train_backbone"), {}, interpreter)

            if not logdir:
                self._send_json(handler, {"ok": False, "error": "could not resolve a training log directory"}, 400)
                return

            result = self.tensorboard.ensure(logdir, interpreter)
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path.startswith("/api/tensorboard/") and path.endswith("/stop"):
            tb_id  = path[len("/api/tensorboard/"):-len("/stop")]
            result = self.tensorboard.stop(tb_id)
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        self._send_json(handler, {"error": "not found"}, 404)

    def _autostart_tensorboard(self, key: str, overrides: dict, interpreter: str) -> None:
        try:
            logdir = self._training_logdir(key, overrides, interpreter)
            if logdir:
                self.tensorboard.ensure(logdir, interpreter)
        except Exception as exc:
            self.logger.error(f"tensorboard autostart failed: {exc}")

    def _training_logdir(self, key: str, overrides: dict, interpreter: str) -> str | None:
        leaf_keys = self.tensorboard.logdir_keys(key)
        if not leaf_keys:
            return None

        for leaf in leaf_keys:
            value = (overrides or {}).get(leaf)
            if value:
                return str(value)

        resolved = self.resolver.resolve(key, interpreter)
        if not resolved.get("ok"):
            return None

        leaves = {item["path"]: item["value"] for item in resolved["leaves"]}
        for leaf in leaf_keys:
            if leaves.get(leaf):
                return str(leaves[leaf])

        return None

    def _runs_root(self, key: str, interpreter: str) -> str | None:
        logdir = self._training_logdir(key, {}, interpreter)
        if not logdir:
            return None
        return str(Path(logdir).parent)

    def _proxy_tensorboard(self, handler) -> None:
        segments = handler.path.split("/")
        tb_id    = segments[2].split("?")[0] if len(segments) > 2 else ""
        record   = self.tensorboard.get(tb_id)

        if record is None:
            self._send_json(handler, {"error": "unknown tensorboard instance"}, 404)
            return

        length = int(handler.headers.get("Content-Length", 0) or 0)
        body   = handler.rfile.read(length) if length > 0 else None

        headers = {"Host": f"127.0.0.1:{record['port']}", "Accept-Encoding": "identity"}
        for name in ("Content-Type", "Accept", "X-XSRF-Protected"):
            value = handler.headers.get(name)
            if value:
                headers[name] = value

        connection = http.client.HTTPConnection("127.0.0.1", record["port"], timeout=60)
        try:
            connection.request(handler.command, handler.path, body=body, headers=headers)
            response = connection.getresponse()
            payload  = response.read()
            status   = response.status
            passthru = {name: response.getheader(name) for name in ("Content-Type", "Content-Encoding", "Cache-Control", "Location")}
        except OSError as exc:
            self._send_json(handler, {"error": f"tensorboard unreachable: {exc}"}, 502)
            return
        finally:
            connection.close()

        handler.send_response(status)
        for name, value in passthru.items():
            if value:
                handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def _project_payload(self) -> dict:
        interpreters = self.paths.discover_interpreters()
        model_names  = [model["name"] for family in self.models.collect() for model in family["models"]]
        return {
            **self.PROJECT,
            "models"       : model_names,
            "repo_root"    : str(self.paths.repo_root),
            "interpreters" : interpreters,
            "preferred"    : self.paths.preferred_interpreter(interpreters),
            "counts"       : {
                "scripts"   : len(self.catalog.list_scripts()),
                "models"    : len(model_names),
                "pipelines" : len(self.PROJECT["pipelines"]),
            },
        }

    def _preferred_interpreter(self, script_key: str = "") -> str:
        interpreters = self.paths.discover_interpreters()
        return self.paths.preferred_interpreter(interpreters, script_key)

    def _stream_job(self, handler, job_id: str) -> None:
        stream = self.processes.get_stream(job_id)
        if stream is None:
            self._send_json(handler, {"error": "unknown job"}, 404)
            return

        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()

        sub = stream.subscribe()
        try:
            while True:
                try:
                    event = sub.get(timeout=15)
                except queue.Empty:
                    handler.wfile.write(b": keepalive\n\n")
                    handler.wfile.flush()
                    continue

                payload = json.dumps(event)
                handler.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                handler.wfile.flush()

                if event.get("type") == "end":
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            stream.unsubscribe(sub)

    def _read_json(self, handler) -> dict:
        length = int(handler.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = handler.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    @classmethod
    def _jsonsafe(cls, value):
        if isinstance(value, dict):
            return {key: cls._jsonsafe(child) for key, child in value.items()}
        if isinstance(value, list):
            return [cls._jsonsafe(child) for child in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _send_json(self, handler, obj: dict, status: int = 200) -> None:
        payload = json.dumps(self._jsonsafe(obj)).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(payload)

    def _send_bytes(self, handler, blob: bytes | None) -> None:
        if blob is None:
            self._send_json(handler, {"error": "not found"}, 404)
            return

        handler.send_response(200)
        handler.send_header("Content-Type", "application/octet-stream")
        handler.send_header("Content-Length", str(len(blob)))
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        handler.wfile.write(blob)

    def _send_png(self, handler, png: bytes | None) -> None:
        if png is None:
            self._send_json(handler, {"error": "not found"}, 404)
            return

        handler.send_response(200)
        handler.send_header("Content-Type", "image/png")
        handler.send_header("Content-Length", str(len(png)))
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        handler.wfile.write(png)

    def _serve_results_media(self, handler, raw_path: str) -> None:
        target = self.results.file_path(raw_path)
        if target is None:
            self._send_json(handler, {"error": "not found"}, 404)
            return

        data         = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"

        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "max-age=60")
        handler.end_headers()
        handler.wfile.write(data)

    def _serve_static(self, handler, relative: str) -> None:
        target = (self.paths.static_dir / relative).resolve()
        if not target.is_relative_to(self.paths.static_dir.resolve()):
            self._send_json(handler, {"error": "forbidden"}, 403)
            return
        if not target.is_file():
            self._send_json(handler, {"error": "not found"}, 404)
            return

        data         = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"

        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        handler.wfile.write(data)

    def route(self, handler) -> None:
        parsed = urlparse(handler.path)
        path   = parsed.path.rstrip("/") or "/"
        method = handler.command

        try:
            if parsed.path.startswith("/tb/"):
                self._proxy_tensorboard(handler)
            elif method == "GET":
                self._route_get(handler, path)
            elif method == "POST":
                self._route_post(handler, path)
            else:
                self._send_json(handler, {"error": "method not allowed"}, 405)
        except BrokenPipeError:
            pass
        except Exception as exc:
            self.logger.error(f"router error on {method} {path}: {exc}")
            try:
                self._send_json(handler, {"error": str(exc)}, 500)
            except Exception:
                pass
