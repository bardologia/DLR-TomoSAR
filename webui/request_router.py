from __future__ import annotations

import http.client
import json
import mimetypes
import queue
import threading
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from config_registry import ConfigRegistry
from cube_explorer import CubeExplorer
from equation_library import EquationLibrary
from flow_library import FlowLibrary
from model_library import ModelLibrary
from pipeline_library import PipelineLibrary
from process_manager import ProcessManager
from project_paths import ProjectPaths
from resource_watchdog import ResourceWatchdog
from results_browser import ResultsBrowser
from script_catalog import ScriptCatalog
from script_config_resolver import ScriptConfigResolver
from system_monitor import SystemMonitor
from tensorboard_manager import TensorboardManager
from web_logger import WebLogger


class RequestRouter:

    PROJECT = {
        "name"        : "DLR-TomoSAR",
        "tagline"     : "Neural SAR tomography control console",
        "description" : "Supervised deep learning that replaces per-pixel iterative optimisation in SAR tomographic parameter estimation, inferring all 3K Gaussian-mixture parameters of the elevation spectrum in one forward pass.",
        "models"      : ["UNet", "ResUNet", "Attention UNet", "UNet++", "FCN", "LinkNet", "Swin-UNet", "TransUNet", "UNETR"],
        "pipelines"   : ["Processing", "Parameter Extraction", "Dataset", "Training", "Inference", "Tuning"],
    }

    def __init__(self, paths: ProjectPaths, logger: WebLogger, catalog: ScriptCatalog, resolver: ScriptConfigResolver, configs: ConfigRegistry, equations: EquationLibrary, flows: FlowLibrary, models: ModelLibrary, pipelines: PipelineLibrary, processes: ProcessManager, system: SystemMonitor, watchdog: ResourceWatchdog, tensorboard: TensorboardManager, results: ResultsBrowser, cubes: CubeExplorer) -> None:
        self.paths       = paths
        self.logger      = logger
        self.catalog     = catalog
        self.resolver    = resolver
        self.configs     = configs
        self.equations   = equations
        self.flows       = flows
        self.models      = models
        self.pipelines   = pipelines
        self.processes   = processes
        self.system      = system
        self.watchdog    = watchdog
        self.tensorboard = tensorboard
        self.results     = results
        self.cubes       = cubes

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
        if path == "/api/cubes":
            self._send_json(handler, {"cubes": self.cubes.list_cubes()})
            return
        if path == "/api/cubes/status":
            self._send_json(handler, self.cubes.load_status())
            return
        if path == "/api/cubes/topdown":
            query = parse_qs(urlparse(handler.path).query)
            png   = self.cubes.topdown_png(
                cube_id = (query.get("id") or [""])[0],
                source  = (query.get("source") or ["pred"])[0],
            )
            self._send_png(handler, png)
            return
        if path == "/api/cubes/slice":
            query = parse_qs(urlparse(handler.path).query)
            png   = self.cubes.slice_png(
                cube_id = (query.get("id") or [""])[0],
                source  = (query.get("source") or ["pred"])[0],
                axis    = (query.get("axis") or ["range"])[0],
                az      = int((query.get("az") or ["0"])[0]),
                rg      = int((query.get("rg") or ["0"])[0]),
            )
            self._send_png(handler, png)
            return
        if path == "/api/project":
            self._send_json(handler, self._project_payload())
            return
        if path == "/api/equations":
            self._send_json(handler, {"groups": self.equations.collect()})
            return
        if path == "/api/flows":
            self._send_json(handler, {"flows": self.flows.collect()})
            return
        if path == "/api/models":
            self._send_json(handler, {"families": self.models.collect()})
            return
        if path.startswith("/api/models/") and path.endswith("/note"):
            key  = path[len("/api/models/"):-len("/note")]
            note = self.models.note(key)
            if note is None:
                self._send_json(handler, {"error": "not found"}, 404)
            else:
                self._send_json(handler, note)
            return
        if path == "/api/pipelines":
            self._send_json(handler, {"pipelines": self.pipelines.collect()})
            return
        if path == "/api/scripts":
            self._send_json(handler, {"scripts": self.catalog.list_scripts()})
            return
        if path.startswith("/api/scripts/") and path.endswith("/config"):
            key    = path[len("/api/scripts/"):-len("/config")]
            result = self.resolver.resolve(key, self._preferred_interpreter(key))
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path.startswith("/api/scripts/"):
            key    = path[len("/api/scripts/"):]
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
            self._send_json(handler, {"jobs": self.processes.list_jobs()})
            return
        if path == "/api/tensorboard":
            self._send_json(handler, {"instances": self.tensorboard.list_instances()})
            return
        if path == "/api/system":
            payload           = self.system.snapshot()
            payload["alerts"] = self.watchdog.state()
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
            key         = body.get("script_key", "")
            interpreter = body.get("interpreter") or self._preferred_interpreter(key)
            overrides   = body.get("overrides", {})
            follow_up   = body.get("follow_up") or None
            detach      = bool(body.get("detach"))
            result      = self.processes.launch(key, interpreter, overrides, follow_up, detach)

            if result.get("ok") and self.tensorboard.logdir_keys(key):
                threading.Thread(target=self._autostart_tensorboard, args=(key, overrides, interpreter), daemon=True).start()

            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path.startswith("/api/jobs/") and path.endswith("/stop"):
            job_id = path[len("/api/jobs/"):-len("/stop")]
            result = self.processes.stop(job_id)
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/cubes/load":
            result = self.cubes.start_load(body.get("id", ""))
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/tensorboard/start":
            interpreter = body.get("interpreter") or self._preferred_interpreter(body.get("script_key", "train"))
            logdir      = body.get("logdir") or self._training_logdir(body.get("script_key", "train"), {}, interpreter)

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
        return {
            **self.PROJECT,
            "repo_root"    : str(self.paths.repo_root),
            "interpreters" : interpreters,
            "preferred"    : self.paths.preferred_interpreter(interpreters),
            "counts"       : {
                "scripts"   : len(self.catalog.list_scripts()),
                "models"    : len(self.PROJECT["models"]),
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

    def _send_json(self, handler, obj: dict, status: int = 200) -> None:
        payload = json.dumps(obj).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(payload)

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
        if not str(target).startswith(str(self.paths.static_dir.resolve())):
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
