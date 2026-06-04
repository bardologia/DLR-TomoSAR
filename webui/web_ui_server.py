from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config_registry import ConfigRegistry
from equation_library import EquationLibrary
from model_library import ModelLibrary
from pipeline_library import PipelineLibrary
from process_manager import ProcessManager
from project_paths import ProjectPaths
from request_router import RequestRouter
from script_catalog import ScriptCatalog
from script_editor import ScriptEditor
from web_logger import WebLogger


class _Handler(BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self.server.router.route(self)

    def do_POST(self) -> None:
        self.server.router.route(self)

    def log_message(self, fmt: str, *args) -> None:
        return


class WebUIServer:

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host      = host
        self.port      = port
        self.logger    = WebLogger()
        self.paths     = ProjectPaths()

        self.catalog   = ScriptCatalog(self.paths)
        self.editor    = ScriptEditor(self.paths)
        self.configs   = ConfigRegistry(self.paths)
        self.equations = EquationLibrary()
        self.models    = ModelLibrary()
        self.pipelines = PipelineLibrary()
        self.processes = ProcessManager(self.paths, self.logger)

        self.router    = RequestRouter(
            paths     = self.paths,
            logger    = self.logger,
            catalog   = self.catalog,
            editor    = self.editor,
            configs   = self.configs,
            equations = self.equations,
            models    = self.models,
            pipelines = self.pipelines,
            processes = self.processes,
        )

    def serve(self) -> None:
        self.paths.ensure_backups()
        self._report_ready()

        server        = ThreadingHTTPServer((self.host, self.port), _Handler)
        server.router = self.router
        server.daemon_threads = True

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self.logger.warning("shutting down")
        finally:
            server.server_close()

    def _report_ready(self) -> None:
        scripts      = self.catalog.list_scripts()
        interpreters = self.paths.discover_interpreters()
        preferred    = self.paths.preferred_interpreter(interpreters)

        self.logger.banner("DLR-TomoSAR Control Console", [
            f"URL          http://{self.host}:{self.port}",
            f"repo root    {self.paths.repo_root}",
            f"scripts      {len(scripts)} entry points",
            f"interpreter  {preferred}",
            f"envs found   {len(interpreters)}",
        ])
