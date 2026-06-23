from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config_registry        import ConfigRegistry
from cube_explorer          import CubeExplorer
from dataset_browser        import DatasetBrowser
from equation_library       import EquationLibrary
from flow_library           import FlowLibrary
from gpu_watchdog            import GpuWatchdog
from backbone_model_library          import BackboneModelLibrary
from image_autoencoder_model_library  import ImageAutoencoderModelLibrary
from pipeline_library       import PipelineLibrary
from profile_autoencoder_model_library import ProfileAutoencoderModelLibrary
from jepa_model_library               import JepaModelLibrary
from process_manager        import ProcessManager, ProcessNuke
from project_paths          import ProjectPaths
from request_router         import RequestRouter
from resource_watchdog      import ResourceWatchdog
from results_browser        import ResultsBrowser
from script_catalog         import ScriptCatalog
from script_config_resolver import ScriptConfigResolver
from system_monitor         import SystemMonitor
from tensorboard_manager    import TensorboardManager
from web_logger             import WebLogger


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
        self.host   = host
        self.port   = port
        self.logger = WebLogger()
        self.paths  = ProjectPaths()

        self.resolver    = ScriptConfigResolver(self.paths)
        self.catalog     = ScriptCatalog(self.paths, self.resolver)
        self.configs     = ConfigRegistry(self.paths)
        self.equations   = EquationLibrary()
        self.flows       = FlowLibrary()
        self.models           = BackboneModelLibrary()
        self.profile_ae_models = ProfileAutoencoderModelLibrary()
        self.image_ae_models   = ImageAutoencoderModelLibrary()
        self.jepa_models       = JepaModelLibrary()
        self.pipelines   = PipelineLibrary()
        self.processes   = ProcessManager(self.paths, self.logger)
        self.nuke        = ProcessNuke(self.logger)
        self.system      = SystemMonitor(self.paths)
        self.watchdog    = ResourceWatchdog(self.processes, self.logger)
        self.gpu_guard   = GpuWatchdog(self.system, self.paths, self.logger)
        self.tensorboard = TensorboardManager(self.paths, self.logger)
        self.results     = ResultsBrowser(self.logger)
        self.cubes       = CubeExplorer(self.paths, self.logger)
        self.datasets    = DatasetBrowser(self.logger)

        self.router    = RequestRouter(
            paths       = self.paths,
            logger      = self.logger,
            catalog     = self.catalog,
            resolver    = self.resolver,
            configs     = self.configs,
            equations   = self.equations,
            flows       = self.flows,
            models      = self.models,
            profile_ae_models = self.profile_ae_models,
            image_ae_models   = self.image_ae_models,
            jepa_models       = self.jepa_models,
            pipelines   = self.pipelines,
            processes   = self.processes,
            nuke        = self.nuke,
            system      = self.system,
            watchdog    = self.watchdog,
            gpu_guard   = self.gpu_guard,
            tensorboard = self.tensorboard,
            results     = self.results,
            cubes       = self.cubes,
            datasets    = self.datasets,
        )

    def serve(self) -> None:
        self._report_ready()
        self.watchdog.start()
        self.gpu_guard.start()

        server        = ThreadingHTTPServer((self.host, self.port), _Handler)
        server.router = self.router
        server.daemon_threads = True

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self.logger.warning("shutting down")
        finally:
            self.tensorboard.stop_all()
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
