from __future__ import annotations

import threading

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from command_listener                   import CommandListener
from config_registry                    import ConfigRegistry
from cube_explorer                      import CubeExplorer
from dataset_browser                    import DatasetBrowser
from equation_library                   import EquationLibrary
from flow_library                       import FlowLibrary
from gpu_schedule                       import GpuSchedule
from gpu_watchdog                       import GpuWatchdog
from job_describer                      import JobDescriber
from launch_layout                      import LaunchLayout
from backbone_model_library             import BackboneModelLibrary
from image_autoencoder_model_library    import ImageAutoencoderModelLibrary
from physics_loss_library               import PhysicsLossLibrary
from pipeline_library                   import PipelineLibrary
from repomap_library                    import RepoMapLibrary
from profile_autoencoder_model_library  import ProfileAutoencoderModelLibrary
from jepa_model_library                 import JepaModelLibrary
from notifier                           import JobNotifier
from process_manager                    import ProcessManager, ProcessNuke, ServerDetacher
from project_paths                      import ProjectPaths
from request_router                     import RequestRouter
from resource_watchdog                  import ResourceWatchdog
from contention_monitor                 import ContentionMonitor
from results_browser                    import ResultsBrowser
from run_leaderboard                    import RunLeaderboard
from script_catalog                     import ScriptCatalog
from script_config_resolver             import ScriptConfigResolver
from system_monitor                     import SystemMonitor
from telegram_bot                       import TelegramBot
from tensorboard_manager                import TensorboardManager
from training_curves                    import TrainingCurves
from web_logger                         import WebLogger


class _Server(ThreadingHTTPServer):

    request_queue_size = 64
    daemon_threads     = True


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

        self.resolver          = ScriptConfigResolver(self.paths)
        self.catalog           = ScriptCatalog(self.paths, self.resolver)
        self.layout            = LaunchLayout()
        self.configs           = ConfigRegistry(self.paths)
        self.equations         = EquationLibrary()
        self.physics_loss      = PhysicsLossLibrary()
        self.flows             = FlowLibrary()
        self.models            = BackboneModelLibrary()
        self.profile_ae_models = ProfileAutoencoderModelLibrary()
        self.image_ae_models   = ImageAutoencoderModelLibrary()
        self.jepa_models       = JepaModelLibrary()
        self.pipelines         = PipelineLibrary()
        self.repomap           = RepoMapLibrary()
        self.telegram          = TelegramBot(self.paths, self.logger)
        self.notifier          = JobNotifier(self.telegram, self.logger)
        self.describer         = JobDescriber(self.paths, self.resolver)
        self.processes         = ProcessManager(self.paths, self.logger, self.notifier, self.describer)
        self.nuke              = ProcessNuke(self.logger)
        self.detacher          = ServerDetacher(self.paths, self.logger)
        self.system            = SystemMonitor(self.paths)
        self.commands          = CommandListener(self.telegram, self.logger, self.notifier, self.processes, self.nuke, self.system)
        self.watchdog          = ResourceWatchdog(self.processes, self.logger)
        self.contention        = ContentionMonitor(self.paths, self.logger, self.nuke)
        self.gpu_guard         = GpuWatchdog(self.system, self.paths, self.logger, self.processes)
        self.gpu_schedule      = GpuSchedule(self.paths, self.logger, self.processes, self.system)
        self.tensorboard       = TensorboardManager(self.paths, self.logger)
        self.results           = ResultsBrowser(self.logger)
        self.cubes             = CubeExplorer(self.paths, self.logger)
        self.datasets          = DatasetBrowser(self.logger)
        self.leaderboard       = RunLeaderboard(self.logger)
        self.curves            = TrainingCurves(self.logger)

        self.router    = RequestRouter(
            paths             = self.paths,
            logger            = self.logger,
            catalog           = self.catalog,
            resolver          = self.resolver,
            layout            = self.layout,
            configs           = self.configs,
            equations         = self.equations,
            physics_loss      = self.physics_loss,
            flows             = self.flows,
            models            = self.models,
            profile_ae_models = self.profile_ae_models,
            image_ae_models   = self.image_ae_models,
            jepa_models       = self.jepa_models,
            pipelines         = self.pipelines,
            repomap           = self.repomap,
            processes         = self.processes,
            telegram          = self.telegram,
            commands          = self.commands,
            nuke              = self.nuke,
            detacher          = self.detacher,
            system            = self.system,
            watchdog          = self.watchdog,
            contention        = self.contention,
            gpu_guard         = self.gpu_guard,
            gpu_schedule      = self.gpu_schedule,
            tensorboard       = self.tensorboard,
            results           = self.results,
            cubes             = self.cubes,
            datasets          = self.datasets,
            leaderboard       = self.leaderboard,
            curves            = self.curves,
        )

    def serve(self) -> None:
        self._report_ready()
        self.watchdog.start()
        self.contention.start()
        self.gpu_guard.start()
        self.gpu_schedule.start()
        self.commands.start()

        server        = _Server((self.host, self.port), _Handler)
        server.router = self.router

        worker = threading.Thread(target=server.serve_forever, name="HttpServer", daemon=True)
        worker.start()

        try:
            self.detacher.wait_loop()
        except KeyboardInterrupt:
            self.logger.warning("shutting down")
        finally:
            self.tensorboard.stop_all()
            server.shutdown()
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
