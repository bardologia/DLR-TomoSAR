from __future__ import annotations

import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing  import Tuple

from configuration.processing_config             import ProcessingConfiguration
from pipelines.processing_pipeline.artifacts     import ArtifactRegistry, MetadataManager
from pipelines.processing_pipeline.interferogram import InterferogramBuilder
from pipelines.processing_pipeline.tomogram      import TomogramProcessor
from tools.logger                                import Logger


class ProcessingPipeline:
    def __init__(self, config: ProcessingConfiguration, logger: Logger | None = None) -> None:
        self.config = config
        run_dir     = Path(config.paths.run_directory)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir     = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger                = logger or Logger(log_dir = str(log_dir), name = "preprocessing", level = "INFO")
        self.artifact_registry     = ArtifactRegistry (config,   logger=self.logger)
        self.metadata_manager      = MetadataManager  (config,    logger=self.logger)
        self.tomogram_processor    = TomogramProcessor (config,   logger=self.logger)
        self.interferogram_builder = InterferogramBuilder(config, logger=self.logger)

        self.logger.section("[Pre-Processing Pipeline Initialized]")
        self.logger.subsection(f"Stack ID : {config.stack_identifier}")

    def _stage_tomogram(self) -> Tuple[Path, Path]:
        tomogram_path = self.artifact_registry.artifact_path("tomogram_full")
        dem_path      = self.artifact_registry.artifact_path("dem_full")

        self.logger.subsection("[Active] Generating full-stack tomogram")
        self.tomogram_processor.run(
            tomogram_path    = tomogram_path,
            dem_path         = dem_path,
            stack_identifier = self.config.stack_identifier,
            tomogram_config  = self.config.tomogram_config,
        )

        self.metadata_manager.save_stage_metadata(
            stage_name       = "tomogram_full",
            identifier_tag   = self.config.parameter_tag,
            metadata_entries = self.metadata_manager.build_tomogram_metadata(tomogram_path, self.config.stack_identifier, self.config.tomogram_config),
        )

        gc.collect()

        return tomogram_path, dem_path

    def _stage_inputs(self) -> Tuple[Path, Path, Path]:
        primary_path        = self.artifact_registry.artifact_path("primary")
        secondaries_path    = self.artifact_registry.artifact_path("secondaries")
        interferograms_path = self.artifact_registry.artifact_path("interferograms")

        self.logger.subsection("[Active] Building interferometric stack")
        primary_shape, secondaries_shape, interferograms_shape = self.interferogram_builder.run(
            crop_tuple          = self.config.crop.as_tuple(),
            primary_path        = primary_path,
            secondaries_path    = secondaries_path,
            interferograms_path = interferograms_path,
        )

        self.metadata_manager.save_stage_metadata(
            stage_name       = "inputs",
            identifier_tag   = self.config.tomogram_tag,
            metadata_entries = self.metadata_manager.build_inputs_metadata(primary_path, secondaries_path, interferograms_path, primary_shape, secondaries_shape, interferograms_shape),
        )

        if self.interferogram_builder.track_baselines is not None:
            self.metadata_manager.save_baselines(self.interferogram_builder.track_baselines)

        if self.interferogram_builder.track_profiles is not None:
            self.metadata_manager.save_track_profiles(self.interferogram_builder.track_profiles)

        gc.collect()

        return primary_path, secondaries_path, interferograms_path

    def run(self) -> dict[str, Path]:
        self.logger.section("[Pre-Processing Pipeline Execution]")

        self.metadata_manager.save_pipeline_configuration()

        full_tomo_path, full_dem_path = self._stage_tomogram()

        primary_path, secondaries_path, interferograms_path = self._stage_inputs()

        baselines   = self.interferogram_builder.track_baselines
        pass_labels = list(baselines.labels) if baselines is not None else None
        self.metadata_manager.save_dataset_layout(pass_labels=pass_labels)

        self.logger.section("[Pre-Processing Execution Completed]")

        return {
            "tomogram_full"  : full_tomo_path,
            "dem_full"       : full_dem_path,
            "primary"        : primary_path,
            "secondaries"    : secondaries_path,
            "interferograms" : interferograms_path,
            "run_directory"  : self.config.paths.run_directory,
        }


class PreProcessSession:
    def __init__(self, index: int, total: int, dataset_name: str, config: ProcessingConfiguration) -> None:
        self.index        = index
        self.total        = total
        self.dataset_name = dataset_name
        self.config       = config

    def execute(self) -> dict[str, Path]:
        return ProcessingPipeline(self.config).run()


class PreProcessScheduler:
    def __init__(self, sessions: list[PreProcessSession], max_sessions: int, logger: Logger) -> None:
        self.sessions     = sessions
        self.max_sessions = max_sessions
        self.logger       = logger

    def run(self) -> dict[str, dict[str, Path]]:
        slots   = max(1, min(self.max_sessions, len(self.sessions)))
        results = {}

        self.logger.subsection(f"Dispatching {len(self.sessions)} sessions across {slots} concurrent slots")

        with ProcessPoolExecutor(max_workers=slots, mp_context=mp.get_context("spawn")) as executor:
            futures = {executor.submit(session.execute): session for session in self.sessions}

            try:
                for future in as_completed(futures):
                    session = futures[future]
                    outputs = future.result()

                    results[session.dataset_name] = outputs

                    self.logger.section(f"[Session {session.index + 1}/{session.total}] {session.dataset_name} completed")
                    self.logger.kv_table({name: str(path) for name, path in outputs.items()}, title="Outputs")

            except Exception:
                for future in futures:
                    future.cancel()
                raise

        return results


