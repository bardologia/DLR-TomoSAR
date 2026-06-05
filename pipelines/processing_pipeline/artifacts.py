from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal, Tuple

from configuration.processing_config import ProcessingConfiguration, TomogramConfiguration
from pipelines.shared.io             import FileIO
from tools.logger                    import Logger


ArtifactType = Literal["tomogram_full", "tomogram_reduced", "dem_full", "dem_reduced", "primary_reduced", "secondaries_reduced", "interferograms_reduced"]


class ArtifactRegistry:
    def __init__(self, config: ProcessingConfiguration, logger: Logger) -> None:
        self.config = config
        self.logger = logger

    def ensure_directory_structure(self) -> None:
        paths = self.config.paths
        self.logger.section("[Directory Validation]")
        for directory in (paths.data_directory, paths.metadata_directory, paths.temporary_directory):
            target_dir = directory.resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            self.logger.subsection(f"Ensured path : {target_dir}")

    def artifact_filenames(self) -> dict[str, str]:
        tomo_tag  = self.config.tomogram_tag
        param_tag = self.config.parameter_tag

        return {
            "tomogram_full"          : f"tomogram_full_{param_tag}.npy",
            "tomogram_reduced"       : f"tomogram_reduced_{tomo_tag}.npy",
            "dem_full"               : f"dem_full_{param_tag}.npy",
            "dem_reduced"            : f"dem_reduced_{tomo_tag}.npy",
            "primary_reduced"        : f"primary_reduced_{tomo_tag}.npy",
            "secondaries_reduced"    : f"secondaries_reduced_{tomo_tag}.npy",
            "interferograms_reduced" : f"interferograms_reduced_{tomo_tag}.npy",
        }

    def artifact_path(self, artifact_type: ArtifactType) -> Path:
        filenames = self.artifact_filenames()
        return self.config.paths.data_directory / filenames[artifact_type]


class MetadataManager:
    def __init__(self, config: ProcessingConfiguration, logger: Logger) -> None:
        self.config   = config
        self.logger   = logger
        self.registry = ArtifactRegistry(config, logger)

        self.logger.section("[MetadataManager Initialization]")
        self.logger.subsection(f"Run Directory : {config.paths.run_directory}")
        self.logger.subsection(f"Tomogram Tag  : {config.tomogram_tag}")
        self.logger.subsection(f"Parameter Tag : {config.parameter_tag}")

    def build_tomogram_metadata(self, variant: str, output_path: Path, stack_identifier: str, cfg: TomogramConfiguration) -> dict[str, str]:
        return {
            f"tomo_{variant}" : str(output_path),
            "crop"         : f"[{', '.join(str(v) for v in self.config.crop.as_tuple())}]",
            "FuSARproject" : cfg.fusar_project_path,
            "id"           : stack_identifier,
            "basedir"      : cfg.base_directory,
            "polarisation" : cfg.polarisation,
            "select"       : cfg.track_selection,
            "range"        : f"[{', '.join(str(v) for v in cfg.height_range)}]",
            "filter"       : cfg.filter_method,
            "method"       : cfg.beamforming_method,
            "win"          : f"[{', '.join(str(v) for v in cfg.filter_arguments.get('win', []))}]",
        }

    def build_inputs_metadata(self, primary_path: Path, secondaries_path: Path, interferograms_path: Path, primary_shape: Tuple[int, ...], secondaries_shape: Tuple[int, ...], interferograms_shape: Tuple[int, ...]) -> dict[str, str]:
        cfg = self.config.input_configs
        return {
            "primary_path"         : str(primary_path),
            "secondaries_path"     : str(secondaries_path),
            "interferograms_path"  : str(interferograms_path),
            "primary_shape"        : f"[{', '.join(str(v) for v in primary_shape)}]",
            "secondaries_shape"    : f"[{', '.join(str(v) for v in secondaries_shape)}]",
            "interferograms_shape" : f"[{', '.join(str(v) for v in interferograms_shape)}]",
            "crop"                 : f"[{', '.join(str(v) for v in self.config.crop.as_tuple())}]",
            "FuSARproject"         : cfg.fusar_project_path,
            "id"                   : self.config.reduced_stack_identifier,
            "basedir"              : cfg.base_directory,
            "polarisation"         : cfg.polarisation,
            "select"               : cfg.track_selection,
            "data_type"            : self.config.dataset_type,
        }

    def save_stage_metadata(self, stage_name: str, identifier_tag: str, metadata_entries: dict[str, str]) -> Path:
        self.registry.ensure_directory_structure()
        meta_filename = f"meta_{stage_name}_{identifier_tag}.txt"
        meta_path     = self.config.paths.metadata_directory / meta_filename

        self.logger.section(f"[Saving Metadata] Stage: {stage_name} | Tag: {identifier_tag}")
        FileIO.save_text_metadata(metadata_entries, meta_path)

        self.logger.subsection(f"Metadata written: {meta_path}")
        return meta_path

    def save_pipeline_configuration(self) -> Path:
        self.registry.ensure_directory_structure()

        run_tag       = self.config.tomogram_tag
        dump_filename = f"config_state_{run_tag}.json"
        dump_path     = self.config.paths.metadata_directory / dump_filename

        config_dict = asdict(self.config)

        self.logger.section("[Saving Configuration State]")
        FileIO.save_json(config_dict, dump_path)

        self.logger.subsection(f"Configuration preserved at: {dump_path}")
        return dump_path

    def save_dataset_layout(self) -> Path:
        self.registry.ensure_directory_structure()

        layout = {
            "global_crop"   : list(self.config.crop.as_tuple()),
            "dataset_type"  : self.config.dataset_type,
            "tomogram_tag"  : self.config.tomogram_tag,
            "parameter_tag" : self.config.parameter_tag,
            "artifacts"     : self.registry.artifact_filenames(),
        }

        out_path = self.config.paths.data_directory / "dataset.json"
        FileIO.save_json(layout, out_path, indent=2)

        self.logger.section(f"[Dataset Layout Saved] {out_path}")
        return out_path
