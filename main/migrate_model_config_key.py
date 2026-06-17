from __future__ import annotations

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.maintenance          import ModelConfigMigrationConfig
    from tools.data.model_config_migration  import ModelConfigKeyMigrator
    from tools.runtime.config_cli           import ConfigCli
    from tools.monitoring.logger            import Logger

    config = ConfigCli(ModelConfigMigrationConfig(), description="Rename the persisted model-config name key from 'model_name' to 'backbone_name' for backbone runs trained before the backbone-naming refactor").apply()

    with Logger(log_dir=str(config.runs_dir), name="model_config_migration", level=config.log_level) as logger:
        logger.section("Model config key migration")

        migrator = ModelConfigKeyMigrator(runs_dir=config.runs_dir, dry_run=config.dry_run, logger=logger)
        migrator.run()


if __name__ == "__main__":
    main()
