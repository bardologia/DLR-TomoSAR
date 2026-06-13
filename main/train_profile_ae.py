from __future__ import annotations

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.gpu(expandable_segments=True)

    from configuration.jepa_config         import ProfileAeEntryConfig
    from pipelines.jepa_pipeline.pipeline   import SingleProfileAeRunner
    from tools.config_cli                   import ConfigCli

    config = ConfigCli(ProfileAeEntryConfig(), description="Stage-A profile autoencoder training").apply()
    SingleProfileAeRunner(config).run()


if __name__ == "__main__":
    main()
