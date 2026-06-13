from __future__ import annotations

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.gpu(expandable_segments=True)

    from configuration.jepa_config        import JepaEntryConfig
    from pipelines.jepa_pipeline.pipeline  import SingleJepaRunner
    from tools.config_cli                  import ConfigCli

    config = ConfigCli(JepaEntryConfig(), description="Stage-B JEPA predictor training").apply()
    SingleJepaRunner(config).run()


if __name__ == "__main__":
    main()
