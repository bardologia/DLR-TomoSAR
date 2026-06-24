from __future__ import annotations

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.diagnostics.weight_xray_config import WeightXrayEntryConfig
    from tools.diagnostics.weight_xray               import WeightXray
    from tools.runtime.config_cli                    import ConfigCli
    from tools.monitoring.logger                     import Logger

    entry  = ConfigCli(WeightXrayEntryConfig(), description="X-ray a model checkpoint for dead weights, uniform layers, rank collapse, and other structural pathologies").apply()
    config = entry.to_config()

    logger = Logger(log_dir="logs", name="xray_weights")
    result = WeightXray(config, logger).run()

    logger.info(f"Verdict: {result['summary']['verdict']}")
    logger.close()


if __name__ == "__main__":
    main()
