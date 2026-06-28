from __future__ import annotations

from datetime import datetime
from pathlib  import Path

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.comparison                                import ParamExtractionComparisonConfig
    from pipelines.comparison.param_extraction_comparison        import ParamExtractionComparisonPipeline
    from tools.runtime.config_cli                                import ConfigCli
    from tools.monitoring.logger                                 import Logger

    config = ConfigCli(ParamExtractionComparisonConfig(), description="Compare Gaussian-fit parameter-extraction trials, grouped by number of Gaussians").apply()

    params_dir = Path(config.params_dir)
    base_out   = Path(config.output_dir) if config.output_dir else params_dir / "_fit_comparison"
    out_dir    = base_out / datetime.now().strftime("%Y%m%d_%H%M%S")

    with Logger(log_dir=str(out_dir / "logs"), name="compare_param_extraction_trials") as logger:
        logger.section("Parameter-extraction comparison")
        logger.kv_table({
            "Params dir"   : str(params_dir),
            "Trials"       : len(config.run_tags) if config.run_tags else "auto",
            "Pixel sample" : config.pixel_sample,
            "Block size"   : config.block_size,
            "Output dir"   : str(out_dir),
        }, title="Configuration")

        report = ParamExtractionComparisonPipeline(config=config, out_dir=out_dir, logger=logger).run()

        logger.info(f"\nComparison written to: {report.parent}")


if __name__ == "__main__":
    main()
