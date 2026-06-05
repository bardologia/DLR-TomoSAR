from __future__ import annotations

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.gpu(expandable_segments=True)

    from configuration.single_train_config import SingleTrainConfig
    from pipelines.training_pipeline.pipeline import SingleTrainRunner
    from tools.config_cli import ConfigCli

    config = ConfigCli(SingleTrainConfig(), description="Single training run").apply()
    SingleTrainRunner(config).run()


if __name__ == "__main__":
    main()
