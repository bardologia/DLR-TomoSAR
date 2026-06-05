from __future__ import annotations

import _bootstrap

from configuration.physics_check_config import PhysicsCheckEntryConfig
from pipelines.physics_pipeline.check import PhysicsQuantitiesCheck
from tools.config_cli import ConfigCli
from tools.logger import Logger


def main() -> None:
    config = ConfigCli(PhysicsCheckEntryConfig(), description="Physics quantity agreement check: Gaussian fits vs Capon tomogram").apply()
    logger = Logger(log_dir="logs", name="physics_check")

    logger.section("[Physics Quantities Check]")

    check = PhysicsQuantitiesCheck(config, logger)
    check.run()

    logger.close()


if __name__ == "__main__":
    main()
