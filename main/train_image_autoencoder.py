from __future__ import annotations

import argparse

from _bootstrap import EnvironmentPinner


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=0)
    args, _ = parser.parse_known_args()

    EnvironmentPinner.gpu(args.gpu, expandable_segments=True)

    from configuration.training.image_autoencoder    import ImageAeEntryConfig
    from pipelines.image_autoencoder.training.pipeline import SingleTrainRunner
    from pipelines.shared.training_launcher            import SeedSweepLauncher

    SeedSweepLauncher(ImageAeEntryConfig(), SingleTrainRunner, "Image autoencoder training").run()


if __name__ == "__main__":
    main()
