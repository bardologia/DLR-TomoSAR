from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.baselines       import TrackProfiles
from tools.data.regions    import CropRegion
from tools.sar             import GeometryField, GeometryFieldBuilder, TrackParameters


class GeometryFieldBackfiller:
    def __init__(self, run_directory, overwrite=False, dry_run=False) -> None:
        self.run_directory  = Path(run_directory)
        self.meta_directory = self.run_directory / "meta"
        self.data_directory = self.run_directory / "data"
        self.overwrite      = overwrite
        self.dry_run        = dry_run

    def _load_parameters(self) -> TrackParameters:
        path = self.meta_directory / TrackParameters.FILENAME

        if not path.is_file():
            raise FileNotFoundError(f"No {TrackParameters.FILENAME} under {self.meta_directory}; backfill track parameters first.")

        return TrackParameters.load(path)

    def _load_profiles(self) -> TrackProfiles:
        path = self.data_directory / TrackProfiles.FILENAME

        if not path.is_file():
            raise FileNotFoundError(f"No {TrackProfiles.FILENAME} under {self.data_directory}; the dataset has no per-azimuth track profiles.")

        return TrackProfiles.load(path)

    def _load_crop(self) -> CropRegion:
        path = self.data_directory / "dataset.json"

        if not path.is_file():
            raise FileNotFoundError(f"No dataset.json under {self.data_directory}; cannot recover the global crop.")

        global_crop = json.loads(path.read_text(encoding="utf-8"))["global_crop"]

        return CropRegion(azimuth_start=global_crop[0], azimuth_end=global_crop[1], range_start=global_crop[2], range_end=global_crop[3])

    def _build(self, parameters: TrackParameters, profiles: TrackProfiles, crop: CropRegion) -> GeometryField:
        return GeometryFieldBuilder(parameters, profiles, crop).build()

    def run(self) -> bool:
        out_path = self.meta_directory / GeometryField.FILENAME

        if out_path.is_file() and not self.overwrite:
            print(f"skip     {out_path} already present (use --overwrite to regenerate).")
            return True

        parameters = self._load_parameters()
        profiles   = self._load_profiles()
        crop       = self._load_crop()
        field      = self._build(parameters, profiles, crop)

        for line, value in field.describe().items():
            print(f"  {line:20s}: {value}")

        if self.dry_run:
            print(f"dry-run  would write {out_path}")
            return True

        field.save(out_path)
        print(f"wrote    {out_path}  ({field.n_tracks} tracks, {field.n_azimuth} azimuth x {field.n_range} range)")

        return True


class BackfillBatch:
    def __init__(self, roots, overwrite=False, dry_run=False) -> None:
        self.roots     = [Path(root) for root in roots]
        self.overwrite = overwrite
        self.dry_run   = dry_run

    def _discover_runs(self, root: Path) -> list:
        if (root / "meta" / TrackParameters.FILENAME).is_file():
            return [root]

        return sorted({match.parent.parent for match in root.glob(f"**/meta/{TrackParameters.FILENAME}")})

    def run(self) -> int:
        runs = [run for root in self.roots for run in self._discover_runs(root)]

        if not runs:
            print("No datasets with meta/track_parameters.json found under the given paths.")
            return 1

        failures = 0

        for run in runs:
            print(f"\n[{run}]")
            try:
                GeometryFieldBackfiller(run, overwrite=self.overwrite, dry_run=self.dry_run).run()
            except Exception as error:
                failures += 1
                print(f"FAIL     {type(error).__name__}: {error}")

        print(f"\nDone: {len(runs) - failures}/{len(runs)} datasets backfilled, {failures} failed.")

        return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build meta/geometry_field.npz (per-range slant range and look angle, per-azimuth baselines) from meta/track_parameters.json, data/track_profiles.npz, and the dataset global crop, so the physics loss can assemble a per-pixel vertical wavenumber.")
    parser.add_argument("paths", type=str, nargs="+", help="Dataset run directories, or a parent directory to scan recursively for meta/track_parameters.json.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate geometry_field.npz even if it already exists.")
    parser.add_argument("--dry-run",   action="store_true", help="Build and report without writing.")
    args = parser.parse_args()

    exit_code = BackfillBatch(
        roots     = args.paths,
        overwrite = args.overwrite,
        dry_run   = args.dry_run,
    ).run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
