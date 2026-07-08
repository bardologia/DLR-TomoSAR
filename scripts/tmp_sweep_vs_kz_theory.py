from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.monitoring.logger import Logger
from tools.sar.geometry_field import GeometryField

HARD_THRESHOLD = 4.0


class SweepUnitScan:
    def __init__(self, run_dir: Path):
        self.training_dir = run_dir / "training"

        if not self.training_dir.exists():
            raise FileNotFoundError(f"{self.training_dir} does not exist; pass the sweep run root (the directory holding training/ and report/)")

    def units(self) -> list[dict]:
        records = []

        for unit_dir in sorted(self.training_dir.iterdir()):
            name = unit_dir.name
            if not (name.startswith("n") and "-p" in name and unit_dir.is_dir()):
                continue

            dataset_meta = unit_dir / "meta/dataset_creation_config.json"
            metrics_meta = unit_dir / "meta/test_metrics.json"

            if not dataset_meta.exists():
                raise FileNotFoundError(f"{dataset_meta} missing; cannot recover the unit's actual track selection")

            records.append({
                "name"        : name,
                "n"           : int(name.split("-")[0][1:]),
                "patch"       : int(name.split("-p")[1]),
                "secondaries" : tuple(json.loads(dataset_meta.read_text())["secondary_labels"]),
                "test_loss"   : float(json.loads(metrics_meta.read_text())["avg_loss"]) if metrics_meta.exists() else None,
            })

        if not records:
            raise FileNotFoundError(f"No nNN-pSSS unit directories under {self.training_dir}")

        return records


class KzAperture:
    def __init__(self, field_path: Path):
        if field_path.is_dir():
            field_path = field_path / "meta/geometry_field.npz"

        self.field    = GeometryField.load(field_path)
        self.track_kz = dict(zip(self.field.labels, self.field.kz("height").mean(axis=(1, 2))))

    def moment(self, labels: list) -> float:
        values = np.array([self.track_kz[label] for label in labels])
        return float(((values - values.mean()) ** 2).sum())

    def even_spread(self, n_secondaries: int) -> tuple:
        candidates = sorted((label for label in self.field.labels if label != self.field.reference), key=self.track_kz.get)
        total      = len(candidates)

        if n_secondaries == total:
            return tuple(candidates)
        if n_secondaries == 1:
            return (candidates[(total - 1) // 2],)

        return tuple(candidates[round(position * (total - 1) / (n_secondaries - 1))] for position in range(n_secondaries))


class SweepTheoryComparison:
    def __init__(self, run_dir: Path, field_path: Path, boxcar_override: float | None = None):
        self.logger   = Logger(log_dir="logs", name="sweep_vs_kz_theory")
        self.run_dir  = run_dir
        self.aperture = KzAperture(field_path)
        self.units    = SweepUnitScan(run_dir).units()
        self.boxcar   = boxcar_override if boxcar_override is not None else self._boxcar_from_report()
        self.verdicts = []

    def _boxcar_from_report(self) -> float:
        reports = sorted(self.run_dir.glob("report/*/patch_sweep.json"))

        if not reports:
            raise FileNotFoundError(f"No report/*/patch_sweep.json under {self.run_dir} to read boxcar_window from; pass it as the third argument")

        return float(json.loads(reports[-1].read_text())["boxcar_window"])

    def groups(self) -> dict[int, list[dict]]:
        grouped: dict[int, list[dict]] = {}

        for record in self.units:
            grouped.setdefault(record["n"], []).append(record)

        return {n: sorted(group, key=lambda r: r["patch"]) for n, group in sorted(grouped.items())}

    def selection(self, n: int, group: list[dict]) -> list:
        distinct = {record["secondaries"] for record in group}

        if len(distinct) > 1:
            raise ValueError(f"n={n} units mix {len(distinct)} different track selections {sorted(distinct)}; the run straddles the 2026-07-08 selection fix and must be restarted, not compared")

        secondaries = next(iter(distinct))
        missing     = [label for label in secondaries if label not in self.aperture.track_kz]

        if missing:
            raise KeyError(f"n={n} selection labels {missing} are absent from the geometry field; wrong dataset passed?")

        return [self.aperture.field.reference, *secondaries]

    def bands(self, n: int, labels: list) -> dict:
        total   = len(self.aperture.field.labels)
        ceiling = 2.0 * self.boxcar
        s_full  = self.aperture.moment(self.aperture.field.labels)
        s_sub   = self.aperture.moment(labels)

        floors = {
            "track-count": (self.boxcar * np.sqrt(total / n), total / n),
            "kz-aware"   : (self.boxcar * np.sqrt(s_full / s_sub), s_full / s_sub),
        }

        described = {}
        for name, (floor, ratio) in floors.items():
            hard            = ratio >= HARD_THRESHOLD
            described[name] = {"floor": floor, "low": floor, "high": floor if hard else ceiling, "regime": "hard" if hard else "soft"}

        described["kz-aware"]["kappa"] = float(np.sqrt(s_sub / len(labels)))
        described["kz-aware"]["S"]     = s_sub

        return described

    def distance(self, observed: int, band: dict) -> float:
        if band["low"] <= observed <= band["high"]:
            return 0.0

        return min(abs(observed - band["low"]), abs(observed - band["high"]))

    def describe_group(self, n: int, group: list[dict]) -> None:
        labels   = self.selection(n, group)
        bands    = self.bands(n, labels)
        even     = self.aperture.even_spread(n - 1)
        spread   = "even-spread" if tuple(group[0]["secondaries"]) == even else "NON-even-spread (pre-fix or custom)"
        complete = [record for record in group if record["test_loss"] is not None]

        kz_values = ", ".join(f"{self.aperture.track_kz[label]:.2f}" for label in labels)

        self.logger.info("")
        self.logger.info(f"n={n} | selection {spread}: {', '.join(labels)}")
        self.logger.info(f"n={n} | kz [rad/m]: {kz_values}  S={bands['kz-aware']['S']:.3f}  kappa={bands['kz-aware']['kappa']:.3f}")

        if not complete:
            for record in group:
                self.logger.info(f"n={n} |   p={record['patch']:3d}  test avg_loss = incomplete -- excluded")
            self.logger.warning(f"n={n} | no complete units, no verdict")
            return

        base = complete[0]["test_loss"]
        gain = base - min(record["test_loss"] for record in complete)

        for record in group:
            if record["test_loss"] is None:
                self.logger.info(f"n={n} |   p={record['patch']:3d}  test avg_loss = incomplete -- excluded")
                continue

            captured = (base - record["test_loss"]) / gain if gain > 0 else float("nan")
            self.logger.info(f"n={n} |   p={record['patch']:3d}  test avg_loss = {record['test_loss']:.6f}  captured {captured * 100:5.1f}% of total gain")

        best = min(complete, key=lambda record: record["test_loss"])

        lines = {}
        for name, band in bands.items():
            span        = f"{band['floor']:.1f}" if band["regime"] == "hard" else f"[{band['low']:.1f}, {band['high']:.1f}]"
            lines[name] = self.distance(best["patch"], band)
            self.logger.info(f"n={n} | {name:12s} predicts {span} ({band['regime']}), observed best {best['patch']} -> distance {lines[name]:.1f} px")

        patches   = sorted({record["patch"] for record in group})
        grid_step = int(min(np.diff(patches))) if len(patches) > 1 else 0
        margin    = abs(lines["track-count"] - lines["kz-aware"])
        closer    = "kz-aware" if lines["kz-aware"] < lines["track-count"] else "track-count"

        if margin < 1e-9:
            favours = "indistinguishable (identical predictions)"
        elif margin < grid_step / 2:
            favours = f"weakly favours {closer} (margin {margin:.1f} px, below half the {grid_step} px grid step)"
        else:
            favours = f"favours {closer} by {margin:.1f} px"

        ceiling    = 2.0 * self.boxcar
        admissible = next((patch for patch in patches if patch >= ceiling), None)
        knee       = next((record["patch"] for record in complete if gain > 0 and (base - record["test_loss"]) / gain >= 0.8), None)

        if knee is not None and admissible is not None:
            relation = "matches" if knee == admissible else ("below" if knee < admissible else "above")
            self.logger.info(f"n={n} | ceiling 2w={ceiling:.0f} -> first admissible {admissible}; knee80 = {knee} ({relation})")

        self.verdicts.append({"n": n, "best": best["patch"], "knee": knee, "favours": favours})
        self.logger.info(f"n={n} | VERDICT (floors): {favours}")

    def summary(self) -> None:
        self.logger.info("")
        self.logger.info(f"boxcar w={self.boxcar:.0f}  N={len(self.aperture.field.labels)}  ceiling 2w={2 * self.boxcar:.0f}  kappa_N={np.sqrt(self.aperture.moment(self.aperture.field.labels) / len(self.aperture.field.labels)):.3f}")

        for verdict in self.verdicts:
            knee = verdict["knee"] if verdict["knee"] is not None else "-"
            self.logger.info(f"n={verdict['n']:2d}  argmin={verdict['best']:3d}  knee80={knee:>3}  floors: {verdict['favours']}")

    def run(self) -> None:
        for n, group in self.groups().items():
            self.describe_group(n, group)

        self.summary()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: python scripts/tmp_sweep_vs_kz_theory.py <sweep_run_dir> <dataset_dir_or_geometry_field.npz> [boxcar_window]")

    SweepTheoryComparison(
        run_dir         = Path(sys.argv[1]),
        field_path      = Path(sys.argv[2]),
        boxcar_override = float(sys.argv[3]) if len(sys.argv) > 3 else None,
    ).run()
