from __future__ import annotations

import sys
from pathlib import Path
from typing  import List


class CondaEnv:
    @staticmethod
    def _candidates(env_name: str) -> List[Path]:
        executable = Path(sys.executable).resolve()
        candidates : List[Path] = []

        for parent in executable.parents:
            if parent.name == "envs":
                candidates.append(parent / env_name / "bin" / "python")
                break

        for base in (Path.home() / "miniconda3", Path.home() / "anaconda3", Path.home() / ".conda"):
            candidates.append(base / "envs" / env_name / "bin" / "python")

        return candidates

    @staticmethod
    def interpreter(env_name: str) -> Path:
        candidates = CondaEnv._candidates(env_name)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Conda environment '{env_name}' interpreter not found (searched: {searched}); it is required to run pyrat for the reduced tomogram.")
