from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "webui") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "webui"))

from repomap_derive import RepoMapDeriver


def main() -> None:
    deriver = RepoMapDeriver(REPO_ROOT)
    curated = json.loads((REPO_ROOT / "webui" / "repomap_data.json").read_text(encoding="utf-8"))

    out_dir       = REPO_ROOT / "logs" / "repomap"
    skeleton_path = deriver.write_skeleton(out_dir / "derived_skeleton.json")

    drift      = deriver.drift(curated)
    drift_path = out_dir / "drift.json"
    drift_path.write_text(json.dumps(drift, indent=2), encoding="utf-8")

    print(f"skeleton  : {skeleton_path}")
    print(f"drift     : {drift_path}")
    print(f"missing   : {len(drift['missing_files'])} curated nodes point at files that no longer exist")
    print(f"uncovered : {len(drift['uncovered_modules'])} modules not in any curated diagram")


if __name__ == "__main__":
    main()
