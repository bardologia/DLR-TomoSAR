#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

PORT="${1:-8765}"

for candidate in \
  "$HOME/miniconda3/envs/Dune/bin/python" \
  "$HOME/miniconda3/bin/python" \
  "$(command -v python3 || true)"; do
  if [ -x "$candidate" ]; then
    PY="$candidate"
    break
  fi
done

echo "Starting DLR-TomoSAR control console with: $PY"
cd "$REPO_ROOT"
exec "$PY" -m webui.serve --port "$PORT"
