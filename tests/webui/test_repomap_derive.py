from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from repomap_derive import ImportGraph, RepoMapDeriver


def test_import_graph_finds_known_internal_edges():
    graph = ImportGraph(REPO_ROOT).build()

    assert "pipelines.backbone.inference.pipeline" in graph
    assert "pipelines.backbone.inference.metrics" in graph["pipelines.backbone.inference.pipeline"]
    assert all(target in graph for targets in graph.values() for target in targets)


def test_skeleton_groups_by_top_folder():
    skeleton = RepoMapDeriver(REPO_ROOT).skeleton()
    folders  = {folder["folder"] for folder in skeleton["folders"]}

    assert {"main", "pipelines", "tools", "models", "configuration", "webui"} <= folders

    pipelines = next(folder for folder in skeleton["folders"] if folder["folder"] == "pipelines")
    assert pipelines["nodes"] and pipelines["edges"]


def test_curated_repomap_references_only_existing_files():
    curated = json.loads((WEBUI_ROOT / "repomap_data.json").read_text(encoding="utf-8"))
    drift   = RepoMapDeriver(REPO_ROOT).drift(curated)

    assert drift["missing_files"] == []


def test_drift_reports_vanished_modules():
    curated = {"folders": [{"folder": "tools", "diagrams": [{"key": "d", "nodes": [{"id": "gone", "module": "tools/erased_module.py"}]}]}]}
    drift   = RepoMapDeriver(REPO_ROOT).drift(curated)

    assert len(drift["missing_files"]) == 1
    assert drift["missing_files"][0]["module"] == "tools/erased_module.py"
