from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from dataset_browser import DatasetBrowser
from web_logger      import WebLogger


def _make_run(directory: Path, checkpoint: bool = False, inference: bool = False) -> None:
    (directory / "meta").mkdir(parents=True)
    (directory / "meta" / "model_config.json").write_text("{}")
    if checkpoint:
        (directory / "best_model.pt").write_text("x")
    if inference:
        (directory / "inference" / "sub").mkdir(parents=True)


def test_runs_are_discovered_at_any_depth(tmp_path):
    runs = tmp_path / "runs"
    _make_run(runs / "run_top", checkpoint=True)
    _make_run(runs / "group_a" / "run_a1")
    _make_run(runs / "group_a" / "deep" / "run_a2", inference=True)
    (runs / "empty_group").mkdir(parents=True)

    result = DatasetBrowser(WebLogger()).runs([str(runs)])

    assert result["ok"] is True
    names = sorted(entry["name"] for entry in result["runs"])
    assert names == ["group_a/deep/run_a2", "group_a/run_a1", "run_top"]


def test_run_names_are_relative_to_base_and_expose_flags(tmp_path):
    runs = tmp_path / "runs"
    _make_run(runs / "group_a" / "run_a1")
    _make_run(runs / "group_a" / "deep" / "run_a2", checkpoint=True, inference=True)

    entries = {entry["name"]: entry for entry in DatasetBrowser(WebLogger()).runs([str(runs)])["runs"]}

    assert entries["group_a/run_a1"]["has_checkpoint"] is False
    assert entries["group_a/run_a1"]["has_inference"] is False
    assert entries["group_a/deep/run_a2"]["has_checkpoint"] is True
    assert entries["group_a/deep/run_a2"]["has_inference"] is True
    assert Path(entries["group_a/deep/run_a2"]["path"]) == runs / "group_a" / "deep" / "run_a2"


def test_walk_prunes_at_run_boundary_and_skips_hidden(tmp_path):
    runs = tmp_path / "runs"
    _make_run(runs / "run_top", inference=True)
    hidden = runs / ".hidden_run"
    hidden.mkdir()
    (hidden / "meta").mkdir()

    names = [entry["name"] for entry in DatasetBrowser(WebLogger()).runs([str(runs)])["runs"]]

    assert names == ["run_top"]
    assert not any("meta" in name.split("/")[-1] for name in names)
    assert not any("inference" in name for name in names)


def test_run_groups_lists_parent_dirs_of_runs_with_counts(tmp_path):
    runs = tmp_path / "runs"
    _make_run(runs / "group_a" / "seed0")
    _make_run(runs / "group_a" / "seed1")
    _make_run(runs / "group_b" / "seed0")
    _make_run(runs / "run_top")

    result  = DatasetBrowser(WebLogger()).run_groups([str(runs)])
    entries = {entry["name"]: entry for entry in result["groups"]}

    assert result["ok"] is True
    assert sorted(entries) == ["group_a", "group_b"]
    assert entries["group_a"]["n_runs"] == 2
    assert entries["group_b"]["n_runs"] == 1
    assert Path(entries["group_a"]["path"]) == runs / "group_a"
