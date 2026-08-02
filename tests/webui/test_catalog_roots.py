from __future__ import annotations

from pathlib import Path

import pytest

from catalog_roots import CatalogRoots


@pytest.fixture
def roots():
    return CatalogRoots()


def test_an_empty_path_reports_the_caller_message(roots):
    target, error = roots.resolve("   ")

    assert target is None
    assert error  == CatalogRoots.NOT_SET

    target, error = roots.resolve("", "not set")

    assert target is None
    assert error  == "not set"


def test_a_relative_path_is_refused(roots, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()

    target, error = roots.resolve("runs")

    assert target is None
    assert error  == "an absolute path is required"


def test_a_file_or_a_missing_directory_is_refused(roots, tmp_path):
    (tmp_path / "note.md").write_text("x")

    assert roots.resolve(str(tmp_path / "note.md"))[1].startswith("not a directory")
    assert roots.resolve(str(tmp_path / "nowhere"))[1].startswith("not a directory")


def test_resolution_follows_symlinks_and_expands_the_home_marker(roots, tmp_path, monkeypatch):
    real = tmp_path / "real_runs"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)

    target, error = roots.resolve(str(tmp_path / "link"))

    assert error == ""
    assert target == real.resolve()

    monkeypatch.setenv("HOME", str(tmp_path))
    assert roots.resolve("~/real_runs")[0] == real.resolve()


def test_a_refused_path_is_never_recorded(roots, tmp_path):
    roots.open(str(tmp_path / "nowhere"))

    assert roots.snapshot() == ()


def test_open_records_the_resolved_root(roots, tmp_path):
    target, error = roots.open(str(tmp_path) + "/")

    assert error == ""
    assert roots.snapshot() == (str(target),)
    assert roots.known(str(target)) is True
    assert roots.known(str(tmp_path) + "/") is False


def test_containment_covers_nested_paths_only(roots, tmp_path):
    runs = tmp_path / "runs"
    (runs / "group" / "run_a").mkdir(parents=True)
    (tmp_path / "runs_backup").mkdir()

    roots.open(str(runs))

    assert roots.contains(runs / "group" / "run_a") is True
    assert roots.contains(runs)                     is True
    assert roots.contains(tmp_path / "runs_backup") is False
    assert roots.contains(tmp_path)                 is False


def test_containment_rejects_a_shared_name_prefix(roots, tmp_path):
    (tmp_path / "runs").mkdir()
    roots.open(str(tmp_path / "runs"))

    assert roots.contains(Path(str(tmp_path / "runs") + "_backup") / "leak.png") is False


def test_the_enclosing_root_is_the_deepest_match(roots, tmp_path):
    outer = tmp_path / "runs"
    inner = outer / "group"
    inner.mkdir(parents=True)

    roots.open(str(outer))
    roots.open(str(inner))

    assert roots.enclosing(inner / "run_a") == inner.resolve()
    assert roots.enclosing(outer / "run_b") == outer.resolve()
    assert roots.enclosing(tmp_path / "elsewhere") is None


def test_the_same_root_is_recorded_once(roots, tmp_path):
    roots.open(str(tmp_path))
    roots.open(str(tmp_path) + "/")
    roots.open(str(tmp_path / "." ))

    assert len(roots.snapshot()) == 1
