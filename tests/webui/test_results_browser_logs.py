from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from results_browser import ResultsBrowser
from web_logger      import WebLogger


def _browser(root: Path) -> ResultsBrowser:
    browser = ResultsBrowser(WebLogger())
    assert browser.tree(str(root))["ok"]
    return browser


def test_folder_buckets_log_files_with_content(tmp_path):
    (tmp_path / "inference.log").write_text("line1\nline2\n")
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "weights.bin").write_bytes(b"\x00\x01")

    payload = _browser(tmp_path).folder(str(tmp_path), "")

    assert [log["name"] for log in payload["logs"]] == ["inference.log", "notes.txt"]
    assert payload["logs"][0]["text"] == "line1\nline2\n"
    assert payload["logs"][0]["size"] == len("line1\nline2\n")
    assert [entry["name"] for entry in payload["other"]] == ["weights.bin"]


def test_large_log_serves_tail(tmp_path):
    (tmp_path / "big.log").write_bytes(b"x" * 300000 + b"THE_END")

    payload = _browser(tmp_path).folder(str(tmp_path), "")

    text = payload["logs"][0]["text"]
    assert text.startswith("[showing the tail of the file]")
    assert text.endswith("THE_END")
    assert len(text) < 300000


def test_tree_counts_logs(tmp_path):
    (tmp_path / "a.log").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "c.json").write_text("{}")

    tree = _browser(tmp_path).tree(str(tmp_path))["tree"]

    assert tree["counts"]["logs"] == 2
    assert tree["counts"]["configs"] == 1
    assert tree["counts"]["other"] == 0
