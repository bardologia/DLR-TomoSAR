from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from request_router import RequestRouter


def test_jsonsafe_replaces_non_finite_floats():
    payload = {
        "ok"     : True,
        "value"  : float("nan"),
        "high"   : float("inf"),
        "low"    : float("-inf"),
        "fine"   : 1.5,
        "count"  : 3,
        "name"   : "run",
        "nested" : {"mu": float("nan"), "list": [1.0, float("inf"), {"deep": float("nan")}]},
    }

    safe = RequestRouter._jsonsafe(payload)

    assert safe["ok"] is True
    assert safe["value"] is None
    assert safe["high"] is None
    assert safe["low"] is None
    assert safe["fine"] == 1.5
    assert safe["count"] == 3
    assert safe["name"] == "run"
    assert safe["nested"]["mu"] is None
    assert safe["nested"]["list"] == [1.0, None, {"deep": None}]


def test_jsonsafe_keeps_bools_and_none():
    assert RequestRouter._jsonsafe({"flag": False, "none": None}) == {"flag": False, "none": None}
