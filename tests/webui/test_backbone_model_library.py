from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from backbone_model_library import BackboneModelLibrary
from models                 import BACKBONE_MODEL_REGISTRY


@pytest.fixture(scope="module")
def library():
    return BackboneModelLibrary()


@pytest.fixture(scope="module")
def collected(library):
    return library.collect()


def _model_keys(collected):
    return [model["key"] for family in collected for model in family["models"]]


def test_config_classes_cover_the_backbone_registry():
    assert set(BackboneModelLibrary.CONFIG_CLASSES) == set(BACKBONE_MODEL_REGISTRY)


def test_note_files_cover_the_backbone_registry():
    assert set(BackboneModelLibrary.NOTE_FILES) == set(BACKBONE_MODEL_REGISTRY)


def test_families_cover_the_backbone_registry_exactly_once(collected):
    keys = _model_keys(collected)

    assert sorted(keys) == sorted(set(keys))
    assert set(keys) == set(BACKBONE_MODEL_REGISTRY)


def test_every_model_resolves_a_note(library):
    missing = [key for key in BackboneModelLibrary.NOTE_FILES if library.note(key) is None]

    assert not missing, f"models without a resolvable note under notes/models: {missing}"


def test_collect_resolves_display_defaults(collected):
    for family in collected:
        for model in family["models"]:
            assert model["activation"], f"{model['key']} has no activation label"
            assert model["normalization"], f"{model['key']} has no normalization label"
