"""OllamaVisionClient: raises a clear, specific error when Ollama isn't
reachable or the configured model isn't pulled, rather than silently
falling back to fake mock data - a real provider failure must surface as a
real pipeline failure. Mock is only ever used when VISION_PROVIDER=mock is
explicitly configured. Also covers small local models (e.g. moondream) that
don't reliably follow strict JSON-schema instructions the way frontier
hosted models do."""
from __future__ import annotations

import pytest

from src.config import settings
from src.vision.perception import _parse_entities_json, get_vision_client


def test_get_vision_client_raises_when_ollama_unreachable(monkeypatch):
    monkeypatch.setattr(settings, "vision_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_base_url", "http://127.0.0.1:1")  # nothing listens here

    with pytest.raises(RuntimeError, match="not reachable"):
        get_vision_client()


def test_get_vision_client_raises_when_configured_model_not_pulled(monkeypatch):
    # Simulate a reachable Ollama that has some other model pulled, but not
    # the configured one - independent of whether a real Ollama happens to
    # be running in this environment.
    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"models": [{"name": "some-other-model:latest"}]}

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(settings, "vision_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_vision_model", "definitely-not-a-pulled-model")

    with pytest.raises(RuntimeError, match="not pulled"):
        get_vision_client()


def test_parse_entities_json_survives_malformed_model_output():
    # Real failure observed from moondream: missing comma between keys.
    malformed = '{"entities": [{"entity_type": "machine" "label": "thing"}]}'
    assert _parse_entities_json(malformed, 0) == []


def test_parse_entities_json_survives_non_json_prose():
    assert _parse_entities_json("I see a red machine in the image.", 0) == []


def test_parse_entities_json_still_parses_valid_output():
    valid = (
        '{"entities": [{"entity_type": "machine", "label": "JCB 4CX", '
        '"bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}, '
        '"confidence": 0.9, "raw_text": null}]}'
    )
    entities = _parse_entities_json(valid, 3)
    assert len(entities) == 1
    assert entities[0].label == "JCB 4CX"
    assert entities[0].frame_index == 3


def test_parse_entities_json_normalizes_moondream_style_output():
    # Real response observed from moondream on an actual JCB 3CX photo,
    # after the repeat-penalty fix broke its earlier infinite-loop failure:
    # typo'd key ("entitytype"), bbox as a bare list with mixed str/num
    # entries, and an out-of-range negative confidence.
    raw = (
        '{"entities":[{"entitytype":"machine", "label":["JCB"],'
        '"bbox":[0.17,"0.65","0.83", 0.87],"confidence":-0.85}]}'
    )
    entities = _parse_entities_json(raw, 0)
    assert len(entities) == 1
    entity = entities[0]
    assert entity.entity_type.value == "machine"
    assert entity.label == "JCB"
    assert entity.bbox.x_min == 0.17
    assert entity.bbox.y_min == 0.65
    assert 0.0 <= entity.confidence <= 1.0


def test_parse_entities_json_defaults_missing_confidence_to_uncertain():
    raw = '{"entities":[{"entity_type":"serial_tag","label":"tag","bbox":[0,0,1,1]}]}'
    entities = _parse_entities_json(raw, 0)
    assert len(entities) == 1
    assert entities[0].confidence == 0.5  # below the 0.8 checkpoint threshold, i.e. "uncertain"
