"""Image entry path: a single photo of an unidentified spare part, as
opposed to a video of a machine. Covers perception_agent.py's
_perceive_from_image branch, the exact-part-number-first priority in
rag_agent.py, the new insufficient_visual_evidence clarification gate, and
search_catalog_by_part_number(). Uses MockVisionClient (VISION_PROVIDER is
explicitly pinned to "mock" throughout) so these stay deterministic and
offline - real-provider behavior is covered by test_ollama_vision_client.py.
"""
from __future__ import annotations

import cv2
import pytest

from src.agents.clarification_agent import clarification_node
from src.agents.graph import route_after_perception
from src.agents.perception_agent import perception_node
from src.agents.rag_agent import rag_node
from src.config import settings
from src.models.schemas import AgentState, CatalogMatch, JobStatus, PartIdentification
from src.rag.retriever import search_catalog_by_part_number


@pytest.fixture(autouse=True)
def _mock_vision(monkeypatch):
    monkeypatch.setattr(settings, "vision_provider", "mock")


@pytest.fixture
def sharp_image_path(tmp_path, sharp_bright_frame):
    path = tmp_path / "part.jpg"
    cv2.imwrite(str(path), sharp_bright_frame)
    return str(path)


@pytest.fixture
def blurry_image_path(tmp_path, blurry_frame):
    path = tmp_path / "blurry_part.jpg"
    cv2.imwrite(str(path), blurry_frame)
    return str(path)


# --------------------------------------------------------------------------- #
# perception_agent._perceive_from_image
# --------------------------------------------------------------------------- #
def test_perception_node_routes_image_path_and_maps_fields(sharp_image_path, monkeypatch):
    fake = PartIdentification(
        object_type="hydraulic cylinder",
        component_category="Hydraulic Cylinder",
        visual_description="A chrome-rod hydraulic cylinder with a mounting flange.",
        visible_features=["chrome rod", "mounting flange"],
        markings=["45"],
        part_number_if_visible="JCB-3CX-HYDCYL-045",
        part_number_confidence=0.93,
        brand_if_visible="JCB",
        machine_context="3CX backhoe loader",
        uncertainties=[],
        overall_confidence=0.85,
    )
    # get_vision_client() constructs a new MockVisionClient instance inside
    # perception_node, so patch the class method rather than an instance.
    from src.vision.perception import MockVisionClient

    monkeypatch.setattr(MockVisionClient, "identify_part", lambda self, img: fake)

    state = AgentState(job_id="img-1", image_path=sharp_image_path)
    result = perception_node(state)

    assert result["status"] == JobStatus.RUNNING_RAG
    perception = result["perception"]
    assert perception.usable_frame_count == 1
    assert perception.machine_brand == "JCB"
    assert perception.machine_model == "3CX backhoe loader"
    assert perception.damaged_components == ["Hydraulic Cylinder"]
    assert result["part_identification"] == fake
    assert "hydraulic cylinder" in (result["user_query"] or "").lower()


def test_perception_node_image_quality_failure_routes_to_clarification(blurry_image_path):
    state = AgentState(job_id="img-2", image_path=blurry_image_path)
    result = perception_node(state)

    assert result["perception"].usable_frame_count == 0
    routed_state = state.model_copy(update=result)
    assert route_after_perception(routed_state) == "clarification"

    clar = clarification_node(routed_state)
    question = clar["clarification_questions"][-1]
    assert question.reason == "low_frame_quality"
    assert "image" in question.question.lower() or "photo" in question.question.lower()


def test_perception_node_missing_image_file_fails_loudly(tmp_path):
    state = AgentState(job_id="img-3", image_path=str(tmp_path / "does_not_exist.jpg"))
    result = perception_node(state)
    assert result["status"] == JobStatus.FAILED
    assert "not found" in result["error"].lower()


# --------------------------------------------------------------------------- #
# Insufficient visual evidence -> clarification (not a hard RAG failure)
# --------------------------------------------------------------------------- #
def test_insufficient_evidence_routes_to_clarification_not_rag_failure():
    state = AgentState(
        job_id="img-4",
        image_path="irrelevant.jpg",
        perception=None,
        part_identification=PartIdentification(uncertainties=["Nothing identifiable in frame."]),
    )
    # Simulate the state perception_node would have produced (usable_frame_count=1
    # but object_type/visual_description both empty).
    from src.models.schemas import PerceptionResult

    state = state.model_copy(update={"perception": PerceptionResult(usable_frame_count=1)})

    assert route_after_perception(state) == "clarification"
    result = clarification_node(state)
    question = result["clarification_questions"][-1]
    assert question.reason == "insufficient_visual_evidence"


def test_insufficient_evidence_gate_clears_once_refined_query_supplied():
    from src.models.schemas import PerceptionResult

    state = AgentState(
        job_id="img-5",
        perception=PerceptionResult(usable_frame_count=1),
        part_identification=PartIdentification(),
        human_answers={"refined_query": "a small bracket with two mounting holes"},
    )
    assert route_after_perception(state) == "rag"


# --------------------------------------------------------------------------- #
# rag_agent: exact part-number match takes priority over semantic search
# --------------------------------------------------------------------------- #
def test_rag_node_prioritizes_exact_part_number_match(monkeypatch):
    from src.models.schemas import PerceptionResult

    exact_match = CatalogMatch(
        part_id="JCB-3CX-HYDCYL-045",
        name="Boom Hydraulic Cylinder",
        brand="JCB",
        category="Hydraulic Cylinder",
        score=1.0,
        match_type="exact_part_number",
        metadata={},
    )
    monkeypatch.setattr("src.agents.rag_agent.search_catalog_by_part_number", lambda pn: exact_match)

    def _fail(*a, **k):
        raise AssertionError("semantic search_catalog should not be called when an exact match exists")

    monkeypatch.setattr("src.agents.rag_agent.search_catalog", _fail)

    state = AgentState(
        job_id="img-6",
        perception=PerceptionResult(usable_frame_count=1),
        part_identification=PartIdentification(
            part_number_if_visible="JCB-3CX-HYDCYL-045", part_number_confidence=0.95
        ),
    )
    result = rag_node(state)

    assert result["status"] == JobStatus.RUNNING_COMPATIBILITY
    assert result["rag_result"].matches[0].match_type == "exact_part_number"
    assert result["rag_result"].matches[0].score == 1.0


def test_rag_node_falls_back_to_semantic_when_part_number_not_in_catalogue(monkeypatch):
    from src.models.schemas import PerceptionResult, RAGResult

    monkeypatch.setattr("src.agents.rag_agent.search_catalog_by_part_number", lambda pn: None)
    semantic_result = RAGResult(query="q", matches=[], top_score=0.1, passed_threshold=False)
    monkeypatch.setattr("src.agents.rag_agent.search_catalog", lambda query, top_k=5: semantic_result)

    state = AgentState(
        job_id="img-7",
        perception=PerceptionResult(usable_frame_count=1),
        user_query="unlabeled bracket",
        part_identification=PartIdentification(
            part_number_if_visible="XYZ-NOT-REAL", part_number_confidence=0.95
        ),
    )
    result = rag_node(state)
    assert result["rag_result"] is semantic_result


def test_rag_node_ignores_low_confidence_part_number(monkeypatch):
    """A part number the vision model itself is unsure about must not
    silently short-circuit to an 'exact' identity claim."""
    from src.models.schemas import PerceptionResult, RAGResult

    def _fail(*a, **k):
        raise AssertionError("exact lookup should not run below the OCR confidence threshold")

    monkeypatch.setattr("src.agents.rag_agent.search_catalog_by_part_number", _fail)
    semantic_result = RAGResult(query="q", matches=[], top_score=0.1, passed_threshold=False)
    monkeypatch.setattr("src.agents.rag_agent.search_catalog", lambda query, top_k=5: semantic_result)

    state = AgentState(
        job_id="img-8",
        perception=PerceptionResult(usable_frame_count=1),
        user_query="blurry part",
        part_identification=PartIdentification(part_number_if_visible="JCB-3CX-HYDCYL-045", part_number_confidence=0.2),
    )
    result = rag_node(state)
    assert result["rag_result"] is semantic_result


# --------------------------------------------------------------------------- #
# search_catalog_by_part_number: real catalogue, no fabricated match
# --------------------------------------------------------------------------- #
def test_search_catalog_by_part_number_exact_hit():
    match = search_catalog_by_part_number("JCB-3CX-HYDCYL-045")
    assert match is not None
    assert match.match_type == "exact_part_number"
    assert match.score == 1.0
    assert match.source == "internal_catalogue"


def test_search_catalog_by_part_number_case_insensitive():
    match = search_catalog_by_part_number("jcb-3cx-hydcyl-045")
    assert match is not None
    assert match.part_id == "JCB-3CX-HYDCYL-045"


def test_search_catalog_by_part_number_no_fabricated_match():
    assert search_catalog_by_part_number("THIS-PART-NUMBER-DOES-NOT-EXIST") is None
