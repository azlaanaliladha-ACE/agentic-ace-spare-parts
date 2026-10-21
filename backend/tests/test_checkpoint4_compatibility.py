"""Checkpoint 4: Compatibility Validation - conflicting-parts disambiguation."""
from __future__ import annotations

from src.agents.compatibility_agent import compatibility_node
from src.models.schemas import (
    AgentState,
    CatalogMatch,
    JobStatus,
    PerceptionResult,
    RAGResult,
)


def _perception(brand="JCB", model="4CX", serial="JCB4CX0500") -> PerceptionResult:
    return PerceptionResult(machine_brand=brand, machine_model=model, serial_number=serial, serial_confidence=0.95)


def test_single_compatible_part_resolves_without_conflict():
    match = CatalogMatch(
        part_id="JCB-3CX-DIPPERCYL",
        name="Dipper Cylinder",
        brand="JCB",
        category="Hydraulic Cylinder",
        score=0.9,
        metadata={
            "compatible_models": ["3CX", "3CX Compact"],
            "serial_range": {"min": "JCB3CX0001", "max": "JCB3CX9999"},
            "specs": {"rod_diameter_mm": 60, "duty": "standard"},
        },
    )
    state = AgentState(
        job_id="t1",
        perception=_perception(model="3CX", serial="JCB3CX0500"),
        rag_result=RAGResult(query="dipper cylinder", matches=[match], top_score=0.9, passed_threshold=True),
    )
    result = compatibility_node(state)
    assert result["status"] == JobStatus.GENERATING_RECOMMENDATION
    assert result["compatibility"].has_conflict is False
    assert result["compatibility"].verdicts[0].compatible is True


def test_conflicting_duty_variants_trigger_human_clarification():
    standard = CatalogMatch(
        part_id="JCB-3CX-HYDCYL-045",
        name="Boom Cylinder (Standard)",
        brand="JCB",
        category="Hydraulic Cylinder",
        score=0.91,
        metadata={
            "compatible_models": ["4CX"],
            "serial_range": {"min": "JCB4CX0001", "max": "JCB4CX9999"},
            "specs": {"rod_diameter_mm": 45, "duty": "standard"},
        },
    )
    heavy = CatalogMatch(
        part_id="JCB-3CX-HYDCYL-050HD",
        name="Boom Cylinder (Heavy Duty)",
        brand="JCB",
        category="Hydraulic Cylinder",
        score=0.90,
        metadata={
            "compatible_models": ["4CX"],
            "serial_range": {"min": "JCB4CX0001", "max": "JCB4CX9999"},
            "specs": {"rod_diameter_mm": 50, "duty": "heavy"},
        },
    )
    state = AgentState(
        job_id="t2",
        perception=_perception(),
        rag_result=RAGResult(
            query="boom cylinder", matches=[standard, heavy], top_score=0.91, passed_threshold=True
        ),
    )
    result = compatibility_node(state)

    assert result["status"] == JobStatus.AWAITING_HUMAN_CLARIFICATION
    assert result["compatibility"].has_conflict is True
    assert set(result["compatibility"].conflicting_part_ids) == {
        "JCB-3CX-HYDCYL-045",
        "JCB-3CX-HYDCYL-050HD",
    }


def test_no_compatible_candidate_also_escalates():
    wrong_model = CatalogMatch(
        part_id="CAT-330-BOOMCYL",
        name="Excavator Boom Cylinder",
        brand="Caterpillar",
        category="Hydraulic Cylinder",
        score=0.85,
        metadata={
            "compatible_models": ["330", "330D"],
            "serial_range": {"min": "CAT3300001", "max": "CAT3309999"},
            "specs": {"rod_diameter_mm": 110, "duty": "heavy"},
        },
    )
    state = AgentState(
        job_id="t3",
        perception=_perception(brand="JCB", model="4CX", serial="JCB4CX0500"),
        rag_result=RAGResult(query="boom cylinder", matches=[wrong_model], top_score=0.85, passed_threshold=True),
    )
    result = compatibility_node(state)
    assert result["status"] == JobStatus.AWAITING_HUMAN_CLARIFICATION
    assert result["compatibility"].has_conflict is True
