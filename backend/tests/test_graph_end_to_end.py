"""End-to-end LangGraph orchestration test: video in -> either a completed
recommendation or a well-formed clarification request out.

Requires the same offline dependencies as test_checkpoint3 (embedding
model + local Chroma dir); skips gracefully if unavailable.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def ingested_catalog(tmp_path_factory):
    from src.config import settings

    settings.chroma_host = None
    settings.chroma_persist_dir = str(tmp_path_factory.mktemp("chroma_e2e"))
    try:
        from src.rag.ingest_catalog import ingest

        ingest()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Skipping end-to-end graph test: catalogue ingestion unavailable ({exc})")


def test_full_pipeline_reaches_a_terminal_state(ingested_catalog, synthetic_video, monkeypatch):
    from src.agents.graph import run_graph
    from src.config import settings
    from src.models.schemas import AgentState, JobStatus

    # This test exercises graph orchestration (routing/checkpoints), not any
    # particular vision provider - pin explicitly to mock so it stays
    # deterministic and offline regardless of the local .env's
    # VISION_PROVIDER (e.g. a real ollama/anthropic config for manual runs).
    monkeypatch.setattr(settings, "vision_provider", "mock")

    state = AgentState(job_id="e2e-1", video_path=synthetic_video)
    final = run_graph(state)

    assert final.status in (
        JobStatus.COMPLETED,
        JobStatus.AWAITING_HUMAN_CLARIFICATION,
        JobStatus.FAILED,
    )
    # Every terminal path must have left an audit trail of checkpoint results.
    assert len(final.checkpoint_logs) > 0

    if final.status == JobStatus.COMPLETED:
        assert final.recommendations
        assert all(r.confidence >= 0 for r in final.recommendations)
    elif final.status == JobStatus.AWAITING_HUMAN_CLARIFICATION:
        assert final.clarification_questions
        assert final.clarification_questions[-1].question


def test_compatibility_conflict_resolves_after_human_answer(ingested_catalog, synthetic_video):
    """Forces a compatibility conflict deterministically, then verifies the
    graph resumes and reaches COMPLETED once a human answer is supplied -
    exercising the Checkpoint 4 pause/resume flow end to end.
    """
    from src.agents.graph import run_graph
    from src.models.schemas import (
        AgentState,
        CompatibilityResult,
        CompatibilityVerdict,
        JobStatus,
        PerceptionResult,
        RAGResult,
        CatalogMatch,
    )

    matches = [
        CatalogMatch(
            part_id="JCB-3CX-HYDCYL-045",
            name="Boom Cylinder (Standard)",
            brand="JCB",
            category="Hydraulic Cylinder",
            score=0.9,
            metadata={"compatible_models": ["4CX"], "serial_range": {}, "specs": {"duty": "standard"}, "price_usd": 1240.0},
        ),
        CatalogMatch(
            part_id="JCB-3CX-HYDCYL-050HD",
            name="Boom Cylinder (Heavy Duty)",
            brand="JCB",
            category="Hydraulic Cylinder",
            score=0.89,
            metadata={"compatible_models": ["4CX"], "serial_range": {}, "specs": {"duty": "heavy"}, "price_usd": 1560.0},
        ),
    ]
    state = AgentState(
        job_id="e2e-2",
        perception=PerceptionResult(machine_brand="JCB", machine_model="4CX", usable_frame_count=1),
        rag_result=RAGResult(query="boom cylinder", matches=matches, top_score=0.9, passed_threshold=True),
    )
    paused = run_graph(state)
    assert paused.status == JobStatus.AWAITING_HUMAN_CLARIFICATION
    question = next(q for q in paused.clarification_questions if q.reason == "compatibility_conflict")

    resumed_input = paused.model_copy(
        update={"human_answers": {**paused.human_answers, "compatibility_choice": "JCB-3CX-HYDCYL-045"}}
    )
    final = run_graph(resumed_input)

    assert final.status == JobStatus.COMPLETED
    assert any(r.part_id == "JCB-3CX-HYDCYL-045" for r in final.recommendations)
