"""Agent 2: Catalogue & RAG Agent.

Searches the ChromaDB spare-parts collection using the query terms surfaced
by Perception (brand/model/damaged component) or a refined query supplied
by the user during clarification. Enforces Checkpoint 3 (RAG similarity
score thresholding).
"""
from __future__ import annotations

from src.config import settings
from src.models.schemas import AgentState, ChatMessage, CheckpointLog, JobStatus, RAGResult
from src.rag.retriever import search_catalog, search_catalog_by_part_number


def _needs_recompute(state: AgentState) -> bool:
    if state.rag_result is None:
        return True
    if "refined_query" in state.human_answers:
        return True
    return False


def rag_node(state: AgentState) -> dict:
    if not _needs_recompute(state):
        return {"current_node": "rag"}

    query = state.human_answers.get("refined_query", state.user_query)

    # Exact-identity path takes priority over semantic search whenever a
    # part number was actually read off the physical part with reasonable
    # confidence (image entry path) - see Checkpoint 4 note in
    # src/models/schemas.py::CatalogMatch.match_type. Skipped once the user
    # has supplied a refined_query, so a manual clarification answer isn't
    # silently overridden by a stale/wrong OCR read.
    pid = state.part_identification
    if (
        pid is not None
        and pid.part_number_if_visible
        and pid.part_number_confidence >= settings.ocr_confidence_threshold
        and "refined_query" not in state.human_answers
    ):
        exact = search_catalog_by_part_number(pid.part_number_if_visible)
        if exact is not None:
            result = RAGResult(query=pid.part_number_if_visible, matches=[exact], top_score=1.0, passed_threshold=True)
            checkpoint = CheckpointLog(
                checkpoint="rag_similarity",
                node="rag",
                passed=True,
                detail=f"Exact part-number match: '{pid.part_number_if_visible}' found directly in catalogue "
                f"(not a similarity score - Checkpoint 3 is bypassed for an exact ID hit).",
            )
            chat_msg = ChatMessage(
                role="assistant",
                content=f"Exact catalogue match on part number '{pid.part_number_if_visible}': {exact.name}.",
            )
            return {
                "current_node": "rag",
                "rag_result": result,
                "status": JobStatus.RUNNING_COMPATIBILITY,
                "checkpoint_logs": state.checkpoint_logs + [checkpoint],
                "chat_history": state.chat_history + [chat_msg],
            }
        # Visible part number didn't match anything in the catalogue -
        # fall through to semantic search on the descriptive query instead
        # of silently discarding the fact that a number was actually read.

    if not query:
        return {
            "current_node": "rag",
            "status": JobStatus.FAILED,
            "error": "No query available for RAG search (perception yielded nothing usable).",
        }

    result = search_catalog(query, top_k=5)

    checkpoint = CheckpointLog(
        checkpoint="rag_similarity",
        node="rag",
        passed=result.passed_threshold,
        detail=f"Top cosine similarity {result.top_score:.3f} "
        f"({'>=' if result.passed_threshold else '<'} threshold {settings.rag_similarity_threshold}) "
        f"for query: '{query}'.",
    )

    top_summary = ", ".join(f"{m.name} ({m.score:.2f})" for m in result.matches[:3]) or "no matches"
    chat_msg = ChatMessage(
        role="assistant",
        content=f"Catalogue search for '{query}' -> top candidates: {top_summary}.",
    )

    new_answers = dict(state.human_answers)
    new_answers.pop("refined_query", None)

    return {
        "current_node": "rag",
        "rag_result": result,
        "status": JobStatus.RUNNING_COMPATIBILITY if result.passed_threshold else JobStatus.RUNNING_RAG,
        "checkpoint_logs": state.checkpoint_logs + [checkpoint],
        "chat_history": state.chat_history + [chat_msg],
        "human_answers": new_answers,
    }
