"""Visual retrieval agent (image entry path only): real MVID instance
retrieval via DINOv2 embeddings + a dedicated ChromaDB collection. See
src/rag/visual_retriever.py and src/rag/mvid_ingest.py.

This is a genuinely new agent action - it embeds the actual uploaded image
and queries a real vector index, returning real nearest-neighbour
candidates. It never runs for the video entry path (see
route_after_perception in agents/graph.py), and its output
(state.visual_search / VisualCandidate) is kept entirely separate from
rag_node's catalogue search - visual similarity to an MVID photo is never
treated as, or converted into, a catalogue identity or a compatibility
verdict.
"""
from __future__ import annotations

import logging

from PIL import Image

from src.models.schemas import AgentState, ChatMessage
from src.rag.visual_retriever import search_by_image

logger = logging.getLogger(__name__)


def _needs_recompute(state: AgentState) -> bool:
    return state.visual_search is None


def visual_retrieval_node(state: AgentState) -> dict:
    if not state.image_path:
        # Defensive no-op - route_after_perception never sends video jobs here.
        return {"current_node": "visual_retrieval"}

    if not _needs_recompute(state):
        return {"current_node": "visual_retrieval"}

    try:
        image = Image.open(state.image_path).convert("RGB")
        result = search_by_image(image, query_image_path=state.image_path, top_k=5)
    except Exception as exc:
        # A missing/unbuilt MVID index (or any other real failure) must not
        # crash the whole pipeline or silently claim success with empty
        # results - text RAG (rag_node, next) is an independent signal and
        # still runs. state.visual_search is left None: "not available",
        # never a fabricated empty-but-successful result.
        logger.warning("Visual retrieval failed, continuing without visual candidates: %s", exc)
        chat_msg = ChatMessage(role="assistant", content=f"Visual retrieval unavailable: {exc}")
        return {"current_node": "visual_retrieval", "chat_history": state.chat_history + [chat_msg]}

    if result.candidates:
        summary = ", ".join(f"{c.object_id} ({c.similarity:.2f})" for c in result.candidates[:3])
        content = f"Visual retrieval (MVID proof-of-concept index): top candidates {summary}."
    else:
        content = "Visual retrieval (MVID proof-of-concept index): no candidates found."

    return {
        "current_node": "visual_retrieval",
        "visual_search": result,
        "chat_history": state.chat_history + [ChatMessage(role="assistant", content=content)],
    }
