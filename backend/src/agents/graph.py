"""LangGraph state-graph orchestrator wiring the 4 agents together with
verification-checkpoint-driven conditional routing.

Design note on HITL pause/resume: rather than depending on a specific
LangGraph interrupt/Command API version, every node is written to be
*idempotent* - it skips recomputation when its output already exists and
is valid. This lets the backend simply call `run_graph(state)` again after
a human answer is merged into `state.human_answers`; the graph naturally
re-enters at START, fast-forwards through already-satisfied stages, and
resumes real work exactly where the human input unblocks it. This avoids
tight-coupling the HITL flow to a fast-moving experimental API surface
while still using a genuine LangGraph StateGraph with conditional edges,
which is what actually gives us the checkpoint-driven routing behavior.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.clarification_agent import clarification_node, finalize_node
from src.agents.compatibility_agent import compatibility_node
from src.agents.perception_agent import perception_node
from src.agents.rag_agent import rag_node
from src.agents.visual_retrieval_agent import visual_retrieval_node
from src.models.schemas import AgentState, JobStatus

RECURSION_LIMIT = 50


def route_after_perception(state: AgentState) -> str:
    if state.status == JobStatus.FAILED:
        return "end"
    if state.perception is None or state.perception.usable_frame_count == 0:
        return "clarification"
    # Image entry path only (part_identification is never set for video
    # jobs): the vision model found essentially nothing describable, so
    # there's no usable query to search the catalogue with yet - ask,
    # rather than let rag_node hard-fail on an empty query. Once the user
    # has supplied a refined_query answer, fall through to rag regardless
    # of identification confidence, mirroring the low_rag_similarity retry
    # pattern below.
    #
    # Two independent escape hatches gate this pause, not just
    # "refined_query" being absent from human_answers: rag_node pops
    # "refined_query" out of human_answers once it's used, so a later
    # re-entry into this router (e.g. after a compatibility-conflict answer
    # sends the graph back through perception on its idempotent no-op path)
    # would otherwise see "refined_query" missing again and route back to
    # clarification forever - clarification_node has nothing new to ask at
    # that point once rag_result already exists, so the graph bounced
    # between the two nodes with no progress until hitting the recursion
    # limit. Checking rag_result alone breaks the *first* pass instead
    # (rag_result is legitimately still None right after refined_query is
    # first supplied, before rag_node has run even once) - so both checks
    # are needed: let it through once EITHER a fresh answer is waiting to
    # be consumed OR rag has already succeeded on a previous pass.
    if (
        state.part_identification is not None
        and not state.part_identification.object_type
        and not state.part_identification.visual_description
        and "refined_query" not in state.human_answers
        and state.rag_result is None
    ):
        return "clarification"
    # Image entry path only: run real visual instance retrieval (MVID)
    # before text RAG. The video entry path (image_path unset) is
    # completely untouched and goes straight to rag, as before.
    if state.image_path:
        return "visual_retrieval"
    return "rag"


def route_after_rag(state: AgentState) -> str:
    if state.status == JobStatus.FAILED:
        return "end"
    if state.rag_result is None or not state.rag_result.passed_threshold:
        return "clarification"
    return "compatibility"


def route_after_compatibility(state: AgentState) -> str:
    if state.status == JobStatus.FAILED:
        return "end"
    if state.compatibility is not None and state.compatibility.has_conflict:
        return "clarification"
    return "finalize"


def route_after_clarification(state: AgentState) -> str:
    if state.status in (JobStatus.AWAITING_HUMAN_CLARIFICATION, JobStatus.FAILED):
        return "end"
    return "perception"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("perception", perception_node)
    graph.add_node("visual_retrieval", visual_retrieval_node)
    graph.add_node("rag", rag_node)
    graph.add_node("compatibility", compatibility_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("perception")

    graph.add_conditional_edges(
        "perception",
        route_after_perception,
        {"clarification": "clarification", "visual_retrieval": "visual_retrieval", "rag": "rag", "end": END},
    )
    graph.add_edge("visual_retrieval", "rag")
    graph.add_conditional_edges(
        "rag", route_after_rag, {"clarification": "clarification", "compatibility": "compatibility", "end": END}
    )
    graph.add_conditional_edges(
        "compatibility",
        route_after_compatibility,
        {"clarification": "clarification", "finalize": "finalize", "end": END},
    )
    graph.add_conditional_edges(
        "clarification", route_after_clarification, {"perception": "perception", "end": END}
    )
    graph.add_edge("finalize", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_graph(state: AgentState) -> AgentState:
    """Synchronous single invocation; safe to call repeatedly (idempotent
    nodes) each time new human input lands in `state.human_answers`.
    """
    graph = get_compiled_graph()
    result = graph.invoke(state, config={"recursion_limit": RECURSION_LIMIT})
    return AgentState.model_validate(result)


def stream_graph(state: AgentState):
    """Yields a full `AgentState` snapshot after every node executes, for
    live UI telemetry. Each snapshot's `current_node` field identifies which
    agent just ran.
    """
    graph = get_compiled_graph()
    for step in graph.stream(state, config={"recursion_limit": RECURSION_LIMIT}, stream_mode="values"):
        yield AgentState.model_validate(step)
