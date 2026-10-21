"""Agent 4: Clarification & Dialogue Agent.

Two responsibilities, split into two graph nodes:

  clarification_node - inspects which verification checkpoint is currently
    blocking (frame quality / RAG similarity / compatibility conflict),
    either applies a human answer already present in `state.human_answers`
    (unblocking the pipeline) or formulates a HITL question and pauses the
    graph by setting status=AWAITING_HUMAN_CLARIFICATION.

  finalize_node - once nothing is blocking, generates the final purchase
    recommendation(s) via the Checkpoint 5 structured-output guardrail.
"""
from __future__ import annotations

import logging

from src.config import settings
from src.models.schemas import (
    AgentState,
    ChatMessage,
    ClarificationQuestion,
    CompatibilityResult,
    JobStatus,
    PurchaseRecommendation,
)
from src.agents.llm_utils import StructuredOutputError, generate_structured_recommendation

logger = logging.getLogger(__name__)


def _pause(state: AgentState, question: ClarificationQuestion) -> dict:
    questions = [q for q in state.clarification_questions if q.reason != question.reason]
    questions.append(question)
    return {
        "current_node": "clarification",
        "status": JobStatus.AWAITING_HUMAN_CLARIFICATION,
        "clarification_questions": questions,
        "chat_history": state.chat_history + [ChatMessage(role="assistant", content=question.question)],
    }


def clarification_node(state: AgentState) -> dict:
    perception = state.perception

    # --- Checkpoint 1 fallback: frame quality --------------------------------
    if perception is None or perception.usable_frame_count == 0:
        if "video_path" in state.human_answers:
            new_answers = dict(state.human_answers)
            new_path = new_answers.pop("video_path")
            return {
                "current_node": "clarification",
                "video_path": new_path,
                "perception": None,
                "status": JobStatus.RUNNING_PERCEPTION,
                "human_answers": new_answers,
                "chat_history": state.chat_history
                + [ChatMessage(role="user", content=f"Re-uploaded video for reprocessing: {new_path}")],
            }
        if "image_path" in state.human_answers:
            new_answers = dict(state.human_answers)
            new_path = new_answers.pop("image_path")
            return {
                "current_node": "clarification",
                "image_path": new_path,
                "perception": None,
                "part_identification": None,
                "status": JobStatus.RUNNING_PERCEPTION,
                "human_answers": new_answers,
                "chat_history": state.chat_history
                + [ChatMessage(role="user", content=f"Re-uploaded image for reprocessing: {new_path}")],
            }
        if state.image_path:
            question = ClarificationQuestion(
                field="image_path",
                reason="low_frame_quality",
                question=(
                    "This image failed the quality check (too blurry, too dark/bright, or "
                    "resolution too low). Please re-upload a clearer, well-lit photo of the part."
                ),
            )
        else:
            question = ClarificationQuestion(
                field="video_path",
                reason="low_frame_quality",
                question=(
                    "Every sampled frame from this video failed the quality check "
                    "(too blurry, too dark/bright, or resolution too low). Please re-upload "
                    "a clearer video, ideally with steady framing and good lighting on the "
                    "brand plate and serial tag."
                ),
            )
        return _pause(state, question)

    # --- Image entry path: insufficient visual evidence to even form a query -
    # Also gated on rag_result being unset (not just "refined_query" being
    # absent), matching route_after_perception's gate in graph.py - see that
    # function's comment for why both checks are needed: rag_result alone
    # would block the very first pass (before rag_node has run even once),
    # while "refined_query" alone re-triggers this question forever once
    # rag_node pops it out of human_answers on a later, unrelated pause
    # (e.g. a compatibility conflict).
    if (
        state.part_identification is not None
        and not state.part_identification.object_type
        and not state.part_identification.visual_description
        and state.rag_result is None
    ):
        if "refined_query" in state.human_answers:
            return {"current_node": "clarification", "status": JobStatus.RUNNING_RAG}
        question = ClarificationQuestion(
            field="part_description",
            reason="insufficient_visual_evidence",
            question=(
                "I couldn't identify enough from this image to search the catalogue. "
                "What machine is this part from (brand/model)? Is there a part number "
                "stamped or labeled on it? What is its general shape or function "
                "(e.g. seal, bracket, pump, filter)?"
            ),
        )
        return _pause(state, question)

    # --- Checkpoint 3 fallback: RAG similarity -------------------------------
    if state.rag_result is None or not state.rag_result.passed_threshold:
        if "refined_query" in state.human_answers:
            return {"current_node": "clarification", "status": JobStatus.RUNNING_RAG}
        top = state.rag_result.top_score if state.rag_result else 0.0
        question = ClarificationQuestion(
            field="query",
            reason="low_rag_similarity",
            question=(
                f"I couldn't find a confident catalogue match (top similarity {top:.2f} "
                f"< {settings.rag_similarity_threshold:.2f}). Can you describe the part more "
                "specifically - e.g. exact component name, machine model, or a dimension "
                "such as rod/bore diameter?"
            ),
        )
        return _pause(state, question)

    # --- Checkpoint 4 fallback: compatibility conflict -----------------------
    if state.compatibility is not None and state.compatibility.has_conflict:
        if "compatibility_choice" in state.human_answers:
            new_answers = dict(state.human_answers)
            chosen_id = new_answers.pop("compatibility_choice")
            new_verdicts = []
            for v in state.compatibility.verdicts:
                v2 = v.model_copy(deep=True)
                if v.part_id in state.compatibility.conflicting_part_ids:
                    v2.compatible = v.part_id == chosen_id
                new_verdicts.append(v2)
            new_compat = CompatibilityResult(
                verdicts=new_verdicts, has_conflict=False, conflict_reason=None, conflicting_part_ids=[]
            )
            return {
                "current_node": "clarification",
                "compatibility": new_compat,
                "status": JobStatus.GENERATING_RECOMMENDATION,
                "human_answers": new_answers,
                "chat_history": state.chat_history
                + [ChatMessage(role="user", content=f"Selected part: {chosen_id}")],
            }
        options = [
            f"{v.part_name} ({v.part_id})"
            for v in state.compatibility.verdicts
            if v.part_id in state.compatibility.conflicting_part_ids
        ]
        question = ClarificationQuestion(
            field="part_id",
            reason="compatibility_conflict",
            question=(
                f"Multiple parts are plausibly compatible: {state.compatibility.conflict_reason}. "
                "Which one do you need?"
            ),
            options=options,
        )
        return _pause(state, question)

    # Nothing blocking - safety net, let the router send this to finalize.
    return {"current_node": "clarification", "status": JobStatus.GENERATING_RECOMMENDATION}


def finalize_node(state: AgentState) -> dict:
    """Checkpoint 5: generates a validated PurchaseRecommendation for every
    compatible verdict, via the structured-output retry loop.
    """
    if state.recommendations:
        return {"current_node": "finalize"}

    if state.compatibility is None or state.rag_result is None:
        return {
            "current_node": "finalize",
            "status": JobStatus.FAILED,
            "error": "finalize_node reached without compatibility/RAG results.",
        }

    match_by_id = {m.part_id: m for m in state.rag_result.matches}
    recommendations: list[PurchaseRecommendation] = []
    new_logs = []
    errors: list[str] = []

    for verdict in state.compatibility.verdicts:
        if not verdict.compatible:
            continue
        match = match_by_id.get(verdict.part_id)
        if match is None:
            continue
        try:
            validated, logs = generate_structured_recommendation(verdict, match, PurchaseRecommendation)
            recommendations.append(validated)  # type: ignore[arg-type]
            new_logs.extend(logs)
        except StructuredOutputError as exc:
            logger.error("Structured output guardrail exhausted retries: %s", exc)
            errors.append(str(exc))

    if not recommendations:
        return {
            "current_node": "finalize",
            "status": JobStatus.FAILED,
            "error": "; ".join(errors) or "No compatible parts to recommend.",
            "checkpoint_logs": state.checkpoint_logs + new_logs,
        }

    summary = "; ".join(f"{r.name} (${r.price_usd:.2f}, confidence {r.confidence:.2f})" for r in recommendations)
    chat_msg = ChatMessage(role="assistant", content=f"Final recommendation: {summary}")

    return {
        "current_node": "finalize",
        "recommendations": recommendations,
        "status": JobStatus.COMPLETED,
        "checkpoint_logs": state.checkpoint_logs + new_logs,
        "chat_history": state.chat_history + [chat_msg],
    }
