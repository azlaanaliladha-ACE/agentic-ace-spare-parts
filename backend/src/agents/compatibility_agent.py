"""Agent 3: Compatibility & Constraint Agent.

Validates each RAG candidate part's specs (compatible machine models,
serial range, dimensional specs such as rod/bore diameter) against what
Perception observed on the machine. Enforces Checkpoint 4 (compatibility
validation): when multiple parts remain plausibly compatible but conflict
on a distinguishing spec (e.g. standard vs heavy-duty seal kit), the graph
must pause for human disambiguation rather than guessing.
"""
from __future__ import annotations

import re

from src.models.schemas import (
    AgentState,
    ChatMessage,
    CheckpointLog,
    CompatibilityResult,
    CompatibilityVerdict,
    JobStatus,
    RAGResult,
    SpecCheck,
)

# Specs that, if two otherwise-compatible parts disagree on them, constitute
# a genuine ambiguity requiring the user to pick (rather than silently
# choosing e.g. the standard-duty part over the heavy-duty one).
_DISTINGUISHING_SPECS = ("duty", "rod_diameter_mm", "bore_diameter_mm", "seal_material")


def _model_matches(machine_model: str | None, compatible_models: list[str]) -> bool | None:
    if not machine_model or not compatible_models:
        return None
    needle = machine_model.lower().strip()
    return any(needle in cm.lower() or cm.lower() in needle for cm in compatible_models)


def _serial_in_range(serial_number: str | None, serial_range: dict) -> bool | None:
    if not serial_number or not serial_range:
        return None
    lo, hi = serial_range.get("min"), serial_range.get("max")
    if not lo or not hi or lo == "N/A" or hi == "N/A":
        return None

    def _numeric_suffix(s: str) -> tuple[str, int] | None:
        m = re.match(r"^([A-Za-z]*)(\d+)$", s)
        return (m.group(1), int(m.group(2))) if m else None

    sn, lon, hin = _numeric_suffix(serial_number), _numeric_suffix(lo), _numeric_suffix(hi)
    if not sn or not lon or not hin:
        return None
    prefix, value = sn
    if prefix != lon[0] or prefix != hin[0]:
        return False
    return lon[1] <= value <= hin[1]


def _build_verdict(match, state: AgentState) -> CompatibilityVerdict:
    perception = state.perception
    checks: list[SpecCheck] = []

    model_match = _model_matches(
        f"{perception.machine_brand or ''} {perception.machine_model or ''}".strip() if perception else None,
        match.metadata.get("compatible_models", []),
    )
    if model_match is not None:
        checks.append(
            SpecCheck(
                field="compatible_models",
                required_value=f"{perception.machine_brand} {perception.machine_model}",
                actual_value=match.metadata.get("compatible_models", []),
                match=model_match,
            )
        )

    serial_match = _serial_in_range(
        perception.serial_number if perception else None, match.metadata.get("serial_range", {})
    )
    if serial_match is not None:
        checks.append(
            SpecCheck(
                field="serial_range",
                required_value=perception.serial_number if perception else None,
                actual_value=match.metadata.get("serial_range", {}),
                match=serial_match,
            )
        )

    if not checks:
        # No hard data to validate against (e.g. universal filter, or
        # perception couldn't resolve model/serial) - accept on semantic
        # similarity alone but say so explicitly.
        compatible = True
        notes = "No serial/model constraint data available; matched on semantic similarity only."
    else:
        compatible = all(c.match for c in checks)
        notes = None

    return CompatibilityVerdict(
        part_id=match.part_id, part_name=match.name, compatible=compatible, checks=checks, notes=notes
    )


def _find_conflict(verdicts: list[CompatibilityVerdict], rag_result: RAGResult) -> tuple[bool, str | None, list[str]]:
    compatible = [v for v in verdicts if v.compatible]
    if len(compatible) <= 1:
        return False, None, []

    match_by_id = {m.part_id: m for m in rag_result.matches}
    spec_sets = []
    for v in compatible:
        specs = match_by_id[v.part_id].metadata.get("specs", {})
        spec_sets.append({k: specs.get(k) for k in _DISTINGUISHING_SPECS if k in specs})

    if any(s != spec_sets[0] for s in spec_sets[1:]):
        diffs = ", ".join(f"{v.part_name} ({s})" for v, s in zip(compatible, spec_sets))
        return True, f"Multiple compatible parts differ on key specs: {diffs}", [v.part_id for v in compatible]

    return False, None, []


def compatibility_node(state: AgentState) -> dict:
    if state.compatibility is not None and not state.compatibility.has_conflict:
        return {"current_node": "compatibility"}

    if state.rag_result is None or not state.rag_result.matches:
        return {
            "current_node": "compatibility",
            "status": JobStatus.FAILED,
            "error": "Compatibility agent invoked with no RAG results.",
        }

    qualifying = [m for m in state.rag_result.matches if m.score >= state.rag_result.matches[0].score - 0.05]
    verdicts = [_build_verdict(m, state) for m in qualifying]

    has_conflict, conflict_reason, conflicting_ids = _find_conflict(verdicts, state.rag_result)
    n_compatible = sum(1 for v in verdicts if v.compatible)

    if n_compatible == 0:
        has_conflict = True
        conflict_reason = "No candidate part could be fully validated against the detected machine/serial."
        conflicting_ids = [v.part_id for v in verdicts]

    result = CompatibilityResult(
        verdicts=verdicts,
        has_conflict=has_conflict,
        conflict_reason=conflict_reason,
        conflicting_part_ids=conflicting_ids,
    )

    if not verdicts:
        checkpoint_detail = "No verdicts produced."
    else:
        checkpoint_detail = conflict_reason or f"Single compatible part resolved: {verdicts[0].part_name}"

    checkpoint = CheckpointLog(
        checkpoint="compatibility",
        node="compatibility",
        passed=not has_conflict,
        detail=checkpoint_detail,
    )

    chat_msg = ChatMessage(
        role="assistant",
        content=(
            f"Compatibility check: {n_compatible}/{len(verdicts)} candidates validated. "
            + (f"Conflict: {conflict_reason}" if has_conflict else "Resolved to a single compatible part.")
        ),
    )

    return {
        "current_node": "compatibility",
        "compatibility": result,
        "status": JobStatus.AWAITING_HUMAN_CLARIFICATION if has_conflict else JobStatus.GENERATING_RECOMMENDATION,
        "checkpoint_logs": state.checkpoint_logs + [checkpoint],
        "chat_history": state.chat_history + [chat_msg],
    }
