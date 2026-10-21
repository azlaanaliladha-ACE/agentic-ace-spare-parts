"""Runs one real job through the actual LangGraph pipeline (in-process, same
code path the FastAPI backend uses) and records everything observable about
it - per-agent timing, real Ollama token/latency stats, checkpoint results,
context/memory footprint, and the final AgentState - into a static JSON
fixture under data/recorded_runs/.

This fixture is what src/frontend/ops_dashboard.py replays. The dashboard
intentionally does NOT call the live backend: recording once and replaying
a fixed run keeps the "Agent Ops" view deterministic and demo-safe instead
of depending on a live Ollama call every time someone opens the tab.

Run with: .venv\\Scripts\\python.exe -m scripts.record_agent_run [video_path]
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.graph import stream_graph  # noqa: E402
from src.config import settings  # noqa: E402
from src.models.schemas import AgentState  # noqa: E402
from src.observability import call_log  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "uploads" / "40b845ca6e1e44df91b1ed7695dd9362_01_jcb_4cx_boom_cylinder.mp4"
OUT_DIR = ROOT / "data" / "recorded_runs"


def _ollama_model_info(model: str) -> dict:
    """Real /api/show call - parameter_size, quantization, family are
    whatever Ollama actually reports for the locally pulled model, not
    hardcoded guesses."""
    try:
        resp = httpx.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/show", json={"name": model}, timeout=10
        )
        resp.raise_for_status()
        payload = resp.json()
        details = payload.get("details", {})
        return {
            "model": model,
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "family": details.get("family"),
            "format": details.get("format"),
        }
    except httpx.HTTPError as exc:
        return {"model": model, "error": str(exc)}


def _embedding_model_info() -> dict:
    """Loads the real sentence-transformers model already cached locally
    (from the catalog ingest step) and counts its actual torch parameters -
    a real number computed from the loaded model, not a quoted spec."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model)
    n_params = sum(p.numel() for p in model[0].auto_model.parameters())
    return {
        "model": settings.embedding_model,
        "parameter_count": n_params,
        "embedding_dim": model.get_sentence_embedding_dimension(),
    }


def main() -> None:
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    if not video_path.exists():
        raise SystemExit(f"Sample video not found: {video_path}")

    job_id = uuid.uuid4().hex
    state = AgentState(job_id=job_id, video_path=str(video_path))

    call_log.drain()  # clear anything left over from a prior in-process run

    timeline: list[dict] = []
    t_run_start = time.time()
    prev_t = t_run_start
    prev_node: str | None = None
    final_state: AgentState | None = None

    # Fixed, honestly-labeled auto-answers for a demo continuation: when the
    # graph pauses for human input (AWAITING_HUMAN_CLARIFICATION), this
    # script plays a scripted human turn and resumes, so the recorded run
    # can reach compatibility/finalize instead of stopping at the first
    # checkpoint pause. Real human answers in the live app come from the
    # Streamlit clarification card (src/frontend/app.py::handle_clarification),
    # not from here.
    _AUTO_ANSWERS = {
        "low_rag_similarity": ("refined_query", "hydraulic boom cylinder seal kit for JCB 4CX backhoe loader"),
        "compatibility_conflict": None,  # resolved dynamically below (needs an actual candidate id)
    }
    round_num = 0

    def _run_round(s: AgentState):
        nonlocal prev_t, prev_node, final_state
        for snapshot in stream_graph(s):
            now = time.time()
            node = snapshot.current_node
            # Each yielded snapshot corresponds to exactly one node having
            # just completed (stream_mode="values"), so every iteration is a
            # real, distinct timeline event - no dedup needed.
            timeline.append(
                {
                    "round": round_num,
                    "node": node,
                    "entered_at_rel_ms": round((now - t_run_start) * 1000, 1),
                    "duration_ms": round((now - prev_t) * 1000, 1),
                    "status_after": snapshot.status.value,
                }
            )
            prev_t = now
            prev_node = node
            final_state = snapshot
        return final_state

    state = _run_round(state)

    while final_state.status.value == "AWAITING_HUMAN_CLARIFICATION" and round_num < 3:
        round_num += 1
        pending = final_state.clarification_questions[-1] if final_state.clarification_questions else None
        if pending is None:
            break
        answers = dict(final_state.human_answers)
        if pending.reason == "compatibility_conflict" and pending.options:
            # Deterministic demo choice: take the first offered option's part id.
            first_option = pending.options[0]
            part_id = first_option.split("(")[-1].rstrip(")") if "(" in first_option else first_option
            answers["compatibility_choice"] = part_id
        elif pending.reason in _AUTO_ANSWERS and _AUTO_ANSWERS[pending.reason]:
            key, value = _AUTO_ANSWERS[pending.reason]
            answers[key] = value
        else:
            break  # no scripted answer for this reason - stop honestly rather than guess
        state = final_state.model_copy(update={"human_answers": answers})
        final_state = _run_round(state)

    t_run_end = time.time()
    assert final_state is not None

    vision_calls = call_log.drain()
    model_info = _ollama_model_info(settings.ollama_vision_model)
    embedding_info = _embedding_model_info()

    total_prompt_tokens = sum(c.get("prompt_eval_count") or 0 for c in vision_calls)
    total_completion_tokens = sum(c.get("eval_count") or 0 for c in vision_calls)

    state_json = final_state.model_dump(mode="json")
    state_bytes = len(json.dumps(state_json).encode("utf-8"))

    record = {
        "recorded_at": time.time(),
        "job_id": job_id,
        "source_video": str(video_path.name),
        "run_wall_clock_sec": round(t_run_end - t_run_start, 2),
        "config": {
            "vision_provider": settings.vision_provider,
            "vision_model": settings.ollama_vision_model,
            "embedding_model": settings.embedding_model,
            "rag_similarity_threshold": settings.rag_similarity_threshold,
            "ocr_confidence_threshold": settings.ocr_confidence_threshold,
            "blur_variance_threshold": settings.blur_variance_threshold,
            "max_structured_output_retries": settings.max_structured_output_retries,
            "keyframe_sample_interval_sec": settings.keyframe_sample_interval_sec,
            "max_keyframes": settings.max_keyframes,
        },
        "model_info": {
            "vision_model": model_info,
            "embedding_model": embedding_info,
        },
        "agent_timeline": timeline,
        "vision_calls": vision_calls,
        "token_usage": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "call_count": len(vision_calls),
        },
        "memory_footprint": {
            "agent_state_bytes": state_bytes,
            "checkpoint_log_count": len(final_state.checkpoint_logs),
            "chat_history_count": len(final_state.chat_history),
            "retry_counts": final_state.retry_counts,
            "human_answers_count": len(final_state.human_answers),
        },
        "final_state": state_json,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = sys.argv[2] if len(sys.argv) > 2 else "demo_run_001.json"
    out_path = OUT_DIR / out_name
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Recorded run -> {out_path} ({state_bytes} bytes final_state, {len(vision_calls)} vision calls, "
          f"{total_prompt_tokens + total_completion_tokens} total tokens, "
          f"{record['run_wall_clock_sec']}s wall clock)")


if __name__ == "__main__":
    main()
