"""Streamlit web UI for the Industrial Spare Parts Agentic System.

Product shell: LIVE | CV | AGENTIC AI | PLAN, with AGENTIC AI as the
primary working experience (video -> perception -> catalogue search ->
compatibility -> clarification -> recommendation). Every element renders
directly from the existing FastAPI/SSE contract (`AgentState` /
`JobStateResponse`, see src/models/schemas.py) - no backend endpoint,
schema, LangGraph node, or routing rule was changed to build this.

There are two independent pipeline "contexts" that can each hold their own
running job: "agentic" (file upload, under AGENTIC AI) and "live" (webcam
capture, under LIVE - see src/frontend/live.py). Both drive the exact same
Evidence / Agent Flow / Decision rendering via render_pipeline(prefix, ...)
below; the prefix namespaces session-state keys and widget keys so the two
contexts never collide, since st.tabs renders every tab body on every run
regardless of which tab is visually active.

Run with: streamlit run src/frontend/app.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import settings  # noqa: E402
from src.frontend import components as C  # noqa: E402
from src.frontend import live as L  # noqa: E402
from src.frontend import ops_dashboard  # noqa: E402
from src.frontend.styles import inject_theme  # noqa: E402
from src.vision.video_processor import extract_keyframes  # noqa: E402

BACKEND = settings.backend_base_url.rstrip("/")

_PROVIDER_LABELS = {
    "mock": "Demo / Mock Provider",
    "anthropic": f"Anthropic · {settings.vision_model}",
    "openai": f"OpenAI · {settings.vision_model}",
    "ollama": f"Ollama · {settings.ollama_vision_model} (local)",
}

_PREFIXES = ("agentic", "live")


# --------------------------------------------------------------------------- #
# Session state
#
# Each pipeline context ("agentic", "live") gets its own namespaced set of
# keys: {prefix}_job_id, {prefix}_local_video_path, {prefix}_job_state,
# {prefix}_needs_stream, {prefix}_stream_warning. frame_cache is shared
# (keyed by video path, harmless to share across contexts).
# --------------------------------------------------------------------------- #
def sk(prefix: str, name: str) -> str:
    return f"{prefix}_{name}"


def init_session_state() -> None:
    if "frame_cache" not in st.session_state:
        st.session_state["frame_cache"] = {}
    for prefix in _PREFIXES:
        defaults = {
            sk(prefix, "job_id"): None,
            sk(prefix, "local_video_path"): None,
            sk(prefix, "local_image_path"): None,
            sk(prefix, "source_label"): None,
            sk(prefix, "job_state"): {},
            sk(prefix, "needs_stream"): False,
            sk(prefix, "stream_warning"): None,
            sk(prefix, "chat_thread"): [],
            sk(prefix, "chat_seen_markers"): set(),
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value


def reset_job(prefix: str) -> None:
    st.session_state[sk(prefix, "job_id")] = None
    st.session_state[sk(prefix, "local_video_path")] = None
    st.session_state[sk(prefix, "local_image_path")] = None
    st.session_state[sk(prefix, "source_label")] = None
    st.session_state[sk(prefix, "job_state")] = {}
    st.session_state[sk(prefix, "needs_stream")] = False
    st.session_state[sk(prefix, "stream_warning")] = None
    st.session_state[sk(prefix, "chat_thread")] = []
    st.session_state[sk(prefix, "chat_seen_markers")] = set()
    # frame_cache intentionally kept - harmless to retain across jobs


# --------------------------------------------------------------------------- #
# Backend / provider status (real checks, no fabricated "online" state)
# --------------------------------------------------------------------------- #
def check_backend_health() -> bool:
    try:
        resp = requests.get(f"{BACKEND}/api/health", timeout=1.5)
        return resp.ok
    except requests.RequestException:
        return False


def check_ollama_reachable() -> bool:
    try:
        resp = requests.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=1.5)
        return resp.ok
    except requests.RequestException:
        return False


def vision_label() -> str:
    return _PROVIDER_LABELS.get(settings.vision_provider.lower(), settings.vision_provider)


def recommendation_mode_label() -> str:
    # Mirrors the exact condition in src/agents/llm_utils.py::generate_structured_recommendation
    use_real_llm = settings.vision_provider == "anthropic" and bool(settings.anthropic_api_key)
    if use_real_llm:
        return f"Live Anthropic call ({settings.vision_model})"
    return "Deterministic template (no live LLM call for this provider configuration)"


# --------------------------------------------------------------------------- #
# Frame handling (cached client-side to avoid re-decoding the whole video
# on every slider interaction - a real perf issue in the original UI)
# --------------------------------------------------------------------------- #
def get_cached_frames(video_path: str):
    cache = st.session_state.frame_cache
    if video_path not in cache:
        cache[video_path] = extract_keyframes(video_path)
    return cache[video_path]


def draw_boxes_on_frame(frame_bgr, entities: list[dict], frame_index: int):
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    colors = {
        "machine": (0, 200, 0),
        "brand_plate": (255, 140, 0),
        "serial_tag": (0, 120, 255),
        "damaged_component": (0, 0, 255),
    }
    for e in entities:
        if e["frame_index"] != frame_index:
            continue
        bbox = e["bbox"]
        x1, y1 = int(bbox["x_min"] * w), int(bbox["y_min"] * h)
        x2, y2 = int(bbox["x_max"] * w), int(bbox["y_max"] * h)
        color = colors.get(e["entity_type"], (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{e['label']} {e['confidence']:.2f}"
        cv2.putText(img, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def render_frame_evidence(prefix: str, entities: list[dict], slider_key: str) -> None:
    idx = C.render_frame_strip(st, entities, slider_key)
    video_path = st.session_state[sk(prefix, "local_video_path")]
    if idx is None:
        return
    if not video_path:
        st.caption("Frame image unavailable (video not held in this session).")
        return
    frames = get_cached_frames(video_path)
    frame = next((f for f in frames if f.index == idx), None)
    if frame is None:
        return
    annotated = draw_boxes_on_frame(frame.image, entities, idx)
    st.image(annotated, caption=f"Frame {idx} · t={frame.timestamp_sec:.1f}s", use_column_width=True)


# --------------------------------------------------------------------------- #
# SSE streaming
# --------------------------------------------------------------------------- #
def stream_job(prefix: str, job_id: str, on_update) -> dict:
    """Consumes the existing SSE endpoint unchanged; calls on_update(state,
    running_key) after every event so callers can progressively redraw the
    Evidence / Agent Flow / Decision zones as the pipeline actually runs.
    """
    url = f"{BACKEND}/api/agent/stream/{job_id}"
    final_state = st.session_state[sk(prefix, "job_state")]
    try:
        with requests.get(url, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                payload = json.loads(raw_line[len("data: "):])
                if "error" in payload and "status" not in payload:
                    st.session_state[sk(prefix, "stream_warning")] = payload["error"]
                    break
                final_state = payload
                st.session_state[sk(prefix, "job_state")] = payload
                status = payload.get("status")
                stages = C.compute_stage_statuses(payload)
                running = C.predicted_running_key(stages) if status not in C.TERMINAL_STATUSES else None
                on_update(payload, running)
                if status in C.TERMINAL_STATUSES:
                    break
    except requests.RequestException as exc:
        st.session_state[sk(prefix, "stream_warning")] = str(exc)
    return final_state


def resolve_pending_clarification(state: dict, on_update) -> dict:
    """Defensive, frontend-only fallback for a backend race condition:
    src/agents/compatibility_agent.py can set
    status=AWAITING_HUMAN_CLARIFICATION directly on a spec conflict, one
    node before src/agents/clarification_agent.py actually runs and
    populates `clarification_questions`. Since that status is one of the
    SSE endpoint's _TERMINAL_STATUSES (src/backend/app.py), the stream can
    close a step early, leaving the client with the terminal status but no
    question text yet. This briefly polls the existing, unmodified
    GET /api/jobs/{job_id} endpoint until the question actually lands,
    without altering any backend behavior or contract.
    """
    if state.get("status") != "AWAITING_HUMAN_CLARIFICATION" or state.get("clarification_questions"):
        return state
    job_id = state.get("job_id")
    if not job_id:
        return state
    for _ in range(6):
        time.sleep(0.35)
        try:
            resp = requests.get(f"{BACKEND}/api/jobs/{job_id}", timeout=5)
            resp.raise_for_status()
            polled = resp.json()
        except requests.RequestException:
            break
        if polled.get("clarification_questions"):
            merged = {**state, **polled}
            on_update(merged, None)
            return merged
    return state


# --------------------------------------------------------------------------- #
# Shared pipeline zone renderers (used by both AGENTIC AI and LIVE)
#
# Streamlit enforces widget-key uniqueness once per script execution, not
# once per placeholder redraw - so any stateful widget (select_slider,
# radio, text_input, button, file_uploader) must be instantiated exactly
# once per run. During live SSE streaming, on_update() fires many times in
# the same run to progressively reveal results, so those calls render an
# `interactive=False` (widget-free) version; the single call after the
# stream settles renders the full interactive version.
# --------------------------------------------------------------------------- #
def show_static_frame(prefix: str, entities: list[dict], caption_suffix: str = "(latest analyzed)") -> None:
    video_path = st.session_state[sk(prefix, "local_video_path")]
    if not video_path or not entities:
        return
    latest_idx = max(e["frame_index"] for e in entities)
    frames = get_cached_frames(video_path)
    frame = next((f for f in frames if f.index == latest_idx), None)
    if frame is None:
        return
    annotated = draw_boxes_on_frame(frame.image, entities, latest_idx)
    st.image(annotated, caption=f"Frame {latest_idx} {caption_suffix}", use_column_width=True)


def draw_evidence_zone(prefix: str, state: dict, interactive: bool = True) -> None:
    with st.container(border=True):
        st.markdown('<div class="section-label">Live / Evidence</div>', unsafe_allow_html=True)
        video_path = st.session_state[sk(prefix, "local_video_path")]
        image_path = st.session_state[sk(prefix, "local_image_path")]
        if image_path:
            st.image(image_path, use_column_width=True)
        elif video_path:
            st.video(video_path)
        else:
            st.caption("No image or video loaded for this session.")

        perception = state.get("perception")
        part_identification = state.get("part_identification")
        if perception:
            entities = perception.get("detected_entities") or []
            if entities:
                if interactive:
                    render_frame_evidence(prefix, entities, f"{prefix}_frame_scrub")
                else:
                    show_static_frame(prefix, entities)
            st.divider()
            C.render_extracted_fields(st, perception)
            if part_identification:
                st.divider()
                C.render_part_identification(st, part_identification)
            st.divider()
            C.render_machine_context(
                st, perception, part_identification,
                source_label=st.session_state[sk(prefix, "source_label")] or "Unknown",
                agent_label=(state.get("status") or "idle").replace("_", " ").title(),
            )
        else:
            C.render_empty_state(
                st, "&#127909;", "Awaiting perception",
                "Machine and component evidence will appear here once frames are analyzed.",
            )


def draw_flow_zone(prefix: str, state: dict, running_key: str | None) -> None:
    with st.container(border=True):
        st.markdown('<div class="section-label">AI Agent</div>', unsafe_allow_html=True)
        C.render_agent_state_badge(st, state, running_key)
        C.render_agent_flow(st, state, running_key)
    warning = st.session_state[sk(prefix, "stream_warning")]
    if warning:
        st.warning(warning)
    with st.container(border=True):
        st.markdown('<div class="section-label">Agent Activity</div>', unsafe_allow_html=True)
        C.render_activity_timeline(st, state.get("chat_history") or [])


def draw_decision_zone(prefix: str, state: dict, interactive: bool = True) -> None:
    if state.get("part_identification") is not None:
        with st.container(border=True):
            st.markdown('<div class="section-label">Visual Candidates (MVID)</div>', unsafe_allow_html=True)
            C.render_visual_candidates(st, state.get("visual_search"))

    with st.container(border=True):
        st.markdown('<div class="section-label">Catalogue Search</div>', unsafe_allow_html=True)
        C.render_catalogue_results(st, state.get("rag_result"))
        if (state.get("rag_result") or {}).get("matches"):
            st.divider()
            st.markdown('<div class="section-label">Candidate Comparison</div>', unsafe_allow_html=True)
            C.render_candidate_comparison(st, state.get("rag_result"), state.get("compatibility"))

    with st.container(border=True):
        st.markdown('<div class="section-label">Compatibility</div>', unsafe_allow_html=True)
        C.render_compatibility_results(st, state.get("compatibility"))

    if interactive and state.get("status") == "AWAITING_HUMAN_CLARIFICATION" and state.get("clarification_questions"):
        with st.container(border=True):
            st.markdown('<div class="section-label">Clarification</div>', unsafe_allow_html=True)
            handle_clarification(prefix, state)

    with st.container(border=True):
        st.markdown('<div class="section-label">Recommendation</div>', unsafe_allow_html=True)
        C.render_recommendations(st, state.get("recommendations") or [], recommendation_mode_label())
        st.divider()
        st.markdown('<div class="section-label">Source</div>', unsafe_allow_html=True)
        C.render_source_panel(st, state.get("recommendations") or [])

    if state.get("status") == "FAILED" and state.get("error"):
        st.error(f"Pipeline failed: {state['error']}")


def handle_clarification(prefix: str, state: dict) -> None:
    for question in state.get("clarification_questions") or []:
        result = C.render_clarification_card(st, question, key_prefix=prefix)
        if result is None:
            continue
        if result["kind"] == "file":
            f = result["file"]
            is_image = result.get("is_image", False)
            endpoint = "/api/upload-image" if is_image else "/api/upload-video"
            try:
                files = {"file": (f.name, f.getvalue())}
                data = {"existing_job_id": st.session_state[sk(prefix, "job_id")]}
                resp = requests.post(f"{BACKEND}{endpoint}", files=files, data=data, timeout=60)
                resp.raise_for_status()
                path_key = "local_image_path" if is_image else "local_video_path"
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                    tmp.write(f.getbuffer())
                    st.session_state[sk(prefix, path_key)] = tmp.name
                st.session_state[sk(prefix, "needs_stream")] = True
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Re-upload failed: {exc}")
        else:
            try:
                resp = requests.post(
                    f"{BACKEND}/api/clarify",
                    json={"job_id": st.session_state[sk(prefix, "job_id")], "question_id": question["id"], "answer": result["answer"]},
                    timeout=30,
                )
                resp.raise_for_status()
                st.session_state[sk(prefix, "needs_stream")] = True
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Submitting answer failed: {exc}")


# --------------------------------------------------------------------------- #
# Chat
#
# This is a real-state-grounded assistant, not a general LLM conversation:
# the backend has no open-ended chat/intent endpoint, so we deliberately do
# not fabricate one on the frontend. Two things are genuinely real here:
#   1. When a clarification is pending, whatever the user types in chat is
#      submitted through the exact same /api/clarify call the clarification
#      card itself uses - a real backend action, not a scripted reply.
#   2. Every other reply is assembled from fields already present in the
#      real AgentState (rag_result, compatibility, recommendations,
#      clarification_questions) - never invented text, never a model call.
# Anything outside that recognized set gets an honest "not wired up yet"
# reply instead of a guessed answer.
# --------------------------------------------------------------------------- #
def _chat_append(prefix: str, role: str, content: str) -> None:
    st.session_state[sk(prefix, "chat_thread")].append({"role": role, "content": content})


def sync_chat_from_backend(prefix: str, state: dict) -> None:
    """Mirrors real, newly-appeared backend events (a clarification question,
    a recommendation, a failure) into the chat thread exactly once each,
    tracked by a marker set so reruns don't duplicate them."""
    seen = st.session_state[sk(prefix, "chat_seen_markers")]

    for q in state.get("clarification_questions") or []:
        marker = f"clarify:{q['id']}"
        if marker not in seen:
            _chat_append(prefix, "assistant", q["question"])
            seen.add(marker)

    for rec in state.get("recommendations") or []:
        marker = f"rec:{rec.get('part_id')}"
        if marker not in seen:
            _chat_append(prefix, "assistant", f"Recommendation ready: {rec.get('name')} (Part #: {rec.get('part_id')}).")
            seen.add(marker)

    if state.get("status") == "FAILED" and state.get("error"):
        marker = f"error:{state['error']}"
        if marker not in seen:
            _chat_append(prefix, "assistant", f"Something went wrong: {state['error']}")
            seen.add(marker)


def handle_chat_input(prefix: str, state: dict, text: str) -> None:
    _chat_append(prefix, "user", text)
    text_lower = text.lower().strip()
    job_id = st.session_state[sk(prefix, "job_id")]
    pending = state.get("clarification_questions") or []

    if job_id and state.get("status") == "AWAITING_HUMAN_CLARIFICATION" and pending:
        question = pending[0]
        if question["reason"] == "low_frame_quality":
            # This question needs an actual re-uploaded file, not text - the
            # backend has nowhere sensible to route free text for it (see
            # /api/clarify's low_frame_quality branch). Point the user at
            # the real re-upload widget instead of forwarding typed text as
            # a bogus file path.
            _chat_append(
                prefix, "assistant",
                "This needs an actual re-uploaded file, not a description - use the "
                "\"Re-upload\" control above (in the clarification card), not this chat box.",
            )
            return
        try:
            resp = requests.post(
                f"{BACKEND}/api/clarify",
                json={"job_id": job_id, "question_id": question["id"], "answer": text},
                timeout=30,
            )
            resp.raise_for_status()
            _chat_append(prefix, "assistant", f'Got it — using "{text}". Continuing...')
            st.session_state[sk(prefix, "needs_stream")] = True
        except requests.RequestException as exc:
            _chat_append(prefix, "assistant", f"I couldn't submit that answer: {exc}")
        return

    if not job_id:
        _chat_append(prefix, "assistant", "Upload a photo or video first, then I can help you narrow it down.")
        return

    if any(k in text_lower for k in ("why", "reject", "incompatib")):
        verdicts = (state.get("compatibility") or {}).get("verdicts") or []
        bad = [v for v in verdicts if not v.get("compatible")]
        if bad:
            v = bad[0]
            reasons = [c["field"] for c in (v.get("checks") or []) if not c.get("match")]
            reason_text = v.get("notes") or (f"failed check(s): {', '.join(reasons)}" if reasons else "did not pass compatibility validation")
            _chat_append(prefix, "assistant", f"{v.get('part_name', 'That candidate')} was marked incompatible — {reason_text}.")
        else:
            _chat_append(prefix, "assistant", "No candidates have been marked incompatible yet.")
        return

    if any(k in text_lower for k in ("alternative", "other option", "another option", "show alternatives")):
        matches = (state.get("rag_result") or {}).get("matches") or []
        if len(matches) > 1:
            alt = "; ".join(f"{m['name']} ({C.fmt_pct(m.get('score')) or '-'})" for m in matches[1:4])
            _chat_append(prefix, "assistant", f"Other catalogue candidates: {alt}.")
        else:
            _chat_append(prefix, "assistant", "I don't have additional catalogue candidates for this search yet.")
        return

    if any(k in text_lower for k in ("where can i", "find it", "buy", "supplier", "source")):
        recs = state.get("recommendations") or []
        if recs:
            _chat_append(
                prefix, "assistant",
                f"Internal catalogue: {recs[0]['name']} (Part #: {recs[0]['part_id']}). "
                "No verified supplier/source information exists in the current catalogue.",
            )
        else:
            _chat_append(prefix, "assistant", "I don't have a confirmed part yet, so I can't look up a source.")
        return

    if any(k in text_lower for k in ("what do you need", "what information", "what's missing", "what is missing")):
        _chat_append(prefix, "assistant", pending[0]["question"] if pending else "Nothing outstanding right now.")
        return

    if "compatib" in text_lower:
        compat = state.get("compatibility")
        if compat and compat.get("verdicts"):
            summary = "; ".join(
                f"{v['part_name']}: {'compatible' if v['compatible'] else 'incompatible'}" for v in compat["verdicts"]
            )
            _chat_append(prefix, "assistant", summary)
        else:
            _chat_append(prefix, "assistant", "Compatibility hasn't been checked yet.")
        return

    if "search again" in text_lower or "try again" in text_lower:
        _chat_append(
            prefix, "assistant",
            "A new search needs either a clearer photo or a more descriptive answer — "
            "I'll ask if I need one; otherwise try re-uploading.",
        )
        return

    _chat_append(
        prefix, "assistant",
        "I can answer questions about the current evidence, candidates, compatibility, and source, "
        "or take your answer if I've asked a clarifying question. I don't have a general open-ended "
        "reasoning endpoint wired up yet.",
    )


def draw_chat_zone(prefix: str, state: dict, interactive: bool = True) -> None:
    with st.container(border=True):
        st.markdown('<div class="section-label">AI Assistant</div>', unsafe_allow_html=True)
        sync_chat_from_backend(prefix, state)
        thread = st.session_state[sk(prefix, "chat_thread")]
        if thread:
            C.render_chat_messages(st, thread[-16:])
        else:
            st.caption("Ask about the current evidence, candidates, compatibility, or source once an inspection is running.")

        if interactive:
            user_text = st.chat_input("Ask the agent...", key=f"{prefix}_chat_input")
            if user_text:
                handle_chat_input(prefix, state, user_text)
                st.rerun()


def render_pipeline(prefix: str) -> None:
    """Drives + renders the Evidence / Agent Flow / Decision zones for
    whichever job is active in this context (agentic upload or live
    capture). Both tabs call this once they have a job_id.
    """
    col_evidence, col_flow, col_decision = st.columns([1, 1.15, 1.05])
    with col_evidence:
        ph_evidence = st.empty()
    with col_flow:
        ph_flow = st.empty()
    with col_decision:
        ph_decision = st.empty()
    ph_chat = st.empty()

    def draw_progress(state: dict, running_key: str | None = None) -> None:
        with ph_evidence.container():
            draw_evidence_zone(prefix, state, interactive=False)
        with ph_flow.container():
            draw_flow_zone(prefix, state, running_key)
        with ph_decision.container():
            draw_decision_zone(prefix, state, interactive=False)
        with ph_chat.container():
            draw_chat_zone(prefix, state, interactive=False)

    current_state = st.session_state[sk(prefix, "job_state")] or {"status": "PENDING"}

    if st.session_state[sk(prefix, "needs_stream")]:
        draw_progress(current_state, running_key="perception" if not current_state.get("perception") else None)
        final_state = stream_job(prefix, st.session_state[sk(prefix, "job_id")], on_update=draw_progress)
        final_state = resolve_pending_clarification(final_state, on_update=draw_progress)
        st.session_state[sk(prefix, "job_state")] = final_state
        st.session_state[sk(prefix, "needs_stream")] = False
    else:
        final_state = current_state

    # Exactly one interactive render per script execution - this is where
    # the frame-scrub slider, clarification form, and chat input live.
    with ph_evidence.container():
        draw_evidence_zone(prefix, final_state, interactive=True)
    with ph_flow.container():
        draw_flow_zone(prefix, final_state, running_key=None)
    with ph_decision.container():
        draw_decision_zone(prefix, final_state, interactive=True)
    with ph_chat.container():
        draw_chat_zone(prefix, final_state, interactive=True)


# --------------------------------------------------------------------------- #
# AGENTIC AI tab
# --------------------------------------------------------------------------- #
def render_landing() -> None:
    C.render_landing_hero(st)
    col_input, col_util = st.columns([2, 1])

    with col_input:
        with st.container(border=True):
            st.markdown('<div class="section-label">Provide Evidence</div>', unsafe_allow_html=True)
            upload_kind = st.radio(
                "Input type", ["Photo", "Video"], horizontal=True, key="upload_kind", label_visibility="collapsed"
            )

            if upload_kind == "Photo":
                uploaded = st.file_uploader(
                    "Photo of the part or machine", type=["jpg", "jpeg", "png", "webp"], key="image_uploader"
                )
            else:
                uploaded = st.file_uploader(
                    "Technician / drone video", type=["mp4", "avi", "mov", "mkv"], key="video_uploader"
                )

            st.text_input(
                "Or describe what you need",
                key="nl_query_placeholder",
                disabled=True,
                placeholder='"I have a JCB and need the seal for the boom cylinder…" — coming soon',
            )
            st.caption("Text-only input isn't wired into the agent yet; every run currently starts from a photo or footage.")

            if uploaded is not None and st.button(f"Process {upload_kind}", type="primary"):
                is_image = upload_kind == "Photo"
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getbuffer())
                    st.session_state[sk("agentic", "local_image_path" if is_image else "local_video_path")] = tmp.name
                st.session_state[sk("agentic", "source_label")] = "Uploaded Photo" if is_image else "Uploaded Video"
                endpoint = "/api/upload-image" if is_image else "/api/upload-video"
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue())}
                    resp = requests.post(f"{BACKEND}{endpoint}", files=files, timeout=60)
                    resp.raise_for_status()
                    st.session_state[sk("agentic", "job_id")] = resp.json()["job_id"]
                    st.session_state[sk("agentic", "job_state")] = {}
                    st.session_state[sk("agentic", "needs_stream")] = True
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Upload failed: {exc}")

    with col_util:
        with st.expander("Utility: direct catalogue lookup (bypasses the agent) ▸"):
            render_quick_catalogue_lookup()
        with st.expander("MVID visual-retrieval benchmark ▸"):
            C.render_mvid_benchmark_note(st)


def render_quick_catalogue_lookup() -> None:
    query = st.text_input("Search spare parts catalogue", key="catalog_query")
    top_k = st.slider("Top K", 1, 10, 5, key="catalog_topk")
    if st.button("Search catalogue", key="catalog_search_btn") and query:
        try:
            resp = requests.post(f"{BACKEND}/api/catalog/search", json={"query": query, "top_k": top_k}, timeout=30)
            resp.raise_for_status()
            matches = resp.json().get("matches", [])
            if not matches:
                st.caption("No matches.")
            for m in matches:
                st.markdown(f"**{m['name']}** ({m['brand']}) — score `{m['score']:.3f}`")
                st.caption(m["category"])
        except requests.RequestException as exc:
            st.error(f"Search failed: {exc}")


def render_agentic_tab() -> None:
    if not st.session_state[sk("agentic", "job_id")]:
        render_landing()
        return

    top_l, top_r = st.columns([4, 1])
    with top_l:
        st.caption(f"Job `{st.session_state[sk('agentic', 'job_id')][:8]}`")
    with top_r:
        if st.button("+ New inspection", key="new_inspection_btn"):
            reset_job("agentic")
            st.rerun()

    render_pipeline("agentic")

    with st.expander("Utility: direct catalogue lookup (bypasses the agent) ▸"):
        render_quick_catalogue_lookup()

    C.render_developer_drawer(
        st,
        st.session_state[sk("agentic", "job_state")] or {},
        provider_info={
            "vision_provider": settings.vision_provider,
            "vision_model": settings.vision_model,
            "ollama_vision_model": settings.ollama_vision_model,
            "embedding_model": settings.embedding_model,
            "rag_similarity_threshold": settings.rag_similarity_threshold,
            "ocr_confidence_threshold": settings.ocr_confidence_threshold,
            "blur_variance_threshold": settings.blur_variance_threshold,
            "recommendation_mode": recommendation_mode_label(),
        },
        label="Developer Details ▸",
    )


# --------------------------------------------------------------------------- #
# CV tab
# --------------------------------------------------------------------------- #
def render_cv_tab() -> None:
    st.markdown('<div class="section-label">Computer Vision Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="provider-chip">VISION &middot; {vision_label()}</span>', unsafe_allow_html=True)
    if settings.vision_provider.lower() == "mock":
        st.caption(
            "Demo/mock provider active: detections are deterministic placeholders used to exercise the "
            "pipeline end to end — not output from a trained object detector."
        )
    else:
        st.caption(
            "Detections come from a vision-language-model prompt per frame (see src/vision/perception.py), "
            "not a trained/fine-tuned object detector such as YOLO or SAM2."
        )

    state = st.session_state[sk("agentic", "job_state")]
    perception = (state or {}).get("perception")
    if not perception:
        C.render_empty_state(
            st, "&#128065;", "No CV analysis yet",
            "Run an inspection under AGENTIC AI to see frame-level detections here.",
        )
        return

    with st.container(border=True):
        col1, col2 = st.columns([1.3, 1])
        with col1:
            entities = perception.get("detected_entities") or []
            if entities:
                render_frame_evidence("agentic", entities, "cv_frame_scrub")
            else:
                st.caption("No entities detected in the sampled frames.")
        with col2:
            st.markdown('<span class="evidence-tag tag-evidence">VISUAL EVIDENCE</span>', unsafe_allow_html=True)
            usable = perception.get("usable_frame_count", 0)
            discarded = perception.get("discarded_frame_count", 0)
            st.caption(f"{usable} usable frames · {discarded} discarded for quality (Checkpoint 1)")
            C.render_extracted_fields(st, perception)

    reports = perception.get("frame_reports") or []
    if reports:
        st.markdown('<div class="section-label">Frame quality reports</div>', unsafe_allow_html=True)
        st.dataframe(
            [
                {
                    "frame": r["frame_index"],
                    "t (s)": round(r["timestamp_sec"], 2),
                    "blur variance": round(r["laplacian_variance"], 1),
                    "brightness": round(r["brightness_mean"], 1),
                    "resolution": f'{r["width"]}x{r["height"]}',
                    "passed": r["passed"],
                    "reason": r.get("reason") or "",
                }
                for r in reports
            ],
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------------------------------- #
# LIVE tab
# --------------------------------------------------------------------------- #
def render_live_tab() -> None:
    st.markdown('<div class="section-label">Live Inspection</div>', unsafe_allow_html=True)
    st.caption(
        "Real, continuous webcam video (via streamlit-webrtc). \"Capture & Analyze\" packages the last few "
        "seconds into a clip and runs it through the same, unmodified upload -> agent pipeline as AGENTIC AI "
        "- this is on-demand clip analysis, not continuous frame-by-frame AI perception."
    )

    def _on_new_job(job_id: str, video_path: str) -> None:
        st.session_state[sk("live", "job_id")] = job_id
        st.session_state[sk("live", "local_video_path")] = video_path
        st.session_state[sk("live", "source_label")] = "Local Camera Capture"
        st.session_state[sk("live", "job_state")] = {}
        st.session_state[sk("live", "needs_stream")] = True
        st.rerun()

    with st.container(border=True):
        # Honest architecture-readiness panel: real fields only, never
        # fabricated telemetry. When real drone hardware is connected in the
        # future, these same rows report real values instead.
        st.markdown(
            '<div class="live-status-grid">'
            '<span class="k">SOURCE</span><span class="v">Local Camera</span>'
            '<span class="k">DRONE</span><span class="v" style="color:var(--text-faint)">Not Connected</span>'
            '<span class="k">GPS</span><span class="v" style="color:var(--text-faint)">Unavailable</span>'
            '<span class="k">MISSION</span><span class="v">Manual Inspection</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        L.render_live_capture(BACKEND, on_new_job=_on_new_job)

    if not st.session_state[sk("live", "job_id")]:
        return

    st.divider()
    top_l, top_r = st.columns([4, 1])
    with top_l:
        st.caption(f"Capture job `{st.session_state[sk('live', 'job_id')][:8]}`")
    with top_r:
        if st.button("+ New capture", key="new_capture_btn"):
            reset_job("live")
            st.rerun()

    render_pipeline("live")

    C.render_developer_drawer(
        st,
        st.session_state[sk("live", "job_state")] or {},
        provider_info={
            "vision_provider": settings.vision_provider,
            "vision_model": settings.vision_model,
            "ollama_vision_model": settings.ollama_vision_model,
            "recommendation_mode": recommendation_mode_label(),
        },
        label="Developer Details (live capture) ▸",
    )


# --------------------------------------------------------------------------- #
# PLAN tab
# --------------------------------------------------------------------------- #
def render_plan_tab() -> None:
    st.markdown('<div class="section-label">Mission Planning</div>', unsafe_allow_html=True)
    with st.container(border=True):
        C.render_empty_state(
            st, "&#128506;", "Autonomous inspection planning is not implemented in this build.",
            "Mission planning controls will appear here once a drone autonomy module is connected.",
        )
        st.caption("Planned capabilities: inspection waypoints · GPS mission · camera capture · target identification · inspection progress")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Spare Parts Agent", layout="wide", page_icon="🛠️")
    inject_theme(st)
    init_session_state()

    backend_ok = check_backend_health()
    ollama_ok = check_ollama_reachable() if settings.vision_provider.lower() == "ollama" else None
    job_status = (
        (st.session_state[sk("agentic", "job_state")] or {}).get("status")
        or (st.session_state[sk("live", "job_state")] or {}).get("status")
    )

    C.render_header(
        st,
        backend_ok=backend_ok,
        vision_label=vision_label(),
        ollama_ok=ollama_ok,
        job_status=job_status,
    )

    tab_live, tab_cv, tab_agentic, tab_plan, tab_ops = st.tabs(
        ["LIVE", "CV", "AGENTIC AI", "PLAN", "AGENT OPS"]
    )
    with tab_live:
        render_live_tab()
    with tab_cv:
        render_cv_tab()
    with tab_agentic:
        render_agentic_tab()
    with tab_plan:
        render_plan_tab()
    with tab_ops:
        ops_dashboard.render(st)


main()
