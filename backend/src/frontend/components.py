"""Reusable render functions for the Streamlit shell.

Every function here renders directly from the real backend payload shapes
(`AgentState` / `JobStateResponse`, see src/models/schemas.py) delivered by
the existing, unmodified FastAPI + SSE contract. Nothing here invents
fields that don't exist in that payload - if a value isn't present, the
corresponding row/card is simply omitted rather than shown as a placeholder.
"""
from __future__ import annotations

import os
from typing import Any

STAGES = [
    {
        "key": "perception",
        "number": "01",
        "title": "PERCEPTION",
        "action": "Analyzing footage and extracting machine / component information.",
    },
    {
        "key": "visual_retrieval",
        "number": "02",
        "title": "VISUAL RETRIEVAL",
        "action": "Searching the MVID visual index for similar-looking objects.",
        "image_only": True,
    },
    {
        "key": "rag",
        "number": "03",
        "title": "CATALOGUE SEARCH",
        "action": "Searching the spare-parts catalogue.",
    },
    {
        "key": "compatibility",
        "number": "04",
        "title": "COMPATIBILITY",
        "action": "Checking candidate parts against the machine and serial information.",
    },
    {
        "key": "clarification",
        "number": "05",
        "title": "CLARIFICATION",
        "action": "Additional information required.",
    },
    {
        "key": "finalize",
        "number": "06",
        "title": "RECOMMENDATION",
        "action": "Generating the final recommendation.",
    },
]

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "AWAITING_HUMAN_CLARIFICATION"}


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_pct(value: float | int | None) -> str | None:
    if value is None:
        return None
    return f"{float(value) * 100:.0f}%"


def fmt_score(value: float | int | None) -> str | None:
    if value is None:
        return None
    return f"{float(value):.2f}"


def fmt_money(value: float | int | None) -> str | None:
    if value is None:
        return None
    return f"${float(value):,.2f}"


def _latest(logs: list[dict], checkpoint: str) -> dict | None:
    matching = [c for c in logs if c.get("checkpoint") == checkpoint]
    return matching[-1] if matching else None


def _all(logs: list[dict], checkpoint: str) -> list[dict]:
    return [c for c in logs if c.get("checkpoint") == checkpoint]


# --------------------------------------------------------------------------- #
# Stage-status computation (data-driven; no chain-of-thought, no fabrication)
# --------------------------------------------------------------------------- #
def compute_stage_statuses(state: dict) -> dict[str, dict[str, Any]]:
    """Classifies each of the 5 real pipeline stages as one of:
    pending | complete | blocked | failed | (clarification only) inactive | active | resolved

    Driven entirely by which AgentState fields are already populated - this
    mirrors, but does not duplicate, the actual routing rules in
    src/agents/graph.py (it only reads outcomes, never re-decides them).
    """
    perception = state.get("perception")
    part_identification = state.get("part_identification")
    visual_search = state.get("visual_search")
    rag_result = state.get("rag_result")
    compatibility = state.get("compatibility")
    recommendations = state.get("recommendations") or []
    clarification_questions = state.get("clarification_questions") or []
    human_answers = state.get("human_answers") or {}
    status = state.get("status")
    current_node = state.get("current_node")
    error = state.get("error")
    is_image_mode = part_identification is not None

    stages: dict[str, dict[str, Any]] = {}

    # 01 perception
    if perception is not None:
        usable = perception.get("usable_frame_count", 0)
        stages["perception"] = {"status": "complete" if usable > 0 else "blocked"}
    else:
        stages["perception"] = {"status": "pending"}

    # 02 visual retrieval (MVID) - image entry path only; video jobs never
    # populate visual_search and never will (see agents/graph.py routing).
    if not is_image_mode:
        stages["visual_retrieval"] = {"status": "inactive"}
    elif visual_search is not None:
        stages["visual_retrieval"] = {"status": "complete" if visual_search.get("candidates") else "blocked"}
    elif stages["perception"]["status"] == "complete":
        stages["visual_retrieval"] = {"status": "pending"}
    else:
        stages["visual_retrieval"] = {"status": "inactive"}

    # 03 catalogue search (rag)
    if rag_result is not None:
        stages["rag"] = {"status": "complete" if rag_result.get("passed_threshold") else "blocked"}
    else:
        stages["rag"] = {"status": "pending"}

    # 03 compatibility
    if compatibility is not None:
        stages["compatibility"] = {"status": "complete" if not compatibility.get("has_conflict") else "blocked"}
    else:
        stages["compatibility"] = {"status": "pending"}

    # 04 clarification - conditional stage, only "appears" when actually invoked
    currently_blocking = status == "AWAITING_HUMAN_CLARIFICATION" and bool(clarification_questions)
    if currently_blocking:
        stages["clarification"] = {"status": "active"}
    elif human_answers:
        # The user has answered at least one clarification question in this
        # job and nothing is blocking right now - the loop was resolved.
        stages["clarification"] = {"status": "resolved"}
    else:
        stages["clarification"] = {"status": "inactive"}

    # 05 recommendation (finalize)
    if recommendations:
        stages["finalize"] = {"status": "complete"}
    else:
        stages["finalize"] = {"status": "pending"}

    # Failure overrides whichever stage the graph died in
    if status == "FAILED" and current_node in stages:
        stages[current_node]["status"] = "failed"
        stages[current_node]["error"] = error

    return stages


def predicted_running_key(stages: dict[str, dict[str, Any]]) -> str | None:
    """Best-effort UI-only lookahead: while actively streaming, the stage
    right after the last-completed one is presumed to be executing next.
    This does not affect correctness - it is purely a transient "running"
    label applied by the caller between SSE events.
    """
    if stages["clarification"]["status"] == "active":
        return None
    for key in ("perception", "visual_retrieval", "rag", "compatibility", "finalize"):
        if stages[key]["status"] == "pending":
            return key
    return None


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
def render_header(st, *, backend_ok: bool | None, vision_label: str, ollama_ok: bool | None, job_status: str | None) -> None:
    if backend_ok is None:
        backend_dot, backend_text = "dot-gray", "Backend unknown"
    elif backend_ok:
        backend_dot, backend_text = "dot-green", "Backend online"
    else:
        backend_dot, backend_text = "dot-red", "Backend unreachable"

    if ollama_ok is None:
        ollama_html = ""
    elif ollama_ok:
        ollama_html = '<span class="status-pill"><span class="dot dot-green"></span>Ollama reachable</span>'
    else:
        ollama_html = '<span class="status-pill"><span class="dot dot-red"></span>Ollama unreachable</span>'

    job_html = (
        f'<span class="status-pill"><span class="dot dot-amber"></span>Job: {job_status}</span>'
        if job_status
        else '<span class="status-pill"><span class="dot dot-gray"></span>No active job</span>'
    )

    # Status pills are joined into a single line (not one-per-line inside the
    # f-string) because a conditionally-empty segment (ollama_html == "" when
    # ollama_ok is None) would otherwise leave a whitespace-only line inside
    # this raw HTML block. Streamlit/Python-Markdown treats that blank line
    # as the end of the HTML block, so the next tag - <span>, not a
    # block-level tag - gets parsed as an indented code block and rendered
    # as literal escaped text instead of markup.
    status_row = "".join(
        [
            f'<span class="status-pill"><span class="dot {backend_dot}"></span>{backend_text}</span>',
            f'<span class="status-pill"><span class="dot dot-gray"></span>Vision: {vision_label}</span>',
            ollama_html,
            job_html,
        ]
    )

    st.markdown(
        f"""
        <div class="app-header">
          <div class="brand">
            <div class="brand-name">Industrial Inspection &amp; AI Spare-Parts Agent</div>
            <div class="brand-tag">Observe &rarr; understand &rarr; search &rarr; verify &rarr; ask &rarr; recommend</div>
          </div>
          <div class="status-row">{status_row}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Landing / empty state
# --------------------------------------------------------------------------- #
def render_landing_hero(st) -> None:
    st.markdown(
        """
        <div class="hero-panel">
          <div class="hero-title">Don't know the part number? Show us the machine.</div>
          <div class="hero-sub">
            Upload inspection footage and the agent will identify the machine and component,
            search the spare-parts catalogue, verify compatibility against what it detected,
            ask a clarifying question when it isn't confident enough to guess, and recommend
            the correct part.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(st, icon: str, title: str, sub: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="icon">{icon}</div>
          <div class="title">{title}</div>
          <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Evidence panel
# --------------------------------------------------------------------------- #
def render_extracted_fields(st, perception: dict) -> None:
    st.markdown('<span class="evidence-tag tag-evidence">EXTRACTED</span>', unsafe_allow_html=True)
    rows = []
    if perception.get("machine_brand"):
        rows.append(("Brand", perception["machine_brand"]))
    if perception.get("machine_model"):
        rows.append(("Model", perception["machine_model"]))
    if perception.get("serial_number"):
        rows.append(("Serial", perception["serial_number"]))
        rows.append(("Serial OCR confidence", fmt_pct(perception.get("serial_confidence"))))
    if not rows:
        st.caption("No brand, model, or serial resolved from this footage yet.")
    else:
        html_rows = "".join(
            f'<div class="kv-row"><span class="kv-key">{k}</span><span class="kv-val">{v}</span></div>'
            for k, v in rows
            if v
        )
        st.markdown(html_rows, unsafe_allow_html=True)

    damaged = perception.get("damaged_components") or []
    uncertain = perception.get("uncertain_fields") or []
    if damaged or uncertain:
        st.markdown('<span class="evidence-tag tag-inference">INFERENCE</span>', unsafe_allow_html=True)
        if damaged:
            st.markdown(
                f'<div class="kv-row"><span class="kv-key">Component / damage observed</span>'
                f'<span class="kv-val">{", ".join(damaged)}</span></div>',
                unsafe_allow_html=True,
            )
        if uncertain:
            st.markdown(
                f'<div class="kv-row"><span class="kv-key">Uncertain fields</span>'
                f'<span class="kv-val" style="color:var(--amber)">{", ".join(uncertain)}</span></div>',
                unsafe_allow_html=True,
            )

    usable = perception.get("usable_frame_count", 0)
    discarded = perception.get("discarded_frame_count", 0)
    total = usable + discarded
    if total:
        st.caption(f"Frame quality gate: {usable}/{total} sampled frames usable, {discarded} discarded.")


def render_part_identification(st, part_identification: dict | None) -> None:
    """Real Ollama VLM output for the image entry path (PartIdentification -
    see src/models/schemas.py). Every field is either what the model
    actually reported or omitted - never filled in with a placeholder."""
    if not part_identification:
        return
    pid = part_identification

    st.markdown('<span class="evidence-tag tag-inference">AI VISUAL ANALYSIS (Ollama)</span>', unsafe_allow_html=True)
    rows = []
    if pid.get("object_type"):
        rows.append(("Object type", pid["object_type"]))
    if pid.get("component_category"):
        rows.append(("Category", pid["component_category"]))
    if pid.get("brand_if_visible"):
        rows.append(("Brand (visible)", pid["brand_if_visible"]))
    if pid.get("machine_context"):
        rows.append(("Machine context", pid["machine_context"]))
    if pid.get("dimensions_if_visible"):
        rows.append(("Dimensions (visible)", pid["dimensions_if_visible"]))
    if rows:
        st.markdown(
            "".join(
                f'<div class="kv-row"><span class="kv-key">{k}</span><span class="kv-val">{v}</span></div>'
                for k, v in rows
            ),
            unsafe_allow_html=True,
        )
    if pid.get("visual_description"):
        st.caption(pid["visual_description"])

    if pid.get("visible_features"):
        st.markdown(
            "".join(f'<div class="checkpoint-line"><span>&#10003; {f}</span></div>' for f in pid["visible_features"]),
            unsafe_allow_html=True,
        )
    if pid.get("markings"):
        st.markdown(
            f'<div class="kv-row"><span class="kv-key">Markings</span><span class="kv-val">{", ".join(pid["markings"])}</span></div>',
            unsafe_allow_html=True,
        )

    if pid.get("part_number_if_visible"):
        conf = pid.get("part_number_confidence", 0.0)
        label = "Part number" if conf >= 0.80 else "Possible part number (unconfirmed)"
        st.markdown(
            f'<div class="kv-row"><span class="kv-key">{label}</span>'
            f'<span class="kv-val">{pid["part_number_if_visible"]} ({fmt_pct(conf)})</span></div>',
            unsafe_allow_html=True,
        )

    if pid.get("uncertainties"):
        st.markdown('<span class="evidence-tag tag-inference" style="margin-top:6px">UNCERTAIN</span>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="kv-row"><span class="kv-val" style="color:var(--amber); text-align:left">{"; ".join(pid["uncertainties"])}</span></div>',
            unsafe_allow_html=True,
        )

    if pid.get("overall_confidence") is not None:
        st.caption(f"Overall visual-analysis confidence: {fmt_pct(pid['overall_confidence'])}")


def render_frame_strip(st, entities: list[dict], selected_key: str) -> int | None:
    frame_indices = sorted({e["frame_index"] for e in entities})
    if not frame_indices:
        return None
    return st.select_slider("Scrub analyzed frames", options=frame_indices, key=selected_key)


# --------------------------------------------------------------------------- #
# Agent flow
# --------------------------------------------------------------------------- #
def render_stage_card(st, stage: dict, computed: dict, checkpoint_logs: list[dict], is_last: bool, state: dict | None = None) -> None:
    status = computed["status"]
    key = stage["key"]

    if status == "complete":
        badge, css_cls = '<span class="badge badge-pass">&#10003; DONE</span>', "stage-complete"
    elif status == "running":
        badge, css_cls = '<span class="badge badge-idle">&#8226; RUNNING</span>', "stage-running"
    elif status == "blocked":
        badge, css_cls = '<span class="badge badge-wait">&#9888; NEEDS INPUT</span>', "stage-blocked"
    elif status == "failed":
        badge, css_cls = '<span class="badge badge-fail">&#10007; FAILED</span>', "stage-failed"
    elif status == "active":
        badge, css_cls = '<span class="badge badge-wait">&#9888; WAITING FOR YOU</span>', "stage-blocked"
    elif status == "resolved":
        badge, css_cls = '<span class="badge badge-pass">&#10003; RESOLVED</span>', "stage-complete"
    elif status == "inactive":
        badge, css_cls = '<span class="badge badge-idle">NOT NEEDED</span>', "stage-pending"
    else:
        badge, css_cls = '<span class="badge badge-idle">PENDING</span>', "stage-pending"

    action_text = stage["action"] if status not in ("inactive",) else "Only appears if the agent needs more information."

    html = [f'<div class="{css_cls}">']
    html.append(
        f'<div class="stage-head"><span class="stage-num">{stage["number"]}</span>'
        f'<span class="stage-title">{stage["title"]}</span>{badge}</div>'
    )
    html.append(f'<div class="stage-action">{action_text}</div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    for line in _checkpoint_lines(key, checkpoint_logs, state or {}):
        st.markdown(line, unsafe_allow_html=True)

    if status == "failed" and computed.get("error"):
        st.markdown(
            f'<div class="checkpoint-line" style="color:var(--red)">Reason: {computed["error"]}</div>',
            unsafe_allow_html=True,
        )

    if not is_last:
        st.markdown('<div class="stage-connector"></div>', unsafe_allow_html=True)


def _checkpoint_lines(stage_key: str, logs: list[dict], state: dict) -> list[str]:
    lines: list[str] = []

    if stage_key == "visual_retrieval":
        vs = state.get("visual_search")
        if vs:
            candidates = vs.get("candidates") or []
            if candidates:
                top = candidates[0]
                lines.append(
                    f'<div class="checkpoint-line"><span>MVID index ({vs.get("embedding_model", "")})</span>'
                    f'<span class="mark-pass">&#10003; {len(candidates)} candidate(s), top {fmt_pct(top.get("similarity"))}</span></div>'
                )
            else:
                lines.append(
                    '<div class="checkpoint-line"><span>MVID index</span>'
                    '<span class="mark-fail">&#9888; no candidates found</span></div>'
                )

    elif stage_key == "perception":
        fq = _latest(logs, "frame_quality")
        if fq:
            mark, cls = ("&#10003;", "mark-pass") if fq["passed"] else ("&#9888;", "mark-fail")
            lines.append(
                f'<div class="checkpoint-line"><span>Frame quality</span>'
                f'<span class="{cls}">{mark} {fq["detail"]}</span></div>'
            )
        ocr = _all(logs, "ocr_confidence")
        if ocr:
            passed = sum(1 for c in ocr if c["passed"])
            mark, cls = ("&#10003;", "mark-pass") if passed == len(ocr) else ("&#9888;", "mark-fail")
            lines.append(
                f'<div class="checkpoint-line"><span>OCR / entity confidence</span>'
                f'<span class="{cls}">{mark} {passed}/{len(ocr)} recovered above threshold</span></div>'
            )

    elif stage_key == "rag":
        rs = _latest(logs, "rag_similarity")
        if rs:
            mark, cls = ("&#10003;", "mark-pass") if rs["passed"] else ("&#9888;", "mark-fail")
            lines.append(
                f'<div class="checkpoint-line"><span>Similarity threshold</span>'
                f'<span class="{cls}">{mark} {rs["detail"]}</span></div>'
            )

    elif stage_key == "compatibility":
        cp = _latest(logs, "compatibility")
        if cp:
            mark, cls = ("&#10003;", "mark-pass") if cp["passed"] else ("&#9888;", "mark-fail")
            lines.append(
                f'<div class="checkpoint-line"><span>Compatibility validation</span>'
                f'<span class="{cls}">{mark} {cp["detail"]}</span></div>'
            )

    elif stage_key == "finalize":
        so = _all(logs, "structured_output")
        if so:
            passed = sum(1 for c in so if c["passed"])
            mark, cls = ("&#10003;", "mark-pass") if passed else ("&#9888;", "mark-fail")
            lines.append(
                f'<div class="checkpoint-line"><span>Structured output</span>'
                f'<span class="{cls}">{mark} {passed}/{len(so)} attempts passed schema validation</span></div>'
            )

    return lines


def render_agent_flow(st, state: dict, running_key: str | None = None) -> dict[str, dict[str, Any]]:
    stages = compute_stage_statuses(state)
    checkpoint_logs = state.get("checkpoint_logs") or []

    # Video-mode jobs never populate visual_search (see agents/graph.py) -
    # hide that stage entirely rather than show an irrelevant "NOT NEEDED"
    # row, so the existing video flow's card list is visually unchanged.
    visible = [
        s for s in STAGES if not (s.get("image_only") and stages[s["key"]]["status"] == "inactive")
    ]

    for i, stage in enumerate(visible):
        computed = dict(stages[stage["key"]])
        if running_key == stage["key"] and computed["status"] == "pending":
            computed["status"] = "running"
        render_stage_card(st, stage, computed, checkpoint_logs, is_last=(i == len(visible) - 1), state=state)

    return stages


# --------------------------------------------------------------------------- #
# Agent state badge + activity timeline (real AgentState.chat_history, real
# timestamps - action summaries the backend already writes per node, never
# hidden chain-of-thought and never invented on the frontend)
# --------------------------------------------------------------------------- #
_RUNNING_STATE_LABELS = {
    "perception": ("ANALYSING", "Looking at the component and extracting visible details..."),
    "visual_retrieval": ("SEARCHING", "Comparing against the visual retrieval index..."),
    "rag": ("SEARCHING", "Searching the spare-parts catalogue..."),
    "compatibility": ("VERIFYING", "Checking machine and serial compatibility..."),
    "finalize": ("RECOMMENDING", "Preparing the recommendation..."),
}


def render_agent_state_badge(st, state: dict, running_key: str | None) -> None:
    status = state.get("status")
    if status == "FAILED":
        label, detail, css = "ERROR", state.get("error") or "The pipeline failed.", "agent-state-error"
    elif status == "AWAITING_HUMAN_CLARIFICATION":
        label, detail, css = "CLARIFYING", "Waiting for your answer to continue.", "agent-state-wait"
    elif status == "COMPLETED":
        label, detail, css = "COMPLETE", "Recommendation ready.", "agent-state-done"
    elif running_key and running_key in _RUNNING_STATE_LABELS:
        label, detail = _RUNNING_STATE_LABELS[running_key]
        css = "agent-state-active"
    elif status in (None, "PENDING"):
        label, detail, css = "IDLE", "Waiting for an image, video, or live capture.", "agent-state-idle"
    else:
        label, detail, css = "ANALYSING", "Processing the latest evidence...", "agent-state-active"

    st.markdown(
        f'<div class="agent-state-badge {css}">'
        f'<span class="agent-state-dot"></span>'
        f'<div><div class="agent-state-label">{label}</div>'
        f'<div style="font-size:0.78rem;color:var(--text-dim)">{detail}</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_activity_timeline(st, chat_history: list[dict], limit: int = 12) -> None:
    """Renders the real, already-timestamped node-status trail every agent
    already writes to AgentState.chat_history (e.g. "Perception complete:
    identified...", "Catalogue search for '...' -> top candidates: ...").
    These are concise action summaries, not hidden model reasoning - no text
    here is generated by this function, only formatted.
    """
    if not chat_history:
        st.caption("No agent activity yet.")
        return

    from datetime import datetime

    entries = list(chat_history)[-limit:][::-1]
    html = []
    for c in entries:
        ts = c.get("timestamp")
        time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""
        prefix = "you" if c.get("role") == "user" else "agent"
        html.append(
            f'<div class="timeline-item"><span class="timeline-time">{time_str}</span>'
            f'<span class="timeline-text"><b style="color:var(--text-faint)">{prefix}</b> &middot; {c.get("content", "")}</span></div>'
        )
    st.markdown("".join(html), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Catalogue results (previously computed but never shown)
# --------------------------------------------------------------------------- #
def render_catalogue_results(st, rag_result: dict | None) -> None:
    if rag_result is None:
        render_empty_state(st, "&#128269;", "No catalogue search yet", "Results appear once the machine/component has been identified.")
        return

    matches = rag_result.get("matches") or []
    st.caption(f"Query: “{rag_result.get('query', '')}” — {len(matches)} candidate(s) found")

    if not matches:
        st.markdown(
            '<div class="checkpoint-line" style="color:var(--amber)">&#9888; No sufficiently similar parts found in the catalogue.</div>',
            unsafe_allow_html=True,
        )
        return

    best_score = matches[0].get("score", 0.0)
    for i, m in enumerate(matches):
        is_best = i == 0
        meta = m.get("metadata") or {}
        models = meta.get("compatible_models") or []
        card_cls = "catalogue-card best" if is_best else "catalogue-card"
        best_tag = '<div style="margin-top:6px"><span class="badge badge-pass">BEST MATCH</span></div>' if is_best else ""
        models_line = f'<div class="catalogue-meta">Compatible: {", ".join(models)}</div>' if models else ""
        st.markdown(
            f"""
            <div class="{card_cls}">
              <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <span class="catalogue-name">{m.get('name', 'Unknown part')}</span>
                <span class="catalogue-score">{fmt_pct(m.get('score'))}</span>
              </div>
              <div class="catalogue-meta">{m.get('brand', '')} &middot; {m.get('category', '')} &middot; Part #: {m.get('part_id', '')}</div>
              {models_line}
              {best_tag}
            </div>
            """,
            unsafe_allow_html=True,
        )
    _ = best_score


def render_candidate_comparison(st, rag_result: dict | None, compatibility: dict | None) -> None:
    """Joins real rag_result matches with real compatibility verdicts (both
    already computed by the unmodified backend) into one explainable,
    per-signal table - never a single blended score. Status is derived from
    simple, visible rules (exact ID > confirmed-compatible > confirmed-
    incompatible > unverified), not an invented weighting formula.
    """
    matches = (rag_result or {}).get("matches") or []
    if not matches:
        return
    verdicts_by_id = {v["part_id"]: v for v in ((compatibility or {}).get("verdicts") or [])}

    st.markdown(
        '<div class="fusion-row header"><span>Candidate</span><span>Text sim.</span>'
        "<span>Part #</span><span>Model</span><span>Serial</span><span>Status</span></div>",
        unsafe_allow_html=True,
    )
    for m in matches:
        verdict = verdicts_by_id.get(m["part_id"])
        text_sim = fmt_pct(m.get("score")) or "-"
        part_num_mark = "&#10003;" if m.get("match_type") == "exact_part_number" else "-"

        model_mark, serial_mark = "-", "-"
        for c in (verdict or {}).get("checks") or []:
            mark = "&#10003;" if c.get("match") else "&#10007;"
            if c.get("field") == "compatible_models":
                model_mark = mark
            elif c.get("field") == "serial_range":
                serial_mark = mark

        if m.get("match_type") == "exact_part_number":
            status_label, status_css = "EXACT MATCH", "exact"
        elif verdict and verdict.get("compatible"):
            status_label, status_css = "LIKELY MATCH", "likely"
        elif verdict and not verdict.get("compatible"):
            status_label, status_css = "INCOMPATIBLE", "incompatible"
        else:
            status_label, status_css = "UNVERIFIED", "unverified"

        st.markdown(
            f'<div class="fusion-row"><span>{m.get("name", "")}</span><span>{text_sim}</span>'
            f'<span>{part_num_mark}</span><span>{model_mark}</span><span>{serial_mark}</span>'
            f'<span class="fusion-status {status_css}">{status_label}</span></div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Visual retrieval results (MVID proof-of-concept index - real DINOv2 +
# ChromaDB nearest-neighbour search; never presented as a catalogue part)
# --------------------------------------------------------------------------- #
def render_visual_candidates(st, visual_search: dict | None) -> None:
    if visual_search is None:
        render_empty_state(
            st, "&#128247;", "No visual retrieval yet",
            "Runs automatically for image uploads once perception completes.",
        )
        return

    candidates = visual_search.get("candidates") or []
    st.caption(
        f"MVID visual-retrieval proof of concept &middot; {visual_search.get('embedding_model', '')} "
        f"&middot; {len(candidates)} candidate(s)"
    )
    if not candidates:
        st.markdown(
            '<div class="checkpoint-line" style="color:var(--amber)">&#9888; No visual neighbours found in the MVID index.</div>',
            unsafe_allow_html=True,
        )
        return

    for i, c in enumerate(candidates):
        is_best = i == 0
        card_cls = "mvid-card best" if is_best else "mvid-card"
        img_path = c.get("image_path")
        col_img, col_info = st.columns([1, 3])
        with col_img:
            if img_path and os.path.exists(img_path):
                st.image(img_path, use_column_width=True)
        with col_info:
            best_tag = '<div style="margin-top:4px"><span class="badge badge-pass">TOP MATCH</span></div>' if is_best else ""
            st.markdown(
                f"""
                <div class="{card_cls}">
                  <span class="mvid-tag">MVID visual candidate</span>
                  <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span class="catalogue-name">{c.get('object_id', '')} / view {c.get('view_id', '')}</span>
                    <span class="mvid-score">{fmt_pct(c.get('similarity'))}</span>
                  </div>
                  <div class="catalogue-meta">Source: {c.get('source', 'MVID')} proof-of-concept index - not our spare-parts catalogue</div>
                  {best_tag}
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# Compatibility results (previously computed but never shown)
# --------------------------------------------------------------------------- #
_CHECK_LABELS = {
    "compatible_models": "Machine model",
    "serial_range": "Serial range",
}


def render_compatibility_results(st, compatibility: dict | None) -> None:
    if compatibility is None:
        render_empty_state(st, "&#128203;", "No compatibility check yet", "Runs once catalogue candidates are found.")
        return

    if compatibility.get("has_conflict") and compatibility.get("conflict_reason"):
        st.markdown(
            f'<div class="checkpoint-line" style="color:var(--amber)">&#9888; {compatibility["conflict_reason"]}</div>',
            unsafe_allow_html=True,
        )

    verdicts = compatibility.get("verdicts") or []
    if not verdicts:
        st.caption("No candidates were validated.")
        return

    st.caption("Rule-based compatibility validation — deterministic checks, not a model guess.")
    for v in verdicts:
        status_badge = '<span class="badge badge-pass">&#10003; COMPATIBLE</span>' if v.get("compatible") else '<span class="badge badge-fail">&#10007; INCOMPATIBLE</span>'
        checks_html = ""
        for c in v.get("checks") or []:
            label = _CHECK_LABELS.get(c.get("field"), c.get("field"))
            mark, cls = ("&#10003;", "mark-pass") if c.get("match") else ("&#10007;", "mark-fail")
            checks_html += f'<div class="compat-check"><span class="key">{label}</span><span class="{cls}">{mark}</span></div>'
        notes_html = f'<div class="catalogue-meta" style="margin-top:4px">{v["notes"]}</div>' if v.get("notes") else ""
        st.markdown(
            f"""
            <div class="catalogue-card">
              <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <span class="catalogue-name">{v.get('part_name', '')}</span>
                {status_badge}
              </div>
              {checks_html}
              {notes_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Clarification
# --------------------------------------------------------------------------- #
def render_clarification_card(st, question: dict, key_prefix: str = "") -> dict | None:
    """Renders the clarification prompt + input widget. Returns
    {"answer": ..., "kind": "text"|"file"} once the user submits, else None.

    `key_prefix` namespaces widget keys - needed because more than one
    independent pipeline context (e.g. AGENTIC AI vs. a LIVE capture) can be
    on screen in the same script run.
    """
    st.markdown(
        f"""
        <div class="clarify-card">
          <div class="clarify-title">&#9888; INFORMATION REQUIRED</div>
          <div class="clarify-question">{question['question']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    answer = None
    kind = "text"
    reup_file = None
    k = f"{key_prefix}_{question['id']}" if key_prefix else question["id"]

    is_image_reupload = question["reason"] == "low_frame_quality" and question.get("field") == "image_path"

    if question.get("options"):
        answer = st.radio("Choose one", question["options"], key=f"opt_{k}", label_visibility="collapsed")
    elif question["reason"] == "low_frame_quality":
        # Which file types to offer depends on entry mode - an image job's
        # question has field="image_path" (see clarification_agent.py), a
        # video job's has field="video_path". Previously this always
        # offered only video extensions, so a photo job's re-upload prompt
        # had no way to actually accept a replacement photo.
        if is_image_reupload:
            reup_file = st.file_uploader(
                "Re-upload a clearer photo", type=["jpg", "jpeg", "png", "webp"], key=f"reup_{k}"
            )
        else:
            reup_file = st.file_uploader(
                "Re-upload clearer footage", type=["mp4", "avi", "mov", "mkv"], key=f"reup_{k}"
            )
        kind = "file"
    else:
        answer = st.text_input("Your answer", key=f"txt_{k}", label_visibility="collapsed")

    submitted = st.button("Continue", key=f"submit_{k}", type="primary")
    if not submitted:
        return None

    if kind == "file":
        if reup_file is None:
            st.warning(f"Choose a {'photo' if is_image_reupload else 'video'} file first.")
            return None
        return {"kind": "file", "file": reup_file, "is_image": is_image_reupload}

    if not answer:
        st.warning("Enter an answer first.")
        return None
    return {"kind": "text", "answer": answer}


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
def render_recommendations(st, recommendations: list[dict], recommendation_mode: str) -> None:
    if not recommendations:
        render_empty_state(st, "&#128230;", "No recommendation yet", "The recommended part appears once compatibility is resolved.")
        return

    for rec in recommendations:
        price_html = f'<div class="recommend-price">{fmt_money(rec.get("price_usd"))}</div>' if rec.get("price_usd") is not None else ""
        st.markdown(
            f"""
            <div class="recommend-hero">
              <div class="recommend-label">Recommended Spare Part</div>
              <div class="recommend-name">{rec.get('name', '')}</div>
              <div class="kv-row"><span class="kv-key">Part Number</span><span class="kv-val">{rec.get('part_id', '')}</span></div>
              <div class="kv-row"><span class="kv-key">Brand</span><span class="kv-val">{rec.get('brand', '')}</span></div>
              {price_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if rec.get("rationale"):
            st.markdown("**Why this part?**")
            st.write(rec["rationale"])
        if rec.get("confidence") is not None:
            conf = min(1.0, max(0.0, float(rec["confidence"])))
            st.progress(conf, text=f"Recommendation confidence: {conf * 100:.0f}%")
        st.caption(f"Recommendation text source: {recommendation_mode}")


# --------------------------------------------------------------------------- #
# Source / supplier panel - distinct from catalogue identification. Today
# this is ALWAYS "not verified": the catalogue has no supplier/source field
# and no external source-lookup tool is implemented (confirmed by repo
# inspection) - so this function can only ever honestly report that, never
# a fabricated supplier/link. Structured now so a real source feed can
# populate it later without changing callers.
# --------------------------------------------------------------------------- #
def render_source_panel(st, recommendations: list[dict]) -> None:
    if not recommendations:
        st.caption("Source lookup runs once a recommendation is confirmed.")
        return

    for rec in recommendations:
        st.markdown(
            f"""
            <div class="source-unverified">
              <b>Internal catalogue:</b> {rec.get('name', '')} (Part #: {rec.get('part_id', '')})<br/>
              <b>Verified supplier/source:</b> Not available.<br/>
              No verified supplier/source information exists in the current catalogue.
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Machine / inspection context strip
# --------------------------------------------------------------------------- #
def render_machine_context(st, perception: dict | None, part_identification: dict | None, source_label: str, agent_label: str) -> None:
    machine = None
    if perception:
        brand = perception.get("machine_brand")
        model = perception.get("machine_model")
        machine = " ".join(filter(None, [brand, model])) or None
    component = None
    if part_identification:
        component = part_identification.get("component_category") or part_identification.get("object_type")
    elif perception and perception.get("damaged_components"):
        component = ", ".join(perception["damaged_components"])

    rows = [
        ("MACHINE", machine or "Unknown"),
        ("COMPONENT", component or "Unknown"),
        ("SOURCE", source_label),
        ("AGENT", agent_label),
    ]
    st.markdown(
        '<div class="live-status-grid">' + "".join(f'<span class="k">{k}</span><span class="v">{v}</span>' for k, v in rows) + "</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# MVID benchmark note - a real, measured, one-time offline evaluation
# result (see src/rag/mvid_evaluate.py). Deliberately NOT rendered inside
# the main operational flow, and explicitly labeled as a visual-retrieval
# benchmark, never as spare-part identification accuracy.
# --------------------------------------------------------------------------- #
def render_mvid_benchmark_note(st) -> None:
    st.markdown(
        """
        <div class="benchmark-note">
        <b>MVID multi-view industrial-object retrieval benchmark</b> (offline evaluation, 2,520/2,520 real images, 7 objects)<br/>
        Recall@1: 100% &middot; Recall@5: 100% &middot; Recall@10: 100% &middot; MRR: 1.0 &middot; Avg. query latency: 39.2 ms<br/>
        This measures DINOv2's ability to retrieve other views of the <i>same physical MVID object</i> -
        it is not spare-part identification accuracy, and MVID contains no JCB/CAT/Parker parts.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Chat (native st.chat_message bubbles; message routing/backend calls live
# in app.py, consistent with this file staying pure-render)
# --------------------------------------------------------------------------- #
def render_chat_messages(st, messages: list[dict]) -> None:
    """messages: list of {"role": "user"|"assistant", "content": str}."""
    for m in messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])


# --------------------------------------------------------------------------- #
# Developer / debug drawer
# --------------------------------------------------------------------------- #
def render_developer_drawer(st, state: dict, provider_info: dict, label: str = "Developer Details ▸") -> None:
    with st.expander(label, expanded=False):
        st.markdown("**Active configuration (frontend process env)**")
        st.json(provider_info)
        st.markdown("**Checkpoint log trail**")
        st.json(state.get("checkpoint_logs") or [])
        st.markdown("**Chat / event history**")
        st.json(state.get("chat_history") or [])
        st.markdown("**Raw AgentState snapshot**")
        st.json(state)
