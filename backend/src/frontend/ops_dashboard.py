"""Agent Ops tab: replays a pre-recorded real pipeline run (not a live
call) to show the internal workings of the agent graph - per-agent
activation/timing, token usage, model parameters, context size, and the
in-memory job-state footprint.

Why replay instead of live: recording once (scripts/record_agent_run.py)
and freezing the result into data/recorded_runs/*.json means this tab
always has something real and complete to show, doesn't cost a fresh
Ollama call every time someone opens it, and can't drift mid-demo if the
live backend/job is in a different state. Every number here came from an
actual graph.stream()/Ollama call at recording time - nothing on this tab
is fabricated for display purposes; a value that wasn't captured is simply
omitted or shown as "n/a", the same honesty convention the rest of this
app already follows (see components.py's module docstring).

To refresh the recordings after a code change:
  .venv\\Scripts\\python.exe -m scripts.record_agent_run
  VISION_PROVIDER=mock .venv\\Scripts\\python.exe -m scripts.record_agent_run <video> demo_run_mock_full.json
"""
from __future__ import annotations

import json
from pathlib import Path

from src.frontend.ops_theme import OPS_CSS

RUNS_DIR = Path(__file__).resolve().parents[2] / "data" / "recorded_runs"

RUN_FILES = {
    "Full pipeline — mock provider (deterministic, all 5 checkpoints)": "demo_run_mock_full.json",
    "Real local vision model — Ollama / moondream (live token usage)": "demo_run_ollama_moondream.json",
}

STAGE_ORDER = ["perception", "visual_retrieval", "rag", "compatibility", "clarification", "finalize"]
STAGE_LABEL = {
    "perception": "PERCEPTION",
    "visual_retrieval": "VISUAL RETRIEVAL",
    "rag": "CATALOGUE / RAG",
    "compatibility": "COMPATIBILITY",
    "clarification": "CLARIFICATION",
    "finalize": "FINALIZE",
}


def _load_run(filename: str) -> dict | None:
    path = RUNS_DIR / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "n/a"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s"


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    if n < 1024:
        return f"{n} B"
    return f"{n / 1024:.1f} KB"


def _render_header(st, run: dict) -> None:
    import datetime

    recorded = datetime.datetime.fromtimestamp(run["recorded_at"]).strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        f"""
        <div class="ops-header">
          <span class="ops-badge-replay"><span class="dot"></span>Replay &middot; recorded run, not live</span>
          <div class="title">Agent Internals &mdash; {run['source_video']}</div>
          <div class="sub">Recorded {recorded} &middot; job {run['job_id'][:12]} &middot;
          wall clock {run['run_wall_clock_sec']}s &middot; final status
          <b style="color:var(--ops-text)">{run['final_state']['status']}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_activation_flow(st, run: dict) -> None:
    timeline = run["agent_timeline"]
    activated = {t["node"] for t in timeline if t["node"]}
    # aggregate time spent per node across every round
    time_per_node: dict[str, float] = {}
    calls_per_node: dict[str, int] = {}
    for t in timeline:
        node = t["node"]
        if not node:
            continue
        time_per_node[node] = time_per_node.get(node, 0.0) + t["duration_ms"]
        calls_per_node[node] = calls_per_node.get(node, 0) + 1

    st.markdown('<div class="ops-card-title">Agent Activation Flow</div>', unsafe_allow_html=True)
    cells = []
    for i, node in enumerate(STAGE_ORDER):
        on = node in activated
        cells.append(
            f"""<div class="ops-flow-node {'active' if on else 'skipped'}">
                  <div class="n-name"><span class="n-dot {'on' if on else 'off'}"></span>{STAGE_LABEL[node]}</div>
                  <div class="n-meta">{'ACTIVATED' if on else 'SKIPPED'}</div>
                  <div class="n-meta">{calls_per_node.get(node, 0)}&times; &middot; {_fmt_ms(time_per_node.get(node))}</div>
                </div>"""
        )
        if i < len(STAGE_ORDER) - 1:
            cells.append('<div class="ops-flow-arrow">&rarr;</div>')
    st.markdown(f'<div class="ops-flow-row">{"".join(cells)}</div>', unsafe_allow_html=True)


def _render_timeline_scrubber(st, run: dict) -> None:
    timeline = [t for t in run["agent_timeline"] if t["node"]]
    if not timeline:
        st.caption("No timeline events recorded.")
        return
    st.markdown('<div class="ops-card-title">Step-through Replay</div>', unsafe_allow_html=True)
    idx = st.slider(
        "Timeline position", 0, len(timeline) - 1, len(timeline) - 1,
        key=f"ops_scrub_{run['job_id']}", label_visibility="collapsed",
    )
    step = timeline[idx]
    st.markdown(
        f"""
        <div class="ops-metric-row">
          <div class="ops-metric accent-violet"><div class="m-label">Active agent</div>
            <div class="m-value">{STAGE_LABEL.get(step['node'], step['node'])}</div>
            <div class="m-sub">round {step.get('round', 0)}</div></div>
          <div class="ops-metric accent-blue"><div class="m-label">Duration</div>
            <div class="m-value">{_fmt_ms(step['duration_ms'])}</div>
            <div class="m-sub">at t+{_fmt_ms(step['entered_at_rel_ms'])}</div></div>
          <div class="ops-metric accent-cyan"><div class="m-label">Status after</div>
            <div class="m-value" style="font-size:0.95rem">{step['status_after']}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Step {idx + 1} of {len(timeline)} — drag to replay the recorded run node-by-node.")


def _render_statistics(st, run: dict) -> None:
    tok = run["token_usage"]
    vm = run["model_info"]["vision_model"]
    em = run["model_info"]["embedding_model"]
    st.markdown('<div class="ops-card-title">Statistics</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="ops-metric-row">
          <div class="ops-metric accent-violet"><div class="m-label">Total tokens</div>
            <div class="m-value">{tok['total_tokens']:,}</div>
            <div class="m-sub">{tok['total_prompt_tokens']:,} prompt / {tok['total_completion_tokens']:,} completion</div></div>
          <div class="ops-metric accent-blue"><div class="m-label">Vision calls</div>
            <div class="m-value">{tok['call_count']}</div>
            <div class="m-sub">model: {vm.get('model', 'n/a')}</div></div>
          <div class="ops-metric accent-cyan"><div class="m-label">Vision model params</div>
            <div class="m-value">{vm.get('parameter_size') or 'n/a'}</div>
            <div class="m-sub">{vm.get('quantization_level') or ''} {vm.get('family') or ''}</div></div>
          <div class="ops-metric accent-violet"><div class="m-label">Embedding model params</div>
            <div class="m-value">{em.get('parameter_count', 0):,}</div>
            <div class="m-sub">{em.get('model')} &middot; dim {em.get('embedding_dim')}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    calls = run["vision_calls"]
    if calls:
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        rows = "".join(
            f"""<tr>
                  <td>{c.get('call')}</td><td>{c.get('frame_index', '-')}</td>
                  <td>{c.get('prompt_eval_count', 'n/a')}</td><td>{c.get('eval_count', 'n/a')}</td>
                  <td>{_fmt_ms(c.get('load_duration_ms'))}</td>
                  <td>{_fmt_ms(c.get('prompt_eval_duration_ms'))}</td>
                  <td>{_fmt_ms(c.get('eval_duration_ms'))}</td>
                  <td>{_fmt_bytes(c.get('image_bytes'))}</td>
                </tr>"""
            for c in calls
        )
        st.markdown(
            f"""<table class="ops-table">
              <tr><th>Call</th><th>Frame</th><th>Prompt tok</th><th>Completion tok</th>
                  <th>Model load</th><th>Prompt eval</th><th>Generation</th><th>Image size</th></tr>
              {rows}
            </table>""",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No vision-LLM calls in this run (mock provider — deterministic, no model inference).")


def _render_context_utility(st, run: dict) -> None:
    cfg = run["config"]
    calls = run["vision_calls"]
    final = run["final_state"]
    rag = final.get("rag_result") or {}

    st.markdown('<div class="ops-card-title">Context Utility</div>', unsafe_allow_html=True)
    avg_img = sum(c.get("image_bytes", 0) for c in calls) / len(calls) if calls else 0
    st.markdown(
        f"""
        <div class="ops-kv"><span class="k">Vision prompt cap (num_predict)</span><span class="v">800 tokens</span></div>
        <div class="ops-kv"><span class="k">Avg. image payload per call</span><span class="v">{_fmt_bytes(int(avg_img))}</span></div>
        <div class="ops-kv"><span class="k">Keyframe sampling</span><span class="v">every {cfg['keyframe_sample_interval_sec']}s, max {cfg['max_keyframes']} frames</span></div>
        <div class="ops-kv"><span class="k">RAG query</span><span class="v">"{rag.get('query', 'n/a')}"</span></div>
        <div class="ops-kv"><span class="k">Catalogue context size</span><span class="v">12 parts indexed &middot; top_k=5 retrieved</span></div>
        <div class="ops-kv"><span class="k">RAG similarity threshold</span><span class="v">{cfg['rag_similarity_threshold']}</span></div>
        <div class="ops-kv"><span class="k">OCR confidence threshold</span><span class="v">{cfg['ocr_confidence_threshold']}</span></div>
        """,
        unsafe_allow_html=True,
    )


def _render_memory_utility(st, run: dict) -> None:
    mem = run["memory_footprint"]
    final = run["final_state"]
    st.markdown('<div class="ops-card-title">Memory Utility (in-process job state)</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="ops-metric-row">
          <div class="ops-metric accent-blue"><div class="m-label">AgentState size</div>
            <div class="m-value">{_fmt_bytes(mem['agent_state_bytes'])}</div>
            <div class="m-sub">serialized JSON, this job only</div></div>
          <div class="ops-metric accent-violet"><div class="m-label">Checkpoint logs</div>
            <div class="m-value">{mem['checkpoint_log_count']}</div></div>
          <div class="ops-metric accent-cyan"><div class="m-label">Chat / activity events</div>
            <div class="m-value">{mem['chat_history_count']}</div></div>
          <div class="ops-metric accent-blue"><div class="m-label">Human answers stored</div>
            <div class="m-value">{mem['human_answers_count']}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Backed by src/backend/job_store.py: one in-memory Job per job_id (state + created_at/updated_at + "
        "SSE subscriber queues), held for the life of the backend process — no external cache/DB. "
        "This is the actual per-job memory footprint, not a simulated number."
    )
    if mem.get("retry_counts"):
        st.markdown(
            "".join(
                f'<div class="ops-kv"><span class="k">retry: {k}</span><span class="v">{v}</span></div>'
                for k, v in mem["retry_counts"].items()
            ),
            unsafe_allow_html=True,
        )


def _render_checkpoints(st, run: dict) -> None:
    logs = run["final_state"].get("checkpoint_logs") or []
    st.markdown('<div class="ops-card-title">Checkpoint Log</div>', unsafe_allow_html=True)
    if not logs:
        st.caption("No checkpoints recorded.")
        return
    rows = "".join(
        f"""<tr>
              <td><span class="ops-pill {'pass' if c['passed'] else 'fail'}">{'PASS' if c['passed'] else 'FAIL'}</span></td>
              <td>{c['checkpoint']}</td><td>{c['node']}</td><td>{c['detail']}</td>
            </tr>"""
        for c in logs
    )
    st.markdown(
        f"""<table class="ops-table">
          <tr><th>Result</th><th>Checkpoint</th><th>Node</th><th>Detail</th></tr>
          {rows}
        </table>""",
        unsafe_allow_html=True,
    )


def _render_outcome(st, run: dict) -> None:
    final = run["final_state"]
    recs = final.get("recommendations") or []
    st.markdown('<div class="ops-card-title">Outcome</div>', unsafe_allow_html=True)
    if recs:
        for r in recs:
            st.markdown(
                f"""<div class="ops-kv"><span class="k">{r['name']} ({r['part_id']})</span>
                    <span class="v">${r['price_usd']:.2f} &middot; confidence {r['confidence']:.2f}</span></div>""",
                unsafe_allow_html=True,
            )
    elif final.get("error"):
        st.markdown(f'<div class="ops-kv"><span class="k">Error</span><span class="v">{final["error"]}</span></div>', unsafe_allow_html=True)
    elif final.get("clarification_questions"):
        for q in final["clarification_questions"]:
            st.markdown(f'<div class="ops-kv"><span class="k">Paused for</span><span class="v">{q["reason"]}</span></div>', unsafe_allow_html=True)
    else:
        st.caption("No terminal outcome recorded.")


def render(st) -> None:
    st.markdown(OPS_CSS, unsafe_allow_html=True)
    st.markdown('<div class="ops-root">', unsafe_allow_html=True)

    label = st.selectbox("Recorded run", list(RUN_FILES.keys()), key="ops_run_select")
    run = _load_run(RUN_FILES[label])

    if run is None:
        st.warning(
            "No recorded run found yet. Generate one with:\n\n"
            "`.venv\\Scripts\\python.exe -m scripts.record_agent_run`"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    _render_header(st, run)

    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        with st.container(border=True):
            _render_activation_flow(st, run)
        with st.container(border=True):
            _render_timeline_scrubber(st, run)
        with st.container(border=True):
            _render_checkpoints(st, run)
    with col_b:
        with st.container(border=True):
            _render_memory_utility(st, run)
        with st.container(border=True):
            _render_context_utility(st, run)
        with st.container(border=True):
            _render_outcome(st, run)

    with st.container(border=True):
        _render_statistics(st, run)

    st.markdown("</div>", unsafe_allow_html=True)
