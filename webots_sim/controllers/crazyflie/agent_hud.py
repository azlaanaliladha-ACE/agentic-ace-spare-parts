"""Renders a technical-readout HUD panel for the spare-parts agent bridge
directly onto the same OpenCV window crazyflie.py already shows (the
"Crazyflie - YOLO-World Machine Detection" cv2.imshow window) - a second,
purple-accented panel alongside the existing YOLO overlay, showing the
LangGraph pipeline's real internal state (node flow, checkpoint tally, RAG
score, job id, latency, and a scrolling event log) pulled straight from
AgentBridge.state / AgentBridge.events. Every value drawn here came from a
real JobStateResponse polled from the backend - nothing on this panel is
synthesized for looks.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from spare_parts_agent_bridge import NODE_ORDER

# BGR (OpenCV order), matching the purple/blue/black palette used elsewhere
_BG = (26, 14, 10)
_BORDER = (92, 37, 52)
_VIOLET = (250, 139, 167)
_BLUE = (250, 165, 96)
_CYAN = (249, 232, 103)
_GREEN = (147, 211, 52)
_AMBER = (36, 191, 251)
_RED = (113, 113, 248)
_TEXT = (240, 233, 220)
_TEXT_DIM = (150, 130, 108)

_PHASE_COLOR = {
    "idle": _TEXT_DIM,
    "queued": _CYAN,
    "uploading": _BLUE,
    "awaiting_result": _BLUE,
    "completed": _GREEN,
    "needs_clarification": _AMBER,
    "failed": _RED,
    "unreachable": _RED,
}

_NODE_LABEL = {
    "perception": "PERCEPTION",
    "visual_retrieval": "VISUAL RETR.",
    "rag": "CATALOGUE/RAG",
    "compatibility": "COMPATIBILITY",
    "clarification": "CLARIFICATION",
    "finalize": "FINALIZE",
}

PANEL_W = 300
FONT = cv2.FONT_HERSHEY_PLAIN


def _wrap(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) > max_chars:
            if cur:
                lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines[:3]


def draw_agent_hud(frame_bgr: np.ndarray, bridge, detected_now: bool = False,
                    detection_source: str | None = None) -> None:
    """detected_now/detection_source reflect THIS FRAME's live bounding-box
    state (YOLO or ground-truth), independent of `bridge.state["phase"]` -
    the agent pipeline is an async, multi-second backend job and will
    always lag behind what's actually in frame right now. Showing both,
    clearly separated, is the point: a live perception signal plus a
    slower reasoning/decision trace, not one number pretending to be both.
    """
    h, w = frame_bgr.shape[:2]
    x0 = w - PANEL_W - 8
    y0 = 64
    panel_h = h - y0 - 8
    if x0 < 4 or panel_h < 100:
        return  # frame too small for the panel to be legible

    with bridge.lock:
        st = dict(bridge.state)
    events = list(bridge.events)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + PANEL_W, y0 + panel_h), _BG, -1)
    cv2.addWeighted(overlay, 0.78, frame_bgr, 0.22, 0, frame_bgr)
    cv2.rectangle(frame_bgr, (x0, y0), (x0 + PANEL_W, y0 + panel_h), _BORDER, 1)

    pad = 10
    cx = x0 + pad
    cy = y0 + 18

    # --- Header ---------------------------------------------------------
    cv2.putText(frame_bgr, "SPARE-PARTS AGENTIC PIPELINE", (cx, cy), FONT, 1.0, _VIOLET, 1, cv2.LINE_AA)
    cy += 14
    link_up = st.get("link_up")
    reach_txt = "LINK: UP" if link_up else ("LINK: DOWN" if link_up is False else "LINK: ?")
    reach_col = _GREEN if link_up else (_RED if link_up is False else _TEXT_DIM)
    cv2.putText(frame_bgr, reach_txt, (cx, cy), FONT, 0.9, reach_col, 1, cv2.LINE_AA)
    cy += 16

    # --- Live detection (this frame - instant, not pipeline-delayed) ------
    if detected_now:
        det_txt = f"ARM DETECTED  [{detection_source}]"
        det_col = _VIOLET if detection_source == "GROUND-TRUTH" else _GREEN
    else:
        det_txt = "ARM DETECTED  [NO]"
        det_col = _TEXT_DIM
    cv2.putText(frame_bgr, det_txt, (cx, cy), FONT, 1.05, det_col, 1, cv2.LINE_AA)
    cy += 16
    cv2.line(frame_bgr, (x0 + 4, cy), (x0 + PANEL_W - 4, cy), _BORDER, 1)
    cy += 16

    # --- Node flow (async pipeline - lags behind the live signal above) ---
    current = st.get("current_node")
    cur_idx = NODE_ORDER.index(current) if current in NODE_ORDER else -1
    for i, node in enumerate(NODE_ORDER):
        if i < cur_idx:
            dot_col, label_col = _GREEN, _TEXT_DIM
        elif i == cur_idx:
            # simple pulse using wall-clock time so it reads as "live"
            pulse = 0.5 + 0.5 * abs((time.time() * 2) % 2 - 1)
            dot_col = tuple(int(c * pulse + 20 * (1 - pulse)) for c in _VIOLET)
            label_col = _TEXT
        else:
            dot_col, label_col = (70, 60, 55), _TEXT_DIM
        cv2.circle(frame_bgr, (cx + 4, cy - 4), 4, dot_col, -1)
        cv2.putText(frame_bgr, _NODE_LABEL[node], (cx + 16, cy), FONT, 0.95, label_col, 1, cv2.LINE_AA)
        cy += 15

    cy += 4
    cv2.line(frame_bgr, (x0 + 4, cy), (x0 + PANEL_W - 4, cy), _BORDER, 1)
    cy += 16

    # --- Phase / status ----------------------------------------------------
    phase = st.get("phase", "idle")
    phase_col = _PHASE_COLOR.get(phase, _TEXT)
    cv2.putText(frame_bgr, f"PHASE: {phase.upper()}", (cx, cy), FONT, 1.05, phase_col, 1, cv2.LINE_AA)
    cy += 17

    job_id = st.get("job_id")
    job_txt = f"JOB: {job_id[:12]}" if job_id else "JOB: -"
    cv2.putText(frame_bgr, job_txt, (cx, cy), FONT, 0.95, _TEXT_DIM, 1, cv2.LINE_AA)
    cy += 14

    started = st.get("phase_started_at")
    elapsed = f"{time.time() - started:5.1f}s" if started else "  -  "
    latency = st.get("last_latency_sec")
    lat_txt = f"{latency:.2f}s" if latency is not None else "-"
    cv2.putText(frame_bgr, f"ELAPSED: {elapsed}   LAST: {lat_txt}", (cx, cy), FONT, 0.95, _TEXT_DIM, 1, cv2.LINE_AA)
    cy += 14

    ck_p, ck_t = st.get("checkpoints_passed", 0), st.get("checkpoints_total", 0)
    rag_score = st.get("rag_top_score")
    rag_txt = f"{rag_score:.3f}" if rag_score is not None else "-"
    cv2.putText(frame_bgr, f"CHECKPOINTS: {ck_p}/{ck_t}   RAG: {rag_txt}", (cx, cy), FONT, 0.95, _TEXT_DIM, 1, cv2.LINE_AA)
    cy += 14

    cv2.putText(frame_bgr, f"RECS: {st.get('recommendations_count', 0)}   "
                            f"REQUESTS: {st.get('requests_total', 0)}", (cx, cy), FONT, 0.95, _TEXT_DIM, 1, cv2.LINE_AA)
    cy += 14

    last_label = st.get("last_label")
    if last_label:
        cv2.putText(frame_bgr, f"TARGET: {last_label} ({st.get('last_conf') or 0:.3f})",
                    (cx, cy), FONT, 0.95, _TEXT_DIM, 1, cv2.LINE_AA)
        cy += 14

    cy += 6
    cv2.line(frame_bgr, (x0 + 4, cy), (x0 + PANEL_W - 4, cy), _BORDER, 1)
    cy += 14

    # --- Detail (wrapped) ---------------------------------------------------
    for line in _wrap(st.get("detail", ""), 40):
        cv2.putText(frame_bgr, line, (cx, cy), FONT, 0.95, _TEXT, 1, cv2.LINE_AA)
        cy += 13

    cy += 6
    cv2.putText(frame_bgr, "EVENT LOG", (cx, cy), FONT, 0.9, _VIOLET, 1, cv2.LINE_AA)
    cy += 13
    for ts, text in list(events)[-6:]:
        tstr = time.strftime("%H:%M:%S", time.localtime(ts))
        line = f"{tstr} {text}"[:44]
        if cy > y0 + panel_h - 6:
            break
        cv2.putText(frame_bgr, line, (cx, cy), FONT, 0.8, _TEXT_DIM, 1, cv2.LINE_AA)
        cy += 12
