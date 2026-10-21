"""Real, continuous browser webcam video for the LIVE tab.

Streamlit has no built-in continuous video widget - st.camera_input() only
takes a single snapshot. streamlit-webrtc is the standard way to get a
genuine live camera feed inside a Streamlit app; there is no lighter-weight
alternative for true continuous video. It is pinned to an older release
(see requirements.txt) specifically because the latest release forces a
streamlit/starlette upgrade that breaks this project's pinned FastAPI
backend - see the version note there.

Design: frames from the live feed are buffered client-side in a rolling
window. "Capture & Analyze" drains that buffer, writes it to a short .mp4
(the same technique scripts/generate_demo_videos.py already uses), and
uploads it through the existing, completely unmodified
POST /api/upload-video endpoint as a brand-new job - every capture is a
real, independent run of the real LangGraph pipeline, not a simulated
result.

Honesty note: this gives on-demand clip analysis (point the camera, click
Capture, get a real pipeline run over the last few seconds), not
frame-by-frame continuous AI perception - the backend's pipeline is
architected around a complete video file (see
src/vision/video_processor.py), not a live frame stream, and this module
does not change that.
"""
from __future__ import annotations

import tempfile
import threading
from collections import deque

import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

BUFFER_SECONDS = 6
ASSUMED_FPS = 15
BUFFER_MAXLEN = BUFFER_SECONDS * ASSUMED_FPS

# Public STUN server so the browser can actually negotiate a connection
# off pure-localhost networks. Standard, widely-used default for WebRTC
# demos - not new backend infrastructure, just ICE candidate discovery.
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

MEDIA_CONSTRAINTS = {
    "video": {"width": {"ideal": 960}, "height": {"ideal": 540}},
    "audio": False,
}


def _get_buffer() -> tuple[deque, threading.Lock]:
    if "_live_frame_buffer" not in st.session_state:
        st.session_state["_live_frame_buffer"] = deque(maxlen=BUFFER_MAXLEN)
        st.session_state["_live_frame_lock"] = threading.Lock()
    return st.session_state["_live_frame_buffer"], st.session_state["_live_frame_lock"]


def _make_video_frame_callback(buffer: deque, lock: threading.Lock):
    # Runs on streamlit-webrtc's own media thread, not the Streamlit
    # script thread - must not touch st.session_state directly. Appending
    # to an already-created deque is safe (single deque.append() calls are
    # atomic under the GIL); the frame is returned unchanged so the user
    # still sees their own live feed (passthrough, not a transformation).
    def _callback(frame):
        img = frame.to_ndarray(format="bgr24")
        with lock:
            buffer.append(img)
        return frame

    return _callback


def write_clip(frames: list[np.ndarray], fps: int = ASSUMED_FPS) -> str | None:
    """Writes a list of BGR frames to a temporary .mp4 and returns its
    path, or None if there are no frames. Pure frontend helper - does not
    touch the backend.
    """
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        writer.write(f)
    writer.release()
    return tmp.name


def render_live_capture(backend_url: str, on_new_job) -> None:
    """Renders the live camera widget + capture action.

    `on_new_job(job_id, video_path)` is called once a captured clip has
    been uploaded as a new job via the existing /api/upload-video endpoint,
    so the caller can drive the same pipeline rendering used elsewhere.
    """
    buffer, lock = _get_buffer()

    ctx = webrtc_streamer(
        key="live-inspection-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints=MEDIA_CONSTRAINTS,
        video_frame_callback=_make_video_frame_callback(buffer, lock),
        async_processing=True,
    )

    if ctx.state.playing:
        st.markdown(
            '<span class="status-pill"><span class="dot dot-green"></span>Camera live</span>',
            unsafe_allow_html=True,
        )
    elif ctx.state.signalling:
        st.markdown(
            '<span class="status-pill"><span class="dot dot-amber"></span>Connecting…</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill"><span class="dot dot-gray"></span>Camera not connected</span>',
            unsafe_allow_html=True,
        )

    with lock:
        buffered_count = len(buffer)

    if ctx.state.playing:
        st.caption(f"Buffering the last ~{BUFFER_SECONDS}s of footage · {buffered_count} frame(s) held.")
    else:
        st.caption('Click "START" above to enable your camera, then capture a clip to analyze.')

    disabled = not ctx.state.playing or buffered_count == 0
    if st.button("Capture & Analyze", type="primary", disabled=disabled, key="live_capture_btn"):
        with lock:
            frames = list(buffer)
        clip_path = write_clip(frames)
        if clip_path is None:
            st.error("No frames captured yet - wait for the camera to connect first.")
            return
        try:
            with open(clip_path, "rb") as fh:
                resp = requests.post(
                    f"{backend_url}/api/upload-video",
                    files={"file": ("live_capture.mp4", fh, "video/mp4")},
                    timeout=30,
                )
            resp.raise_for_status()
            job = resp.json()
            on_new_job(job["job_id"], clip_path)
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")
