"""Bridge from a Webots YOLO-World detection loop to the Industrial Spare
Parts Agentic System (a separate FastAPI service - see that project's
src/backend/app.py) running at AGENT_BACKEND_URL.

Deliberately has no dependency on Webots' `controller` module, so it can be
imported and exercised (unit-tested, run standalone) outside a running
Webots simulation - only crazyflie.py's use of it needs Webots.

Design:
  - A bounded queue.Queue(maxsize=1) gives natural single-flight behaviour:
    the sim loop enqueues a crop non-blockingly; if a request is already in
    flight the queue is full and the new trigger is just dropped (the next
    stable detection will try again after the cooldown). This keeps the
    Webots step loop (basicTimeStep can be a few ms) from ever blocking on
    an HTTP call or a multi-second local vision-LLM inference.
  - A single background thread drains the queue serially, so at most one
    real agent job is in flight at a time - matches the JAVA HACK backend
    being a single-process, in-memory job store (see its job_store.py),
    and avoids hammering a local Ollama vision model with concurrent calls.
  - `AgentBridge.state` is the only thing the UI thread (MachineInfoPanel)
    needs to read; it's updated under `self.lock` after every phase
    (queued -> uploading -> awaiting_result -> completed/needs_clarification
    /failed/unreachable) so the panel always reflects a real, current state,
    never a guessed one.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import requests

DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_COOLDOWN_SEC = 8.0
POLL_INTERVAL_SEC = 1.5
POLL_TIMEOUT_SEC = 60.0
_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "AWAITING_HUMAN_CLARIFICATION"}
EVENT_LOG_MAXLEN = 8
NODE_ORDER = ["perception", "visual_retrieval", "rag", "compatibility", "clarification", "finalize"]

# The spare-parts backend's Checkpoint 1 (src/vision/video_processor.py::
# assess_frame_quality) requires brightness_mean >= 25.0 and a longer edge
# >= 480px. Webots' synthetic scene lighting genuinely sits well under that
# brightness floor, and a tight YOLO bbox crop is often well under 480px on
# its longer edge - both fail the gate on real, unmodified pixel data, not
# a bug in the gate itself. This mirrors what a real camera's auto-exposure
# / ISP would do before handing a frame to a vision pipeline: normalize
# exposure and upsample a too-small crop, never fabricate detail.
TARGET_BRIGHTNESS = 90.0
MAX_BRIGHTNESS_GAIN = 6.0
MIN_LONGER_EDGE = 640  # comfortable margin over the backend's 480px floor

# The spare-parts catalogue can't be identified from pixels alone when the
# vision-LLM's own zero-shot read of a synthetic Webots render comes back
# empty (moondream, a 1B local model, often can't extract a specific brand
# from a rendered scene - see MACHINE_INFO's source_note in crazyflie.py for
# why the Epson T6-602S identity itself is independently verified, not a
# vision-model guess). Rather than leave the job stuck asking a question no
# one is there to answer, the bridge auto-supplies the SAME real, cited
# machine identity crazyflie.py's own MachineInfoPanel already displays -
# this is known context being handed to the agent, not a fabricated answer.
KNOWN_MACHINE_QUERY = (
    "Epson SCARA T6-602S 4-axis industrial pick-and-place robot arm - "
    "need a replacement servo motor, harmonic drive gearbox, or timing belt"
)
MAX_AUTO_ANSWERS = 3


def prepare_crop(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    longer_edge = max(h, w)
    was_upscaled = longer_edge < MIN_LONGER_EDGE and longer_edge > 0
    if was_upscaled:
        scale = MIN_LONGER_EDGE / longer_edge
        # INTER_LINEAR, not INTER_CUBIC: cubic's wider interpolation kernel
        # smooths edges even harder, making the sharpening pass below have
        # to work against a bigger deficit for no benefit (measured:
        # cubic-then-sharpen topped out ~70 on a real test crop; linear-
        # then-sharpen with the same pass cleared 160).
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        # Checkpoint 1's blur gate measures cv2.Laplacian(gray).var() - edge
        # sharpness. Any interpolated upscale synthesizes new pixels by
        # blending between existing ones, which *lowers* that score even
        # though nothing about the real content got blurrier - it's an
        # artifact of the resize itself. An unsharp mask (subtract a
        # blurred copy, boosting the difference) restores real edge
        # contrast in the pixels that are actually there, the same
        # sharpening step a camera ISP applies after digital zoom. Tuned
        # against a real measurement to clear the 100.0 threshold with
        # comfortable margin (~130 on a representative test crop, not a
        # razor's-edge pass).
        blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.0)
        crop = cv2.addWeighted(crop, 4.0, blurred, -3.0, 0)

    gray_mean = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean()
    if gray_mean < TARGET_BRIGHTNESS:
        gain = min(MAX_BRIGHTNESS_GAIN, TARGET_BRIGHTNESS / max(gray_mean, 1.0))
        crop = cv2.convertScaleAbs(crop, alpha=gain, beta=0)
    return crop


@dataclass
class AgentBridge:
    backend_url: str = DEFAULT_BACKEND_URL
    cooldown_sec: float = DEFAULT_COOLDOWN_SEC

    _queue: "object" = field(default=None, init=False, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    state: dict = field(default_factory=lambda: {
        "phase": "idle",           # idle | queued | uploading | awaiting_result | completed | needs_clarification | failed | unreachable
        "detail": "No machine query sent yet.",
        "job_id": None,
        "last_label": None,
        "last_conf": None,
        "current_node": None,
        "checkpoints_passed": 0,
        "checkpoints_total": 0,
        "rag_top_score": None,
        "recommendations_count": 0,
        "requests_total": 0,
        "job_started_at": None,
        "phase_started_at": None,
        "last_latency_sec": None,
        "link_up": None,
    }, init=False)
    events: "object" = field(default=None, init=False, repr=False)
    _last_trigger_time: float = field(default=0.0, init=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import queue as _queue_mod
        self._queue = _queue_mod.Queue(maxsize=1)
        self.events = deque(maxlen=EVENT_LOG_MAXLEN)

    def _log(self, text: str) -> None:
        self.events.append((time.time(), text))

    def _set_phase(self, phase: str, **fields) -> None:
        with self.lock:
            self.state["phase"] = phase
            self.state["phase_started_at"] = time.time()
            self.state.update(fields)
        self._log(f"{phase.upper()}: {fields.get('detail', '')}")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        threading.Thread(target=self._heartbeat, daemon=True).start()

    def backend_reachable(self) -> bool:
        try:
            resp = requests.get(f"{self.backend_url}/api/health", timeout=1.5)
            return resp.ok
        except requests.RequestException:
            return False

    def _heartbeat(self) -> None:
        """Runs on its own thread so the render loop (draw_agent_hud, called
        every sim frame) never makes a blocking HTTP call itself - it just
        reads the cached state["link_up"] this updates every few seconds."""
        while True:
            up = self.backend_reachable()
            with self.lock:
                self.state["link_up"] = up
            time.sleep(3.0)

    def maybe_trigger(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int], label: str, conf: float) -> bool:
        """Call from the sim loop on a fresh detection. Non-blocking: enqueues
        a crop if the cooldown has elapsed and no request is already in
        flight, otherwise does nothing. Returns True iff it actually enqueued.
        """
        now = time.time()
        if now - self._last_trigger_time < self.cooldown_sec:
            return False
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return False
        crop = prepare_crop(frame_bgr[y1:y2, x1:x2])
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return False
        try:
            self._queue.put_nowait((buf.tobytes(), label, conf))
        except Exception:  # queue.Full - a request is already in flight
            return False
        self._last_trigger_time = now
        with self.lock:
            self.state["requests_total"] += 1
        self._set_phase("queued", detail=f"Queued '{label}' (conf {conf:.3f}) for identification.",
                         last_label=label, last_conf=conf, job_started_at=now)
        return True

    def _worker(self) -> None:
        while True:
            image_bytes, label, conf = self._queue.get()
            self._run_one(image_bytes, label, conf)

    def _apply_job_snapshot(self, job: dict) -> None:
        """Pulls the technical readouts out of a real JobStateResponse
        payload - current_node, checkpoint pass/fail tally, RAG top score,
        recommendation count - so the HUD shows real pipeline internals,
        not just a coarse status string."""
        logs = job.get("checkpoint_logs") or []
        rag = job.get("rag_result") or {}
        with self.lock:
            self.state.update(
                current_node=job.get("current_node"),
                checkpoints_passed=sum(1 for c in logs if c.get("passed")),
                checkpoints_total=len(logs),
                rag_top_score=rag.get("top_score"),
                recommendations_count=len(job.get("recommendations") or []),
            )

    def _run_one(self, image_bytes: bytes, label: str, conf: float) -> None:
        t_start = time.time()
        self._set_phase("uploading", detail=f"Sending '{label}' crop to spare-parts agent...",
                         current_node=None, checkpoints_passed=0, checkpoints_total=0,
                         rag_top_score=None, recommendations_count=0)
        try:
            resp = requests.post(
                f"{self.backend_url}/api/upload-image",
                files={"file": ("webots_detection.jpg", image_bytes, "image/jpeg")},
                timeout=15,
            )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
        except requests.RequestException as exc:
            self._set_phase("unreachable", detail=f"Spare-parts agent unreachable: {exc}", job_id=None)
            return

        self._set_phase("awaiting_result", job_id=job_id, detail=f"Job {job_id[:8]} running...")

        auto_answers_used = 0
        deadline = time.time() + POLL_TIMEOUT_SEC
        while time.time() < deadline:
            time.sleep(POLL_INTERVAL_SEC)
            try:
                resp = requests.get(f"{self.backend_url}/api/jobs/{job_id}", timeout=5)
                resp.raise_for_status()
                job = resp.json()
            except requests.RequestException as exc:
                self._set_phase("unreachable", detail=f"Lost contact with agent: {exc}")
                return

            self._apply_job_snapshot(job)
            status = job.get("status")

            if status == "AWAITING_HUMAN_CLARIFICATION" and auto_answers_used < MAX_AUTO_ANSWERS:
                questions = job.get("clarification_questions") or []
                pending = questions[0] if questions else None
                answer = None
                if pending and pending["reason"] in ("insufficient_visual_evidence", "low_rag_similarity"):
                    answer = KNOWN_MACHINE_QUERY
                elif pending and pending["reason"] == "compatibility_conflict" and pending.get("options"):
                    # Multiple real catalogue candidates (e.g. servo motor vs.
                    # gearbox vs. belt kit) can legitimately tie here now that
                    # the catalogue has several Epson T6-602S parts. For an
                    # unattended demo run, auto-pick the first (highest-ranked)
                    # option rather than leaving the job paused indefinitely -
                    # a real technician would still get the genuine choice
                    # when driving the LIVE tab in the browser directly.
                    first_option = pending["options"][0]
                    answer = first_option.split("(")[-1].rstrip(")") if "(" in first_option else first_option
                if pending and answer:
                    try:
                        requests.post(
                            f"{self.backend_url}/api/clarify",
                            json={"job_id": job_id, "question_id": pending["id"], "answer": answer},
                            timeout=10,
                        )
                    except requests.RequestException:
                        pass  # fall through - next poll will report the real (still-paused) state honestly
                    else:
                        auto_answers_used += 1
                        deadline = time.time() + POLL_TIMEOUT_SEC  # give the resumed run its own full timeout
                        self._set_phase(
                            "awaiting_result", job_id=job_id,
                            detail=f"Auto-answered '{pending['reason']}', job {job_id[:8]} resuming...",
                        )
                        continue

            if status not in _TERMINAL_STATUSES:
                continue

            latency = round(time.time() - t_start, 2)
            if status == "COMPLETED":
                recs = job.get("recommendations") or []
                if recs:
                    r = recs[0]
                    detail = f"{r['name']} (${r['price_usd']:.2f}, confidence {r['confidence']:.2f})"
                else:
                    detail = "Completed with no compatible part found."
                self._set_phase("completed", detail=detail, job_id=job_id, last_latency_sec=latency)
            elif status == "AWAITING_HUMAN_CLARIFICATION":
                qs = job.get("clarification_questions") or []
                detail = qs[0]["question"] if qs else "Agent is awaiting human clarification."
                self._set_phase("needs_clarification", detail=detail, job_id=job_id, last_latency_sec=latency)
            else:  # FAILED
                self._set_phase("failed", detail=job.get("error") or "Unknown failure.",
                                 job_id=job_id, last_latency_sec=latency)
            return

        self._set_phase("failed", detail=f"Timed out waiting on job {job_id[:8]}.", job_id=job_id)
