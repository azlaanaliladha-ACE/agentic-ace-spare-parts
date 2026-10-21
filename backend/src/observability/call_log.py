"""In-process side-channel recorder for vision-LLM call statistics.

Not part of AgentState or any API contract - this exists solely so
scripts/record_agent_run.py can capture real token/timing numbers (from
Ollama's /api/generate response fields) alongside the real AgentState
produced by a real graph run, and bundle both into a static JSON fixture
that the Agent Ops dashboard tab replays. Thread-safe because the backend
may run jobs off the main thread.
"""
from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_calls: list[dict] = []


def record(node: str, **fields) -> None:
    with _lock:
        _calls.append({"node": node, "wall_time": time.time(), **fields})


def drain() -> list[dict]:
    with _lock:
        calls = list(_calls)
        _calls.clear()
    return calls
