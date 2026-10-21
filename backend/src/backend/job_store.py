"""In-memory, thread-safe job store for tracking video-processing jobs and
fanning out live agent telemetry to SSE subscribers.

A single-process in-memory store is a deliberate scope choice for this
self-hosted reference system (docker-compose spins up one backend
replica). Swap for Redis/Postgres-backed storage if scaling beyond one
process.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

from src.models.schemas import AgentState, JobStatus


@dataclass
class Job:
    job_id: str
    filename: str
    state: AgentState
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    subscribers: list[asyncio.Queue] = field(default_factory=list)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

    def create(self, job_id: str, filename: str, video_path: str | None = None, *, image_path: str | None = None) -> Job:
        state = AgentState(job_id=job_id, video_path=video_path, image_path=image_path, status=JobStatus.PENDING)
        job = Job(job_id=job_id, filename=filename, state=state)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def require(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"No job with id {job_id}")
        return job

    def update_state(self, job_id: str, state: AgentState) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.state = state
            job.updated_at = time.time()
            for queue in list(job.subscribers):
                queue.put_nowait(state.model_copy(deep=True))

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._jobs[job_id].subscribers.append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and queue in job.subscribers:
                job.subscribers.remove(queue)


job_store = JobStore()
