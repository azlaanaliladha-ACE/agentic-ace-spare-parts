"""FastAPI backend: video upload, async multi-agent pipeline execution,
SSE telemetry streaming, human-in-the-loop clarification, and direct
catalogue search.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.agents.graph import stream_graph
from src.backend.job_store import job_store
from src.config import settings
from src.models.schemas import (
    CatalogSearchRequest,
    CatalogSearchResponse,
    ClarifyRequest,
    ClarifyResponse,
    JobStateResponse,
    JobStatus,
    VideoUploadResponse,
)
from src.rag.retriever import search_catalog
from src.vision.video_processor import SUPPORTED_EXTENSIONS

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Industrial Spare Parts Agentic System",
    description="Multi-agent video -> spare-parts identification & recommendation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_TERMINAL_STATUSES = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.AWAITING_HUMAN_CLARIFICATION}


def _get_job_or_404(job_id: str):
    try:
        return job_store.require(job_id)
    except KeyError:
        raise HTTPException(404, f"No job with id {job_id}")


@app.on_event("startup")
def _check_catalog_ready() -> None:
    try:
        from src.rag.ingest_catalog import get_chroma_collection

        count = get_chroma_collection().count()
        if count == 0:
            logger.warning(
                "Spare parts catalogue collection is empty. Run "
                "`python -m src.rag.ingest_catalog` before uploading videos."
            )
        else:
            logger.info("Catalogue ready: %d parts indexed in '%s'.", count, settings.chroma_collection_name)
    except Exception as exc:
        logger.warning("Could not verify catalogue readiness at startup: %s", exc)


async def _run_pipeline(job_id: str) -> None:
    job = job_store.require(job_id)
    loop = asyncio.get_running_loop()

    def _worker() -> None:
        try:
            state = job.state.model_copy(update={"status": JobStatus.RUNNING_PERCEPTION, "error": None})
            loop.call_soon_threadsafe(job_store.update_state, job_id, state)
            for snapshot in stream_graph(state):
                loop.call_soon_threadsafe(job_store.update_state, job_id, snapshot)
        except Exception as exc:  # keep the pipeline crash-safe & visible to the UI
            logger.exception("Pipeline failed for job %s", job_id)
            failed = job.state.model_copy(update={"status": JobStatus.FAILED, "error": str(exc)})
            loop.call_soon_threadsafe(job_store.update_state, job_id, failed)

    await asyncio.to_thread(_worker)


@app.post("/api/upload-video", response_model=VideoUploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    existing_job_id: str | None = Form(default=None),
) -> VideoUploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    job_id = existing_job_id or uuid.uuid4().hex
    dest = settings.upload_path / f"{job_id}_{file.filename}"

    size = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024
    with open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb}MB limit.")
            out.write(chunk)

    if existing_job_id:
        job = _get_job_or_404(existing_job_id)
        new_answers = dict(job.state.human_answers)
        new_answers["video_path"] = str(dest)
        new_state = job.state.model_copy(update={"human_answers": new_answers})
        job_store.update_state(existing_job_id, new_state)
    else:
        job_store.create(job_id, file.filename or dest.name, str(dest))

    asyncio.create_task(_run_pipeline(job_id))
    job = job_store.require(job_id)
    return VideoUploadResponse(job_id=job_id, filename=job.filename, status=job.state.status)


@app.post("/api/upload-image", response_model=VideoUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    existing_job_id: str | None = Form(default=None),
) -> VideoUploadResponse:
    """Image entry path: a single photo of an unidentified spare part,
    rather than a video of a machine. Mirrors /api/upload-video exactly
    (same size limit, same existing_job_id re-upload convention for the
    low_frame_quality clarification path) but sets AgentState.image_path
    instead of video_path - see perception_agent.py::_perceive_from_image
    for how the graph branches on which one is set.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}")

    job_id = existing_job_id or uuid.uuid4().hex
    dest = settings.upload_path / f"{job_id}_{file.filename}"

    size = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024
    with open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb}MB limit.")
            out.write(chunk)

    if existing_job_id:
        job = _get_job_or_404(existing_job_id)
        new_answers = dict(job.state.human_answers)
        new_answers["image_path"] = str(dest)
        new_state = job.state.model_copy(update={"human_answers": new_answers})
        job_store.update_state(existing_job_id, new_state)
    else:
        job_store.create(job_id, file.filename or dest.name, image_path=str(dest))

    asyncio.create_task(_run_pipeline(job_id))
    job = job_store.require(job_id)
    return VideoUploadResponse(job_id=job_id, filename=job.filename, status=job.state.status)


@app.get("/api/agent/stream/{job_id}")
async def agent_stream(job_id: str) -> StreamingResponse:
    _get_job_or_404(job_id)
    queue = job_store.subscribe(job_id)

    async def event_generator():
        try:
            current = job_store.require(job_id).state
            yield f"data: {current.model_dump_json()}\n\n"
            if current.status in _TERMINAL_STATUSES:
                return
            while True:
                state = await asyncio.wait_for(queue.get(), timeout=120)
                yield f"data: {state.model_dump_json()}\n\n"
                if state.status in _TERMINAL_STATUSES:
                    break
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'error': 'stream timeout, poll /api/jobs/{job_id} instead'})}\n\n"
        finally:
            job_store.unsubscribe(job_id, queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/clarify", response_model=ClarifyResponse)
async def clarify(payload: ClarifyRequest) -> ClarifyResponse:
    job = _get_job_or_404(payload.job_id)
    question = next((q for q in job.state.clarification_questions if q.id == payload.question_id), None)
    if question is None:
        raise HTTPException(404, f"No pending question with id {payload.question_id}")

    new_answers = dict(job.state.human_answers)
    extra_updates: dict = {}

    if question.reason == "low_frame_quality":
        # Route the re-upload path to whichever entry mode this job actually
        # used - image jobs have image_path set and video_path None (see
        # upload_image()/upload_video() above). Writing unconditionally to
        # "video_path" made an image job's perception_node treat the answer
        # as a video file path (see clarification_agent.py's video_path/
        # image_path branch order), producing a bogus
        # "Video file not found: <answer>" error for image re-uploads.
        key = "image_path" if job.state.image_path and not job.state.video_path else "video_path"
        new_answers[key] = payload.answer
    elif question.reason in ("low_rag_similarity", "insufficient_visual_evidence"):
        new_answers["refined_query"] = payload.answer
    elif question.reason == "compatibility_conflict":
        match = re.search(r"\(([^)]+)\)\s*$", payload.answer)
        new_answers["compatibility_choice"] = match.group(1) if match else payload.answer.strip()
    elif question.reason == "low_ocr_confidence":
        if job.state.perception is not None:
            uncertain = [f for f in job.state.perception.uncertain_fields if "serial_tag" not in f]
            extra_updates["perception"] = job.state.perception.model_copy(
                update={"serial_number": payload.answer, "serial_confidence": 1.0, "uncertain_fields": uncertain}
            )
            # This is a non-blocking checkpoint, so the pipeline may already
            # have reached COMPLETED with a recommendation generated off the
            # low-confidence serial. Clear compatibility *and* recommendations
            # so compatibility_node and finalize_node both recompute against
            # the corrected serial instead of finalize_node short-circuiting
            # on the stale (already-populated) recommendations list.
            extra_updates["compatibility"] = None
            extra_updates["recommendations"] = []

    remaining = [q for q in job.state.clarification_questions if q.id != payload.question_id]
    new_status = JobStatus.RUNNING_PERCEPTION if question.reason != "low_ocr_confidence" else job.state.status
    new_state = job.state.model_copy(
        update={
            **extra_updates,
            "human_answers": new_answers,
            "clarification_questions": remaining,
            "status": new_status,
        }
    )
    job_store.update_state(payload.job_id, new_state)
    asyncio.create_task(_run_pipeline(payload.job_id))

    refreshed = job_store.require(payload.job_id).state
    return ClarifyResponse(job_id=payload.job_id, status=refreshed.status, remaining_questions=refreshed.clarification_questions)


@app.post("/api/catalog/search", response_model=CatalogSearchResponse)
async def catalog_search(payload: CatalogSearchRequest) -> CatalogSearchResponse:
    result = await asyncio.to_thread(search_catalog, payload.query, payload.top_k)
    return CatalogSearchResponse(query=result.query, matches=result.matches)


@app.get("/api/jobs/{job_id}", response_model=JobStateResponse)
def get_job(job_id: str) -> JobStateResponse:
    try:
        job = job_store.require(job_id)
    except KeyError:
        raise HTTPException(404, f"No job with id {job_id}")
    s = job.state
    return JobStateResponse(
        job_id=s.job_id,
        status=s.status,
        current_node=s.current_node,
        perception=s.perception,
        part_identification=s.part_identification,
        visual_search=s.visual_search,
        rag_result=s.rag_result,
        compatibility=s.compatibility,
        clarification_questions=s.clarification_questions,
        recommendations=s.recommendations,
        checkpoint_logs=s.checkpoint_logs,
        chat_history=s.chat_history,
        error=s.error,
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
