"""Shared Pydantic data contracts for the spare-parts agentic system.

These models are used as (a) the LangGraph `AgentState` schema, (b) the
structured-output targets that vision/LLM calls are validated against
(Checkpoint 5), and (c) the FastAPI request/response bodies.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING_VIDEO = "PROCESSING_VIDEO"
    RUNNING_PERCEPTION = "RUNNING_PERCEPTION"
    RUNNING_RAG = "RUNNING_RAG"
    RUNNING_COMPATIBILITY = "RUNNING_COMPATIBILITY"
    AWAITING_HUMAN_CLARIFICATION = "AWAITING_HUMAN_CLARIFICATION"
    GENERATING_RECOMMENDATION = "GENERATING_RECOMMENDATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EntityType(str, Enum):
    MACHINE = "machine"
    BRAND_PLATE = "brand_plate"
    SERIAL_TAG = "serial_tag"
    DAMAGED_COMPONENT = "damaged_component"


# --------------------------------------------------------------------------- #
# Vision / Perception (Agent 1)
# --------------------------------------------------------------------------- #
class BoundingBox(BaseModel):
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)


class DetectedEntity(BaseModel):
    entity_type: EntityType
    label: str
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    frame_index: int
    raw_text: str | None = None  # OCR text for plates/tags


class FrameQualityReport(BaseModel):
    frame_index: int
    timestamp_sec: float
    laplacian_variance: float
    brightness_mean: float
    width: int
    height: int
    passed: bool
    reason: str | None = None


class PerceptionResult(BaseModel):
    frame_reports: list[FrameQualityReport] = Field(default_factory=list)
    detected_entities: list[DetectedEntity] = Field(default_factory=list)
    machine_brand: str | None = None
    machine_model: str | None = None
    serial_number: str | None = None
    serial_confidence: float = 0.0
    damaged_components: list[str] = Field(default_factory=list)
    uncertain_fields: list[str] = Field(default_factory=list)
    usable_frame_count: int = 0
    discarded_frame_count: int = 0


class PartIdentification(BaseModel):
    """Structured single-image spare-part identification (Agent 1, image
    entry path). A real vision-model call fills this in from actual pixels -
    every field is None/empty when the model reports it isn't visible,
    never guessed. See src/vision/perception.py::PART_IDENTIFICATION_INSTRUCTIONS.
    """

    object_type: str | None = None
    component_category: str | None = None
    visual_description: str | None = None
    visible_features: list[str] = Field(default_factory=list)
    dimensions_if_visible: str | None = None
    markings: list[str] = Field(default_factory=list)
    part_number_if_visible: str | None = None
    part_number_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    brand_if_visible: str | None = None
    machine_context: str | None = None
    uncertainties: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# Visual retrieval (MVID proof-of-concept image index)
#
# IMPORTANT: MVID is a real-photo instance-retrieval dataset used ONLY to
# prove the image -> embedding -> vector-search mechanism works end to end.
# It contains no JCB/CAT/Parker parts, no part numbers, no supplier data,
# and no compatibility information. A VisualCandidate is never a catalogue
# identity - match_type is always "visual_candidate" here, deliberately
# distinct from CatalogMatch.match_type ("exact_part_number" | "semantic"),
# so the two can never be confused downstream.
# --------------------------------------------------------------------------- #
class VisualCandidate(BaseModel):
    dataset: Literal["MVID"] = "MVID"
    object_id: str
    image_path: str
    view_id: str
    source: Literal["MVID"] = "MVID"
    similarity: float
    match_type: Literal["visual_candidate"] = "visual_candidate"


class VisualSearchResult(BaseModel):
    query_image_path: str
    candidates: list[VisualCandidate] = Field(default_factory=list)
    embedding_model: str
    index_collection: str


# --------------------------------------------------------------------------- #
# RAG / Catalogue (Agent 2)
# --------------------------------------------------------------------------- #
class CatalogMatch(BaseModel):
    part_id: str
    name: str
    brand: str
    category: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    # "exact_part_number": matched by a directly-read catalogue ID, not a
    # similarity score - identity is certain even though compatibility with
    # a specific machine is still checked separately (Agent 3).
    # "semantic": ranked by embedding/text similarity - a plausible
    # candidate, not a confirmed identity. Never conflate the two in UI/API
    # copy (see Checkpoint 3 vs Checkpoint 4 distinction).
    match_type: Literal["exact_part_number", "semantic"] = "semantic"
    # Every match currently returned by this system comes from the bundled
    # internal JSON catalogue (data/spare_parts_catalog.json) - there is no
    # external supplier/source lookup implemented. This field exists so a
    # future external-source integration has somewhere honest to report
    # from without silently overloading this one.
    source: Literal["internal_catalogue"] = "internal_catalogue"


class RAGResult(BaseModel):
    query: str
    matches: list[CatalogMatch] = Field(default_factory=list)
    top_score: float = 0.0
    passed_threshold: bool = False


# --------------------------------------------------------------------------- #
# Compatibility (Agent 3)
# --------------------------------------------------------------------------- #
class SpecCheck(BaseModel):
    field: str
    required_value: Any
    actual_value: Any
    unit: str | None = None
    match: bool


class CompatibilityVerdict(BaseModel):
    part_id: str
    part_name: str
    compatible: bool
    checks: list[SpecCheck] = Field(default_factory=list)
    notes: str | None = None


class CompatibilityResult(BaseModel):
    verdicts: list[CompatibilityVerdict] = Field(default_factory=list)
    has_conflict: bool = False
    conflict_reason: str | None = None
    conflicting_part_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Clarification / Dialogue (Agent 4)
# --------------------------------------------------------------------------- #
class ClarificationQuestion(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    field: str
    question: str
    options: list[str] | None = None
    reason: Literal[
        "low_frame_quality",
        "low_ocr_confidence",
        "low_rag_similarity",
        "compatibility_conflict",
        "insufficient_visual_evidence",
    ]


class PurchaseRecommendation(BaseModel):
    part_id: str
    name: str
    brand: str
    price_usd: float
    confidence: float
    rationale: str
    exploded_diagram_ref: str | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: float = Field(default_factory=time.time)


# --------------------------------------------------------------------------- #
# Checkpoints / observability
# --------------------------------------------------------------------------- #
class CheckpointLog(BaseModel):
    checkpoint: Literal[
        "frame_quality",
        "ocr_confidence",
        "rag_similarity",
        "compatibility",
        "structured_output",
    ]
    node: str
    passed: bool
    detail: str
    timestamp: float = Field(default_factory=time.time)


# --------------------------------------------------------------------------- #
# LangGraph AgentState
# --------------------------------------------------------------------------- #
class AgentState(BaseModel):
    """Full mutable state threaded through the LangGraph orchestrator."""

    job_id: str
    video_path: str | None = None
    image_path: str | None = None
    user_query: str | None = None

    status: JobStatus = JobStatus.PENDING
    current_node: str | None = None

    perception: PerceptionResult | None = None
    # Only populated on the image entry path (image_path set). Kept
    # separate from `perception` - which stays the single cross-mode
    # summary compatibility_agent/clarification_agent already key off of -
    # so the full structured vision evidence for a still-image identify
    # request stays inspectable without duplicating the compatibility logic.
    part_identification: PartIdentification | None = None
    # Real MVID visual-instance-retrieval candidates (image entry path only).
    # These are proof-of-concept visual neighbours, never catalogue matches -
    # see VisualCandidate's docstring. Kept fully separate from rag_result.
    visual_search: VisualSearchResult | None = None
    rag_result: RAGResult | None = None
    compatibility: CompatibilityResult | None = None

    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)
    human_answers: dict[str, str] = Field(default_factory=dict)

    recommendations: list[PurchaseRecommendation] = Field(default_factory=list)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    checkpoint_logs: list[CheckpointLog] = Field(default_factory=list)

    retry_counts: dict[str, int] = Field(default_factory=dict)
    error: str | None = None


# --------------------------------------------------------------------------- #
# API request / response bodies
# --------------------------------------------------------------------------- #
class VideoUploadResponse(BaseModel):
    job_id: str
    filename: str
    status: JobStatus


class ClarifyRequest(BaseModel):
    job_id: str
    question_id: str
    answer: str


class ClarifyResponse(BaseModel):
    job_id: str
    status: JobStatus
    remaining_questions: list[ClarificationQuestion] = Field(default_factory=list)


class CatalogSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class CatalogSearchResponse(BaseModel):
    query: str
    matches: list[CatalogMatch]


class JobStateResponse(BaseModel):
    job_id: str
    status: JobStatus
    current_node: str | None
    perception: PerceptionResult | None
    part_identification: PartIdentification | None
    visual_search: VisualSearchResult | None
    rag_result: RAGResult | None
    compatibility: CompatibilityResult | None
    clarification_questions: list[ClarificationQuestion]
    recommendations: list[PurchaseRecommendation]
    checkpoint_logs: list[CheckpointLog]
    chat_history: list[ChatMessage]
    error: str | None
