"""Agent 1: Frame Sampler & Perception Agent.

Extracts keyframes from the uploaded video (Checkpoint 1: frame quality),
runs them through a Vision LLM / mock YOLO-SAM2 detector, and resolves the
machine brand/model/serial + any visibly damaged components. Enforces
Checkpoint 2 (OCR / visual entity confidence) with an automatic zoomed
rescan correction before escalating to human clarification.

Node is idempotent: if `state.perception` already holds a valid result and
no re-upload was requested, it is a no-op so the graph can safely re-enter
this node after a clarification round-trip (see agents/graph.py).
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from src.config import settings
from src.models.schemas import (
    AgentState,
    ChatMessage,
    CheckpointLog,
    EntityType,
    JobStatus,
    PartIdentification,
    PerceptionResult,
)
from src.vision.perception import MockVisionClient, get_vision_client
from src.vision.video_processor import assess_frame_quality, extract_keyframes, probe_frame_reports

logger = logging.getLogger(__name__)


def _needs_recompute(state: AgentState) -> bool:
    # `clarification_node` clears `perception` back to None whenever it
    # applies a re-upload answer (Checkpoint 1 correction), which is what
    # actually triggers recompute here - see agents/clarification_agent.py.
    return state.perception is None


def _perceive_from_image(state: AgentState) -> dict:
    """Image entry path: a single still photo of a spare part, rather than
    a video of a machine. Reuses Checkpoint 1 (frame quality) by treating
    the image as a one-frame "video", and calls the vision client's
    identify_part() (not analyze_frame(), which is shaped for multi-entity
    machine/serial detection). Converges into the exact same
    perception/rag/compatibility/clarification/finalize graph as the video
    path from here on - see route_after_perception in agents/graph.py for
    the one extra routing check this path needs (insufficient evidence).
    """
    import cv2

    path = Path(state.image_path)
    if not path.exists():
        return {
            "current_node": "perception",
            "status": JobStatus.FAILED,
            "error": f"Image file not found: {path}",
        }

    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        return {
            "current_node": "perception",
            "status": JobStatus.FAILED,
            "error": f"Could not decode image (unsupported/corrupt file): {path}",
        }

    quality = assess_frame_quality(image_bgr, frame_index=0, timestamp_sec=0.0)
    checkpoint_logs = [
        CheckpointLog(
            checkpoint="frame_quality",
            node="perception",
            passed=quality.passed,
            detail=(
                "Image passed the quality gate (blur/brightness/resolution)."
                if quality.passed
                else f"Image failed the quality gate: {quality.reason}"
            ),
        )
    ]

    if not quality.passed:
        result = PerceptionResult(frame_reports=[quality], usable_frame_count=0, discarded_frame_count=1)
        return {
            "current_node": "perception",
            "perception": result,
            # RUNNING_PERCEPTION (not RAG) - route_after_perception sends a
            # usable_frame_count==0 result straight to clarification, which
            # already knows how to ask for a re-upload (Checkpoint 1 path).
            "status": JobStatus.RUNNING_PERCEPTION,
            "checkpoint_logs": state.checkpoint_logs + checkpoint_logs,
        }

    client = get_vision_client()
    part_id: PartIdentification = client.identify_part(image_bgr)

    if part_id.part_number_if_visible:
        passed_ocr = part_id.part_number_confidence >= settings.ocr_confidence_threshold
        checkpoint_logs.append(
            CheckpointLog(
                checkpoint="ocr_confidence",
                node="perception",
                passed=passed_ocr,
                detail=(
                    f"Part-number OCR confidence {part_id.part_number_confidence:.2f} "
                    f"({'>=' if passed_ocr else '<'} threshold {settings.ocr_confidence_threshold})."
                ),
            )
        )

    uncertain_fields = list(part_id.uncertainties)
    if part_id.part_number_if_visible and part_id.part_number_confidence < settings.ocr_confidence_threshold:
        uncertain_fields.append(f"part_number:{part_id.part_number_if_visible}")

    perception_result = PerceptionResult(
        frame_reports=[quality],
        machine_brand=part_id.brand_if_visible,
        machine_model=part_id.machine_context,
        serial_number=None,
        serial_confidence=0.0,
        damaged_components=[part_id.component_category] if part_id.component_category else [],
        uncertain_fields=uncertain_fields,
        usable_frame_count=1,
        discarded_frame_count=0,
    )

    query_terms = " ".join(
        filter(
            None,
            [
                part_id.component_category,
                part_id.object_type,
                part_id.brand_if_visible,
                part_id.visual_description,
                " ".join(part_id.visible_features),
            ],
        )
    ).strip() or None

    chat_msg = ChatMessage(
        role="assistant",
        content=(
            f"Image perception complete: "
            f"{part_id.component_category or part_id.object_type or 'unidentified component'} "
            f"(overall confidence {part_id.overall_confidence:.2f})."
            + (f" Visible part number: {part_id.part_number_if_visible}." if part_id.part_number_if_visible else "")
            + (f" Uncertain: {', '.join(part_id.uncertainties)}." if part_id.uncertainties else "")
        ),
    )

    return {
        "current_node": "perception",
        "perception": perception_result,
        "part_identification": part_id,
        "user_query": state.user_query or query_terms,
        "status": JobStatus.RUNNING_RAG,
        "checkpoint_logs": state.checkpoint_logs + checkpoint_logs,
        "chat_history": state.chat_history + [chat_msg],
    }


def perception_node(state: AgentState) -> dict:
    if not _needs_recompute(state):
        return {"current_node": "perception"}

    if state.image_path and not state.video_path:
        return _perceive_from_image(state)

    if not state.video_path:
        return {
            "current_node": "perception",
            "status": JobStatus.FAILED,
            "error": "No video_path or image_path provided to perception agent.",
        }

    all_reports = probe_frame_reports(state.video_path)
    kept_frames = extract_keyframes(state.video_path)

    passed = sum(1 for r in all_reports if r.passed)
    discarded = len(all_reports) - passed

    checkpoint_logs = [
        CheckpointLog(
            checkpoint="frame_quality",
            node="perception",
            passed=discarded == 0,
            detail=f"{passed}/{len(all_reports)} sampled frames passed quality gate "
            f"(blur/brightness/resolution); {discarded} auto-discarded.",
        )
    ]

    client = get_vision_client()
    entities = []
    for frame in kept_frames:
        entities.extend(client.analyze_frame(frame.image, frame.index))

    # --- Checkpoint 2: OCR & Visual Entity Confidence -----------------------
    uncertain_fields: list[str] = []
    ocr_relevant = [e for e in entities if e.entity_type in (EntityType.SERIAL_TAG, EntityType.BRAND_PLATE)]
    low_conf = [e for e in ocr_relevant if e.confidence < settings.ocr_confidence_threshold]

    for idx, entity in enumerate(low_conf):
        frame = next((f for f in kept_frames if f.index == entity.frame_index), None)
        rescanned = None
        if frame is not None and isinstance(client, MockVisionClient):
            rescanned = client.rescan_crop(frame.image, entity.bbox, entity.frame_index)
        if rescanned is not None and rescanned.confidence >= settings.ocr_confidence_threshold:
            entities[entities.index(entity)] = rescanned
            checkpoint_logs.append(
                CheckpointLog(
                    checkpoint="ocr_confidence",
                    node="perception",
                    passed=True,
                    detail=f"{entity.entity_type.value} confidence {entity.confidence:.2f} < "
                    f"{settings.ocr_confidence_threshold}; secondary zoomed rescan recovered "
                    f"confidence {rescanned.confidence:.2f}.",
                )
            )
        else:
            uncertain_fields.append(f"{entity.entity_type.value}:{entity.label}")
            checkpoint_logs.append(
                CheckpointLog(
                    checkpoint="ocr_confidence",
                    node="perception",
                    passed=False,
                    detail=f"{entity.entity_type.value} confidence {entity.confidence:.2f} < "
                    f"{settings.ocr_confidence_threshold}; rescan did not recover confidence. "
                    "Flagged as 'Uncertain - Requires Clarification'.",
                )
            )

    machine_entities = [e for e in entities if e.entity_type == EntityType.MACHINE]
    brand_plate_entities = [e for e in entities if e.entity_type == EntityType.BRAND_PLATE]
    serial_entities = [e for e in entities if e.entity_type == EntityType.SERIAL_TAG]
    damaged_entities = [e for e in entities if e.entity_type == EntityType.DAMAGED_COMPONENT]

    machine_brand = None
    machine_model = None
    if machine_entities:
        label = Counter(e.label for e in machine_entities).most_common(1)[0][0]
        parts = label.split(" ", 1)
        machine_brand = parts[0]
        machine_model = parts[1] if len(parts) > 1 else None
    elif brand_plate_entities:
        machine_brand = Counter(e.raw_text for e in brand_plate_entities if e.raw_text).most_common(1)[0][0]

    best_serial = max(serial_entities, key=lambda e: e.confidence, default=None)
    serial_number = best_serial.raw_text if best_serial else None
    serial_confidence = best_serial.confidence if best_serial else 0.0

    result = PerceptionResult(
        frame_reports=all_reports,
        detected_entities=entities,
        machine_brand=machine_brand,
        machine_model=machine_model,
        serial_number=serial_number,
        serial_confidence=serial_confidence,
        damaged_components=sorted({e.label for e in damaged_entities}),
        uncertain_fields=uncertain_fields,
        usable_frame_count=len(kept_frames),
        discarded_frame_count=discarded,
    )

    query_terms = " ".join(filter(None, [machine_brand, machine_model, *result.damaged_components]))
    chat_msg = ChatMessage(
        role="assistant",
        content=(
            f"Perception complete: identified {machine_brand or 'unknown brand'} "
            f"{machine_model or ''} (serial {serial_number or 'unresolved'}, "
            f"confidence {serial_confidence:.2f}). "
            f"Damaged/worn components observed: {', '.join(result.damaged_components) or 'none'}. "
            f"{len(kept_frames)} usable frames, {discarded} discarded for quality."
        ),
    )

    status = JobStatus.RUNNING_RAG
    if result.usable_frame_count == 0:
        status = JobStatus.RUNNING_PERCEPTION  # clarification router will catch this

    return {
        "current_node": "perception",
        "perception": result,
        "user_query": state.user_query or query_terms,
        "status": status,
        "checkpoint_logs": state.checkpoint_logs + checkpoint_logs,
        "chat_history": state.chat_history + [chat_msg],
    }
