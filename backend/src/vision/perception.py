"""Vision-LLM client abstraction + mock YOLO/SAM2 detection logic, and the
Checkpoint 2 (OCR / visual entity confidence) verification gate.

Three interchangeable backends are provided behind `get_vision_client()`:
  - MockVisionClient     : deterministic, offline, no API key required.
  - AnthropicVisionClient: real Claude vision call (structured JSON output).
  - OpenAIVisionClient   : real GPT-4o vision call (structured JSON output).

The mock backend also stands in for a YOLO/SAM2 object detector: it returns
bounding boxes + labels + confidences in the exact schema a real detector
would, so swapping in a real model later only requires implementing another
`VisionClient`.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import random
from abc import ABC, abstractmethod

import cv2
import numpy as np

from src.config import settings
from src.models.schemas import BoundingBox, DetectedEntity, EntityType, PartIdentification

logger = logging.getLogger(__name__)

_BRAND_CATALOG = {
    "JCB": {"models": ["3CX", "4CX", "3CX Compact"], "serial_prefix": "JCB"},
    "Caterpillar": {"models": ["320", "320D", "330", "330GC"], "serial_prefix": "CAT"},
    "CASE": {"models": ["580 Super N", "580 SuperM"], "serial_prefix": "CASE"},
}

DETECTION_JSON_INSTRUCTIONS = """
You are an industrial equipment vision inspector. Given the image, identify:
1. The machine type/brand/model if a brand plate or distinctive livery is visible.
2. Any serial/ID tag text (OCR it exactly as printed).
3. Any visibly damaged or worn components (hoses, seals, cylinders, hardware).

Respond ONLY with JSON matching this schema:
{
  "entities": [
    {
      "entity_type": "machine|brand_plate|serial_tag|damaged_component",
      "label": "string",
      "bbox": {"x_min": 0-1, "y_min": 0-1, "x_max": 0-1, "y_max": 0-1},
      "confidence": 0-1,
      "raw_text": "string or null"
    }
  ]
}
"""


PART_IDENTIFICATION_INSTRUCTIONS = """
You are an industrial spare-parts identification expert. You are given a
photo of a single mechanical/industrial component, out of context (not
necessarily attached to a machine). Only report what is actually visible in
THIS image - never guess a specific product line, brand, or part number
that isn't legible or a well-known distinctive design.

Respond ONLY with JSON matching this schema:
{
  "object_type": "string or null - what kind of object this physically is",
  "component_category": "string or null - functional category, e.g. hydraulic cylinder, seal kit, filter, bushing, pump, bracket",
  "visual_description": "string or null - one or two sentence plain description of what you see",
  "visible_features": ["string", ...],
  "dimensions_if_visible": "string or null - only if a ruler/known reference/printed spec is visible, else null",
  "markings": ["string", ...],
  "part_number_if_visible": "string or null - exact text of a stamped/engraved/labeled part or serial number, else null",
  "part_number_confidence": 0.0-1.0,
  "brand_if_visible": "string or null - only if a logo/brand marking is actually legible",
  "machine_context": "string or null - what machine/equipment this looks like it belongs to, only if inferable with real evidence",
  "uncertainties": ["string", ...  - list what is NOT clear/visible enough to determine"],
  "overall_confidence": 0.0-1.0
}

If you cannot determine a field, use null (or an empty list) rather than
guessing. Do not invent a part number, brand, or machine model that isn't
actually legible or unmistakably distinctive in the image.
"""


def _encode_frame_jpeg_b64(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame for vision LLM call")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class VisionClient(ABC):
    @abstractmethod
    def analyze_frame(self, frame_bgr: np.ndarray, frame_index: int) -> list[DetectedEntity]:
        ...

    @abstractmethod
    def identify_part(self, image_bgr: np.ndarray) -> PartIdentification:
        """Single-image spare-part identification (image entry path), as
        opposed to analyze_frame()'s multi-entity machine/serial detection
        used by the video entry path.
        """
        ...


class MockVisionClient(VisionClient):
    """Deterministic mock standing in for a YOLO/SAM2 detector + OCR + VLM.

    Determinism is derived from a hash of the frame's pixel content so the
    same frame always yields the same detections (useful for tests/demos),
    while different frames of the same "video" naturally vary.
    """

    def analyze_frame(self, frame_bgr: np.ndarray, frame_index: int) -> list[DetectedEntity]:
        digest = hashlib.sha256(frame_bgr.tobytes()[:4096] + str(frame_index).encode()).hexdigest()
        rng = random.Random(digest)

        brand = rng.choice(list(_BRAND_CATALOG.keys()))
        model = rng.choice(_BRAND_CATALOG[brand]["models"])
        serial_prefix = _BRAND_CATALOG[brand]["serial_prefix"]
        serial_number = f"{serial_prefix}{rng.randint(0, 9999):04d}"

        entities: list[DetectedEntity] = [
            DetectedEntity(
                entity_type=EntityType.MACHINE,
                label=f"{brand} {model}",
                bbox=BoundingBox(x_min=0.1, y_min=0.15, x_max=0.95, y_max=0.9),
                confidence=round(rng.uniform(0.85, 0.99), 3),
                frame_index=frame_index,
                raw_text=None,
            ),
            DetectedEntity(
                entity_type=EntityType.BRAND_PLATE,
                label=f"{brand} brand plate",
                bbox=BoundingBox(x_min=0.55, y_min=0.6, x_max=0.75, y_max=0.68),
                confidence=round(rng.uniform(0.7, 0.98), 3),
                frame_index=frame_index,
                raw_text=brand,
            ),
            DetectedEntity(
                entity_type=EntityType.SERIAL_TAG,
                # Serial OCR confidence deliberately spans below the 0.80
                # checkpoint threshold so Checkpoint 2 triggers in demos.
                label="serial/ID tag",
                bbox=BoundingBox(x_min=0.58, y_min=0.69, x_max=0.78, y_max=0.74),
                confidence=round(rng.uniform(0.55, 0.97), 3),
                frame_index=frame_index,
                raw_text=serial_number,
            ),
        ]

        if rng.random() < 0.4:
            damaged = rng.choice(
                ["hydraulic hose", "boom cylinder seal", "bucket pin bushing", "track pad"]
            )
            entities.append(
                DetectedEntity(
                    entity_type=EntityType.DAMAGED_COMPONENT,
                    label=damaged,
                    bbox=BoundingBox(x_min=0.3, y_min=0.4, x_max=0.5, y_max=0.55),
                    confidence=round(rng.uniform(0.6, 0.9), 3),
                    frame_index=frame_index,
                    raw_text=None,
                )
            )
        return entities

    def rescan_crop(
        self, frame_bgr: np.ndarray, bbox: BoundingBox, frame_index: int
    ) -> DetectedEntity | None:
        """Checkpoint 2 correction: simulate a secondary zoomed-in rescan of
        a low-confidence region, which typically recovers higher confidence
        because the crop is upsampled before re-inspection.
        """
        h, w = frame_bgr.shape[:2]
        x1, y1 = int(bbox.x_min * w), int(bbox.y_min * h)
        x2, y2 = int(bbox.x_max * w), int(bbox.y_max * h)
        crop = frame_bgr[max(0, y1) : max(y1 + 1, y2), max(0, x1) : max(x1 + 1, x2)]
        if crop.size == 0:
            return None
        upscaled = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

        digest = hashlib.sha256(upscaled.tobytes()[:4096] + b"rescan").hexdigest()
        rng = random.Random(digest)
        serial_prefix = rng.choice([v["serial_prefix"] for v in _BRAND_CATALOG.values()])
        return DetectedEntity(
            entity_type=EntityType.SERIAL_TAG,
            label="serial/ID tag (rescanned)",
            bbox=bbox,
            confidence=round(rng.uniform(0.82, 0.99), 3),  # rescan biased to pass threshold
            frame_index=frame_index,
            raw_text=f"{serial_prefix}{rng.randint(0, 9999):04d}",
        )

    def identify_part(self, image_bgr: np.ndarray) -> PartIdentification:
        """Deterministic mock, explicitly labeled as such - only ever used
        when VISION_PROVIDER=mock is configured (see get_vision_client()).
        """
        digest = hashlib.sha256(image_bgr.tobytes()[:4096]).hexdigest()
        rng = random.Random(digest)
        category = rng.choice(["Hydraulic Cylinder", "Seal Kit", "Filter", "Hydraulic Pump", "Chassis Hardware"])
        return PartIdentification(
            object_type="industrial mechanical component (mock)",
            component_category=category,
            visual_description=f"Mock/demo description standing in for a {category.lower()}.",
            visible_features=["cylindrical body", "mounting flange"],
            markings=[],
            part_number_if_visible=None,
            part_number_confidence=0.0,
            brand_if_visible=None,
            machine_context=None,
            uncertainties=["Mock provider - not derived from actual image pixels."],
            overall_confidence=round(rng.uniform(0.4, 0.6), 3),
        )


class AnthropicVisionClient(VisionClient):
    def __init__(self) -> None:
        import anthropic

        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def analyze_frame(self, frame_bgr: np.ndarray, frame_index: int) -> list[DetectedEntity]:
        image_b64 = _encode_frame_jpeg_b64(frame_bgr)
        response = self._client.messages.create(
            model=settings.vision_model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
                        },
                        {"type": "text", "text": DETECTION_JSON_INSTRUCTIONS},
                    ],
                }
            ],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return _parse_entities_json(text, frame_index)

    def identify_part(self, image_bgr: np.ndarray) -> PartIdentification:
        image_b64 = _encode_frame_jpeg_b64(image_bgr)
        response = self._client.messages.create(
            model=settings.vision_model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
                        },
                        {"type": "text", "text": PART_IDENTIFICATION_INSTRUCTIONS},
                    ],
                }
            ],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        logger.info("IMAGE ANALYSIS Provider=anthropic Model=%s Mock=false Vision request=successful", settings.vision_model)
        return _parse_part_identification_json(text)


class OpenAIVisionClient(VisionClient):
    def __init__(self) -> None:
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=settings.openai_api_key)

    def analyze_frame(self, frame_bgr: np.ndarray, frame_index: int) -> list[DetectedEntity]:
        image_b64 = _encode_frame_jpeg_b64(frame_bgr)
        response = self._client.chat.completions.create(
            model=settings.vision_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DETECTION_JSON_INSTRUCTIONS},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        )
        text = response.choices[0].message.content or "{}"
        return _parse_entities_json(text, frame_index)

    def identify_part(self, image_bgr: np.ndarray) -> PartIdentification:
        image_b64 = _encode_frame_jpeg_b64(image_bgr)
        response = self._client.chat.completions.create(
            model=settings.vision_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PART_IDENTIFICATION_INSTRUCTIONS},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        )
        text = response.choices[0].message.content or "{}"
        logger.info("IMAGE ANALYSIS Provider=openai Model=%s Mock=false Vision request=successful", settings.vision_model)
        return _parse_part_identification_json(text)


class OllamaVisionClient(VisionClient):
    """Free, local, open-source vision inference via Ollama - no API key,
    no per-call cost, no data leaving the machine. Requires Ollama running
    (https://ollama.com) with a vision-capable model already pulled, e.g.
    `ollama pull moondream` or `ollama pull qwen2.5vl:7b`.
    """

    def __init__(self) -> None:
        import httpx

        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_vision_model
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=3)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama not reachable at {self._base_url}: {exc}") from exc

        # Fail fast and specifically ("model not pulled") rather than only
        # discovering this on the first analyze_frame() call, where a
        # missing model would otherwise surface as a generic Ollama HTTP
        # error deep inside the pipeline.
        pulled = {m.get("name") for m in resp.json().get("models", [])}
        if self._model not in pulled and f"{self._model}:latest" not in pulled:
            raise RuntimeError(
                f"Ollama model '{self._model}' is not pulled (available: "
                f"{sorted(pulled) or 'none'}). Run `ollama pull {self._model}` first."
            )

    def analyze_frame(self, frame_bgr: np.ndarray, frame_index: int) -> list[DetectedEntity]:
        import httpx

        from src.observability import call_log

        image_b64 = _encode_frame_jpeg_b64(frame_bgr)
        response = httpx.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": DETECTION_JSON_INSTRUCTIONS,
                "images": [image_b64],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    # Small local models can fall into a degenerate token
                    # loop on this multi-entity JSON task (observed:
                    # repeating `{"confidence":0.83,"raw_text":null}`
                    # until hitting the length cap). A repetition penalty
                    # reliably breaks that loop.
                    "repeat_penalty": 1.3,
                    "repeat_last_n": 64,
                    "num_predict": 800,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        call_log.record(
            "perception",
            call="analyze_frame",
            frame_index=frame_index,
            model=self._model,
            prompt_eval_count=payload.get("prompt_eval_count"),
            eval_count=payload.get("eval_count"),
            prompt_eval_duration_ms=(payload.get("prompt_eval_duration") or 0) / 1e6,
            eval_duration_ms=(payload.get("eval_duration") or 0) / 1e6,
            load_duration_ms=(payload.get("load_duration") or 0) / 1e6,
            total_duration_ms=(payload.get("total_duration") or 0) / 1e6,
            image_bytes=len(image_b64),
        )
        text = payload.get("response", "{}")
        return _parse_entities_json(text, frame_index)

    def identify_part(self, image_bgr: np.ndarray) -> PartIdentification:
        import time

        import httpx

        from src.observability import call_log

        image_b64 = _encode_frame_jpeg_b64(image_bgr)
        start = time.monotonic()
        response = httpx.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": PART_IDENTIFICATION_INSTRUCTIONS,
                "images": [image_b64],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "repeat_penalty": 1.3,
                    "repeat_last_n": 64,
                    "num_predict": 800,
                },
            },
            timeout=120,
        )
        latency_sec = time.monotonic() - start
        response.raise_for_status()
        payload = response.json()
        call_log.record(
            "perception",
            call="identify_part",
            model=self._model,
            prompt_eval_count=payload.get("prompt_eval_count"),
            eval_count=payload.get("eval_count"),
            prompt_eval_duration_ms=(payload.get("prompt_eval_duration") or 0) / 1e6,
            eval_duration_ms=(payload.get("eval_duration") or 0) / 1e6,
            load_duration_ms=(payload.get("load_duration") or 0) / 1e6,
            total_duration_ms=(payload.get("total_duration") or 0) / 1e6,
            image_bytes=len(image_b64),
            latency_sec=latency_sec,
        )
        text = payload.get("response", "{}")
        logger.info(
            "IMAGE ANALYSIS Provider=ollama Model=%s Mock=false Vision request=successful Latency=%.2fs",
            self._model,
            latency_sec,
        )
        return _parse_part_identification_json(text)


def _normalize_raw_entity(raw: dict) -> dict:
    """Defensively normalizes common small-model JSON quirks observed from
    Ollama-hosted models (typo'd keys, bbox as a bare list instead of an
    object, out-of-range confidence) before Pydantic validation. This is
    generic cleanup for known LLM-output shapes, not special-cased to any
    one response.
    """
    raw = dict(raw)

    if "entity_type" not in raw and "entitytype" in raw:
        raw["entity_type"] = raw.pop("entitytype")

    label = raw.get("label")
    if isinstance(label, list):
        raw["label"] = ", ".join(str(v) for v in label) if label else "unlabeled"

    bbox = raw.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            x_min, y_min, x_max, y_max = (float(v) for v in bbox)
            raw["bbox"] = {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
        except (TypeError, ValueError):
            pass

    confidence = raw.get("confidence")
    if isinstance(confidence, (int, float)):
        raw["confidence"] = max(0.0, min(1.0, abs(confidence)))
    elif confidence is None:
        # Model omitted confidence entirely - treat as uncertain (below
        # the OCR/entity confidence threshold) rather than discarding a
        # detection outright; Checkpoint 2 handles it from there.
        raw["confidence"] = 0.5

    return raw


def _parse_entities_json(text: str, frame_index: int) -> list[DetectedEntity]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        # Small/local models (e.g. moondream via Ollama) don't reliably
        # follow strict JSON-schema instructions the way frontier hosted
        # models do. Treat an unparseable response as "no entities found
        # in this frame" rather than crashing the whole pipeline - the
        # graph still runs to a real (if less certain) conclusion, and
        # perception's own confidence checkpoints handle the rest.
        logger.warning("Vision model returned non-JSON response for frame %d: %s", frame_index, exc)
        return []
    entities: list[DetectedEntity] = []
    for raw in payload.get("entities", []):
        if not isinstance(raw, dict):
            continue
        raw = _normalize_raw_entity(raw)
        raw["frame_index"] = frame_index
        try:
            entities.append(DetectedEntity(**raw))
        except Exception as exc:  # malformed entity from the LLM - skip, don't crash the pipeline
            logger.warning("Skipping malformed detected entity from vision LLM: %s (%s)", raw, exc)
    return entities


def _normalize_part_identification(raw: dict) -> dict:
    """Same defensive-cleanup rationale as _normalize_raw_entity: small/local
    vision models don't always follow the requested schema exactly (string
    where a list was asked for, out-of-range confidence, etc.)."""
    raw = dict(raw)

    for list_field in ("visible_features", "markings", "uncertainties"):
        value = raw.get(list_field)
        if isinstance(value, str):
            raw[list_field] = [value] if value.strip() else []
        elif value is None:
            raw[list_field] = []
        elif not isinstance(value, list):
            raw[list_field] = []

    for conf_field in ("part_number_confidence", "overall_confidence"):
        value = raw.get(conf_field)
        if isinstance(value, (int, float)):
            raw[conf_field] = max(0.0, min(1.0, abs(float(value))))
        else:
            raw[conf_field] = 0.0

    # Treat common "not applicable" strings some small models emit as null
    # rather than a real value, so downstream logic (e.g. exact part-number
    # lookup) doesn't try to search for the literal string "N/A".
    _NULLISH = {"", "n/a", "na", "none", "null", "unknown", "not visible", "not applicable"}
    for str_field in (
        "object_type", "component_category", "visual_description", "dimensions_if_visible",
        "part_number_if_visible", "brand_if_visible", "machine_context",
    ):
        value = raw.get(str_field)
        if isinstance(value, str) and value.strip().lower() in _NULLISH:
            raw[str_field] = None

    return raw


def _parse_part_identification_json(text: str) -> PartIdentification:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        logger.warning("Vision model returned non-JSON response for part identification")
        return PartIdentification(uncertainties=["Vision model response could not be parsed as JSON."])
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("Vision model returned malformed JSON for part identification: %s", exc)
        return PartIdentification(uncertainties=["Vision model response could not be parsed as JSON."])
    if not isinstance(payload, dict):
        return PartIdentification(uncertainties=["Vision model response was not a JSON object."])
    try:
        return PartIdentification(**_normalize_part_identification(payload))
    except Exception as exc:  # malformed field from the LLM - degrade to "nothing extracted", don't crash
        logger.warning("Skipping malformed part identification from vision LLM: %s (%s)", payload, exc)
        return PartIdentification(uncertainties=[f"Vision model response failed schema validation: {exc}"])


_CLIENT_REGISTRY: dict[str, type[VisionClient]] = {
    "mock": MockVisionClient,
    "anthropic": AnthropicVisionClient,
    "openai": OpenAIVisionClient,
    "ollama": OllamaVisionClient,
}


def get_vision_client() -> VisionClient:
    """Instantiates the configured provider. Deliberately does NOT catch
    construction failures and fall back to MockVisionClient: a real provider
    that is misconfigured or unreachable must surface as a real pipeline
    failure (JobStatus.FAILED with the actual error), not a silent swap to
    fake deterministic data. Mock is only ever used when explicitly
    configured via VISION_PROVIDER=mock.
    """
    provider = settings.vision_provider.lower()
    cls = _CLIENT_REGISTRY.get(provider)
    if cls is None:
        raise ValueError(f"Unknown VISION_PROVIDER '{provider}'. Options: {list(_CLIENT_REGISTRY)}")
    return cls()
