"""Checkpoint 5: Structured JSON Output Guardrails.

Wraps an LLM call that is supposed to emit JSON matching a Pydantic model.
On a `ValidationError`, the error message is fed back into the next attempt
(mirroring the `instructor` retry pattern) up to
`settings.max_structured_output_retries` times.
"""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel, ValidationError

from src.config import settings
from src.models.schemas import CatalogMatch, CheckpointLog, CompatibilityVerdict

logger = logging.getLogger(__name__)


class StructuredOutputError(RuntimeError):
    pass


def _mock_llm_recommendation_json(
    verdict: CompatibilityVerdict, match: CatalogMatch, attempt: int
) -> str:
    """Deterministically simulates an LLM JSON response.

    Attempt 0 intentionally returns a malformed payload (price encoded as a
    currency string instead of a float) so the Checkpoint 5 retry path is
    exercised every run, not just under flaky real-LLM output. Attempt >= 1
    returns a schema-valid payload, simulating the model correcting itself
    once shown the validation error.
    """
    price = match.metadata.get("price_usd", 0.0)
    avg_conf = sum(c.match for c in verdict.checks) / len(verdict.checks) if verdict.checks else 0.9
    confidence = round(0.6 + 0.35 * avg_conf, 2)
    rationale = (
        f"{match.name} ({match.brand}) matched with catalogue similarity and passed "
        f"{sum(1 for c in verdict.checks if c.match)}/{len(verdict.checks)} compatibility checks."
        if verdict.checks
        else f"{match.name} ({match.brand}) matched via semantic similarity; no hard spec data to cross-check."
    )

    payload = {
        "part_id": match.part_id,
        "name": match.name,
        "brand": match.brand,
        "price_usd": f"${price}" if attempt == 0 else float(price),  # malformed on first attempt
        "confidence": confidence,
        "rationale": rationale,
        "exploded_diagram_ref": match.metadata.get("exploded_diagram_ref") or None,
    }
    return json.dumps(payload)


def _real_llm_recommendation_json(
    verdict: CompatibilityVerdict, match: CatalogMatch, attempt: int, last_error: str | None
) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    schema_hint = (
        '{"part_id": str, "name": str, "brand": str, "price_usd": float, '
        '"confidence": float (0-1), "rationale": str, "exploded_diagram_ref": str|null}'
    )
    error_note = f"\nYour previous response failed schema validation with: {last_error}\nFix and retry." if last_error else ""
    prompt = (
        f"A technician needs a purchase recommendation for this spare part: {match.name} "
        f"(brand={match.brand}, part_id={match.part_id}, price_usd={match.metadata.get('price_usd')}). "
        f"Compatibility checks: {[c.model_dump() for c in verdict.checks]}. "
        f"Respond ONLY with JSON matching: {schema_hint}.{error_note}"
    )
    response = client.messages.create(
        model=settings.vision_model, max_tokens=512, messages=[{"role": "user", "content": prompt}]
    )
    return "".join(b.text for b in response.content if b.type == "text")


def generate_structured_recommendation(
    verdict: CompatibilityVerdict, match: CatalogMatch, target_model: type[BaseModel]
) -> tuple[BaseModel, list[CheckpointLog]]:
    """Runs the generate -> validate -> (retry with error injected) loop.

    Returns the validated model instance plus the checkpoint log trail for
    every attempt (so the UI can show exactly how many retries fired).
    """
    logs: list[CheckpointLog] = []
    last_error: str | None = None
    use_real_llm = settings.vision_provider == "anthropic" and bool(settings.anthropic_api_key)

    for attempt in range(settings.max_structured_output_retries):
        try:
            raw_json = (
                _real_llm_recommendation_json(verdict, match, attempt, last_error)
                if use_real_llm
                else _mock_llm_recommendation_json(verdict, match, attempt)
            )
            start, end = raw_json.find("{"), raw_json.rfind("}")
            parsed = json.loads(raw_json[start : end + 1])
            validated = target_model(**parsed)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            logs.append(
                CheckpointLog(
                    checkpoint="structured_output",
                    node="clarification",
                    passed=False,
                    detail=f"Attempt {attempt + 1}/{settings.max_structured_output_retries} for "
                    f"{match.part_id} failed Pydantic validation: {last_error}. Retrying with "
                    "error injected into next prompt.",
                )
            )
            continue

        logs.append(
            CheckpointLog(
                checkpoint="structured_output",
                node="clarification",
                passed=True,
                detail=f"Attempt {attempt + 1}/{settings.max_structured_output_retries} for "
                f"{match.part_id} passed schema validation.",
            )
        )
        return validated, logs

    raise StructuredOutputError(
        f"Exceeded {settings.max_structured_output_retries} retries generating a valid recommendation "
        f"for {match.part_id}: {last_error}"
    )
