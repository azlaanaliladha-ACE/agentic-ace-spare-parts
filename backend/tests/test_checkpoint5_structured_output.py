"""Checkpoint 5: Structured JSON Output Guardrails (Pydantic validation retry loop)."""
from __future__ import annotations

from src.agents.llm_utils import generate_structured_recommendation
from src.config import settings
from src.models.schemas import CatalogMatch, CompatibilityVerdict, PurchaseRecommendation, SpecCheck


def test_mock_llm_fails_first_attempt_then_recovers():
    match = CatalogMatch(
        part_id="DON-HYDFILT-P550",
        name="Hydraulic Return Filter Element",
        brand="Donaldson",
        category="Filter",
        score=0.95,
        metadata={"price_usd": 38.0, "exploded_diagram_ref": "diagrams/donaldson_p550_filter.svg"},
    )
    verdict = CompatibilityVerdict(
        part_id=match.part_id,
        part_name=match.name,
        compatible=True,
        checks=[SpecCheck(field="compatible_models", required_value="JCB 3CX", actual_value=["JCB 3CX"], match=True)],
    )

    recommendation, logs = generate_structured_recommendation(verdict, match, PurchaseRecommendation)

    assert isinstance(recommendation, PurchaseRecommendation)
    assert recommendation.part_id == "DON-HYDFILT-P550"
    assert isinstance(recommendation.price_usd, float)
    assert recommendation.price_usd == 38.0

    # Attempt 0 must have failed Pydantic validation (malformed price string)
    # and attempt 1 must have passed, proving the retry-with-error-injection
    # loop actually ran rather than succeeding on the first try.
    assert len(logs) >= 2
    assert logs[0].passed is False
    assert "validation" in logs[0].detail.lower()
    assert logs[-1].passed is True


def test_retry_loop_respects_max_retries_setting():
    assert settings.max_structured_output_retries >= 2  # otherwise the recovery above couldn't ever fire
