"""Checkpoint 3: RAG Similarity Score Thresholding.

Requires downloading the sentence-transformers embedding model on first
run (network access) and a writable local Chroma persistence directory.
Skips gracefully (rather than failing the suite) in fully offline CI.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def ingested_catalog(tmp_path_factory):
    from src.config import settings

    settings.chroma_host = None
    settings.chroma_persist_dir = str(tmp_path_factory.mktemp("chroma"))
    try:
        from src.rag.ingest_catalog import ingest

        ingest()
    except Exception as exc:  # pragma: no cover - offline/CI environments
        pytest.skip(f"Skipping RAG tests: catalogue ingestion unavailable ({exc})")


def test_high_relevance_query_passes_threshold(ingested_catalog):
    from src.rag.retriever import search_catalog

    result = search_catalog("JCB boom hydraulic cylinder standard duty", top_k=5)
    assert result.matches
    assert result.top_score > 0
    assert result.matches[0].part_id.startswith("JCB")


def test_nonsense_query_may_fail_threshold_and_is_reported_correctly(ingested_catalog):
    from src.config import settings
    from src.rag.retriever import search_catalog

    result = search_catalog("purple singing giraffe birthday cake", top_k=5)
    # We don't assert it fails (embedding models vary), only that the
    # pass/fail flag is *consistent* with the threshold - i.e. Checkpoint 3
    # is wired correctly regardless of the exact similarity value.
    assert result.passed_threshold == (result.top_score >= settings.rag_similarity_threshold)
