"""Query-side RAG retrieval over the ChromaDB spare-parts collection, plus
Checkpoint 3 (RAG similarity score thresholding).
"""
from __future__ import annotations

import json
import logging
import threading

from sentence_transformers import SentenceTransformer

from src.config import settings
from src.models.schemas import CatalogMatch, RAGResult
from src.rag.ingest_catalog import get_chroma_collection

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    with _model_lock:
        if _embedding_model is None:
            _embedding_model = SentenceTransformer(settings.embedding_model)
        return _embedding_model


def search_catalog(query: str, top_k: int = 5) -> RAGResult:
    """Embeds `query` and searches the vector store. Cosine similarity is
    derived as `1 - cosine_distance` since the collection is configured
    with `hnsw:space=cosine`.
    """
    model = _get_embedding_model()
    collection = get_chroma_collection()

    if collection.count() == 0:
        logger.warning("Catalog collection is empty - run `python -m src.rag.ingest_catalog` first.")
        return RAGResult(query=query, matches=[], top_score=0.0, passed_threshold=False)

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["metadatas", "documents", "distances"],
    )

    matches: list[CatalogMatch] = []
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for part_id, distance, meta in zip(ids, distances, metadatas):
        similarity = max(0.0, 1.0 - distance)
        matches.append(
            CatalogMatch(
                part_id=part_id,
                name=meta["name"],
                brand=meta["brand"],
                category=meta["category"],
                score=round(similarity, 4),
                metadata={
                    "compatible_models": json.loads(meta.get("compatible_models", "[]")),
                    "specs": json.loads(meta.get("specs", "{}")),
                    "serial_range": json.loads(meta.get("serial_range", "{}")),
                    "price_usd": meta.get("price_usd", 0.0),
                    "exploded_diagram_ref": meta.get("exploded_diagram_ref", ""),
                },
            )
        )

    matches.sort(key=lambda m: m.score, reverse=True)
    top_score = matches[0].score if matches else 0.0

    return RAGResult(
        query=query,
        matches=matches,
        top_score=top_score,
        passed_threshold=top_score >= settings.rag_similarity_threshold,
    )


def search_catalog_by_part_number(part_number: str) -> CatalogMatch | None:
    """Exact-identity lookup by catalogue ID, for when a part number was
    actually read off the physical part (OCR/vision), as opposed to
    `search_catalog`'s embedding-similarity ranking. Case/whitespace
    insensitive; catalogue IDs are the part's real `part_id` (see
    src/rag/ingest_catalog.py - ids == part_id at ingest time).

    Returns None (not a low-scoring match) when nothing has that exact ID -
    callers must not fall back to treating this as a semantic result.
    """
    collection = get_chroma_collection()
    if collection.count() == 0:
        return None

    needle = part_number.strip().upper()
    all_ids = collection.get(include=[])["ids"]
    exact_id = next((i for i in all_ids if i.strip().upper() == needle), None)
    if exact_id is None:
        return None

    got = collection.get(ids=[exact_id], include=["metadatas", "documents"])
    if not got["ids"]:
        return None
    meta = got["metadatas"][0]

    return CatalogMatch(
        part_id=exact_id,
        name=meta["name"],
        brand=meta["brand"],
        category=meta["category"],
        score=1.0,
        match_type="exact_part_number",
        metadata={
            "compatible_models": json.loads(meta.get("compatible_models", "[]")),
            "specs": json.loads(meta.get("specs", "{}")),
            "serial_range": json.loads(meta.get("serial_range", "{}")),
            "price_usd": meta.get("price_usd", 0.0),
            "exploded_diagram_ref": meta.get("exploded_diagram_ref", ""),
        },
    )
