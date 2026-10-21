"""Query-side visual instance retrieval over the MVID ChromaDB collection
(src/rag/mvid_ingest.py). Real vector similarity search only - no mock
results, no hardcoded candidates, no generated similarity scores.
"""
from __future__ import annotations

import logging

from PIL import Image

from src.models.schemas import VisualCandidate, VisualSearchResult
from src.rag.mvid_ingest import get_mvid_collection
from src.vision.embedding import DINOV2_MODEL_NAME, get_embedding_client

logger = logging.getLogger(__name__)


def search_by_image(image: Image.Image, query_image_path: str, top_k: int = 5, exclude_id: str | None = None) -> VisualSearchResult:
    """Embeds `image` with the real DINOv2 client and returns the top_k
    nearest MVID images by cosine similarity, drawn only from what's
    actually indexed - never fabricated.

    `exclude_id` (an "<object_id>/<view_id>" collection id) lets evaluation
    code hold a query image out of its own gallery; production calls leave
    it None since a freshly-uploaded user image is never itself in the index.
    """
    collection = get_mvid_collection()
    if collection.count() == 0:
        raise RuntimeError(
            "MVID collection is empty - run `python -m src.rag.mvid_ingest` before calling search_by_image()."
        )

    client = get_embedding_client()
    embedding = client.embed_image(image)

    # Over-fetch by one when excluding the query itself, since Chroma has
    # no server-side "exclude this id" filter for a plain nearest-neighbour
    # query; the extra result (if any) is dropped client-side below.
    n_results = top_k + 1 if exclude_id else top_k
    n_results = min(n_results, collection.count())

    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    candidates: list[VisualCandidate] = []
    for cid, distance, meta in zip(ids, distances, metadatas):
        if exclude_id is not None and cid == exclude_id:
            continue
        similarity = max(0.0, 1.0 - distance)  # collection uses hnsw:space=cosine, same convention as search_catalog()
        candidates.append(
            VisualCandidate(
                object_id=meta["object_id"],
                image_path=meta["image_path"],
                view_id=meta["view_id"],
                similarity=round(similarity, 4),
            )
        )
        if len(candidates) == top_k:
            break

    return VisualSearchResult(
        query_image_path=query_image_path,
        candidates=candidates,
        embedding_model=DINOV2_MODEL_NAME,
        index_collection=collection.name,
    )
