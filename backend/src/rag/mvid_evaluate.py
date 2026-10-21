"""Real Recall@K / MRR evaluation for the MVID visual-retrieval index.

For every indexed image, queries the real mvid_visual_parts vector
collection (excluding the query image itself) and checks whether another
real view of the *same physical object* - defined strictly by matching
`object_id` metadata, never by filename string similarity - appears in the
top-K results. This is a real, executed evaluation over the actual
downloaded MVID images and the actual DINOv2 embeddings already indexed by
mvid_ingest.py; no number here is invented.

Usage:
    python -m src.rag.mvid_evaluate [--sample N] [--seed S]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time

from PIL import Image

from src.rag.mvid_ingest import get_mvid_collection
from src.rag.visual_retriever import search_by_image

logger = logging.getLogger(__name__)


def evaluate(sample_size: int | None = None, top_k_values: tuple[int, ...] = (1, 5, 10), seed: int = 0) -> dict:
    collection = get_mvid_collection()
    total = collection.count()
    if total == 0:
        raise RuntimeError("mvid_visual_parts collection is empty - run `python -m src.rag.mvid_ingest` first.")

    all_data = collection.get(include=["metadatas"])
    ids = all_data["ids"]
    metadatas = all_data["metadatas"]

    if sample_size is not None and sample_size < len(ids):
        rng = random.Random(seed)
        idx = rng.sample(range(len(ids)), sample_size)
        ids = [ids[i] for i in idx]
        metadatas = [metadatas[i] for i in idx]

    max_k = max(top_k_values)
    hits = {k: 0 for k in top_k_values}
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    n_queries = 0
    skipped = 0

    for qid, qmeta in zip(ids, metadatas):
        image_path = qmeta["image_path"]
        object_id = qmeta["object_id"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            logger.warning("Skipping query %s (could not open %s): %s", qid, image_path, exc)
            skipped += 1
            continue

        t0 = time.monotonic()
        result = search_by_image(image, query_image_path=image_path, top_k=max_k, exclude_id=qid)
        latencies.append(time.monotonic() - t0)
        n_queries += 1

        rank = next((i for i, c in enumerate(result.candidates, start=1) if c.object_id == object_id), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in top_k_values:
            if rank is not None and rank <= k:
                hits[k] += 1

        if n_queries % 200 == 0:
            logger.info("Evaluated %d/%d queries", n_queries, len(ids))

    if n_queries == 0:
        raise RuntimeError("No queries could be evaluated (all image files failed to open).")

    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    report = {
        "n_queries": n_queries,
        "skipped": skipped,
        "gallery_size": total,
        **{f"recall@{k}": round(hits[k] / n_queries, 4) for k in top_k_values},
        "mrr": round(sum(reciprocal_ranks) / n_queries, 4),
        "avg_query_latency_ms": round(avg_latency_ms, 1),
    }
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Recall@K / MRR evaluation for the MVID visual retrieval index")
    parser.add_argument("--sample", type=int, default=None, help="Evaluate a random sample instead of the full gallery")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(evaluate(sample_size=args.sample, seed=args.seed), indent=2))
