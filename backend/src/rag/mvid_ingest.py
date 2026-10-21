"""MVID (Multi-View Industrial Part Dataset) ingestion: embeds the clean
multi-view "good" image subset with DINOv2 (src/vision/embedding.py) and
indexes it into its own ChromaDB collection, completely separate from the
text spare-parts catalogue collection (src/rag/ingest_catalog.py).

This is a real-photo visual-retrieval proof-of-concept corpus, NOT our
spare-parts catalogue - see VisualCandidate's docstring in
src/models/schemas.py. Source (MIT license):
https://huggingface.co/datasets/kentaito321/Multi_View_Industrial_part_Dataset

Only MVID/<object_id>/train/good/*.png is ever indexed. The dataset's
test/, ground_truth/, and masks/ subtrees are for a *different* task
(anomaly/defect segmentation) and are deliberately never downloaded or
indexed here.

Usage:
    python -m src.rag.mvid_ingest [--data-dir data/mvid_raw] [--force]
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.rag.ingest_catalog import get_chroma_client
from src.vision.embedding import get_embedding_client

logger = logging.getLogger(__name__)

MVID_COLLECTION_NAME = "mvid_visual_parts"
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mvid_raw"


def discover_images(data_dir: Path) -> list[Path]:
    """Only the clean multi-view 'good' images - never test/ground_truth/masks."""
    return sorted(data_dir.glob("MVID/*/train/good/*.png"))


def _object_id(image_path: Path) -> str:
    # .../MVID/<object_id>/train/good/<view>.png
    return image_path.parents[2].name


def report_dataset(data_dir: Path, images: list[Path]) -> dict:
    object_ids = sorted({_object_id(p) for p in images})
    total_bytes = sum(p.stat().st_size for p in images)
    report = {
        "images_found": len(images),
        "unique_object_ids": len(object_ids),
        "object_ids": object_ids,
        "directory_structure": f"{data_dir}/MVID/<object_id>/train/good/<view>.png",
        "total_disk_size_mb": round(total_bytes / 1e6, 1),
    }
    logger.info("MVID dataset report: %s", report)
    return report


def get_mvid_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(name=MVID_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def ingest(data_dir: Path = _DEFAULT_DATA_DIR, force: bool = False, batch_size: int = 16) -> dict:
    images = discover_images(data_dir)
    if not images:
        raise RuntimeError(
            f"No MVID 'good' images found under {data_dir}/MVID/*/train/good/*.png - "
            "download the dataset first (see README for the huggingface_hub snapshot_download call)."
        )
    report = report_dataset(data_dir, images)

    collection = get_mvid_collection()
    if not force and collection.count() == len(images):
        logger.info("MVID collection already has all %d images indexed - skipping re-embedding.", collection.count())
        report.update(indexed=collection.count(), skipped=True, elapsed_sec=0.0)
        return report

    client = get_embedding_client()
    ids: list[str] = []
    metadatas: list[dict] = []
    for p in images:
        object_id = _object_id(p)
        view_id = p.stem
        ids.append(f"{object_id}/{view_id}")
        metadatas.append(
            {
                "dataset": "MVID",
                "object_id": object_id,
                "image_path": str(p),
                "view_id": view_id,
                "source": "MVID",
                "type": "visual_candidate",
            }
        )

    start = time.monotonic()
    batches: list[np.ndarray] = []
    for i in range(0, len(images), batch_size):
        batch_paths = images[i : i + batch_size]
        pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
        batches.append(client.embed_images(pil_images, batch_size=batch_size))
        done = i + len(batch_paths)
        if (i // batch_size) % 10 == 0 or done == len(images):
            logger.info("Embedded %d/%d MVID images", done, len(images))
    embeddings = np.concatenate(batches, axis=0)
    elapsed = time.monotonic() - start

    collection.upsert(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)
    throughput = len(images) / elapsed if elapsed > 0 else float("inf")
    logger.info("Indexed %d MVID images in %.1fs (%.2f img/s)", len(images), elapsed, throughput)

    report.update(indexed=len(images), skipped=False, elapsed_sec=round(elapsed, 1), images_per_sec=round(throughput, 2))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest MVID visual-retrieval proof-of-concept dataset")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Re-embed even if the collection already looks complete")
    args = parser.parse_args()
    result = ingest(args.data_dir, force=args.force)
    print(result)
