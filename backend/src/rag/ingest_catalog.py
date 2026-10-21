"""Vector DB ingestion script for the spare-parts catalogue.

Usage:
    python -m src.rag.ingest_catalog [--catalog data/spare_parts_catalog.json]

Loads the JSON catalogue, builds a searchable text document per part,
embeds it with a sentence-transformers model, and upserts into a
persistent ChromaDB collection configured for cosine similarity (so
Checkpoint 3's similarity threshold is directly comparable across runs).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)

_DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "spare_parts_catalog.json"


def _build_document(part: dict) -> str:
    specs_str = ", ".join(f"{k}={v}" for k, v in part.get("specs", {}).items())
    models_str = ", ".join(part.get("compatible_models", []))
    return (
        f"{part['name']} by {part['brand']}. Category: {part['category']}. "
        f"Compatible models: {models_str}. Specs: {specs_str}. "
        f"{part.get('description', '')}"
    )


def load_catalog(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        parts = json.load(f)
    if not isinstance(parts, list) or not parts:
        raise ValueError(f"Catalog at {path} must be a non-empty JSON array")
    return parts


def get_chroma_client() -> chromadb.ClientAPI:
    if settings.chroma_host:
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return chromadb.PersistentClient(path=str(settings.chroma_path))


def get_chroma_collection() -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def ingest(catalog_path: Path = _DEFAULT_CATALOG_PATH) -> int:
    parts = load_catalog(catalog_path)
    model = SentenceTransformer(settings.embedding_model)
    collection = get_chroma_collection()

    documents = [_build_document(p) for p in parts]
    embeddings = model.encode(documents, show_progress_bar=False, normalize_embeddings=True).tolist()
    ids = [p["part_id"] for p in parts]
    metadatas = [
        {
            "part_id": p["part_id"],
            "name": p["name"],
            "brand": p["brand"],
            "category": p["category"],
            "compatible_models": json.dumps(p.get("compatible_models", [])),
            "specs": json.dumps(p.get("specs", {})),
            "serial_range": json.dumps(p.get("serial_range", {})),
            "price_usd": p.get("price_usd", 0.0),
            "exploded_diagram_ref": p.get("exploded_diagram_ref", ""),
        }
        for p in parts
    ]

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    logger.info("Ingested %d spare parts into collection '%s'", len(parts), settings.chroma_collection_name)
    return len(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest spare parts catalogue into ChromaDB")
    parser.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG_PATH)
    args = parser.parse_args()
    count = ingest(args.catalog)
    print(f"Ingested {count} parts into '{settings.chroma_collection_name}' at {settings.chroma_path}")
