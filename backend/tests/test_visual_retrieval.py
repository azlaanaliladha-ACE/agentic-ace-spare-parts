"""Real visual instance retrieval (MVID proof-of-concept): DINOv2 embeddings
+ a dedicated ChromaDB collection, completely separate from the text
catalogue and from MockVisionClient. These tests actually load the real
facebook/dinov2-base model and run real ChromaDB vector search - nothing
here is mocked, hardcoded, or hand-set as a similarity score.

Since downloading the real ~2,521-image MVID dataset is a slow, one-time,
network-heavy operation unsuitable for a fast test suite, these tests build
a small set of real (not mocked) synthetic PNG images with clearly
distinguishable visual content, arranged in MVID's exact directory
convention (`MVID/<object_id>/train/good/<view>.png`), to exercise the real
embedding + real indexing + real vector-search mechanism end to end. This
proves the retrieval *mechanism* is real and correct; it does not itself
constitute the Recall@K claim on the actual MVID photos, which is reported
separately (see mvid_evaluate.py / the session's written report) once the
real dataset has finished downloading.

Skips automatically if facebook/dinov2-base can't be loaded (e.g. no
network on a fresh machine and it isn't cached yet) rather than failing the
whole suite on an environment issue unrelated to the code under test.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.rag import mvid_ingest


def _make_gallery(tmp_path):
    """Two clearly-different 'physical objects', each with 3 real synthetic
    views (solid color + per-view positional noise pattern, so views of the
    same object are similar but not byte-identical, mirroring real
    multi-view photography of one object from different angles).
    """
    rng = np.random.default_rng(0)
    data_dir = tmp_path / "mvid_raw"
    colors = {"object_red": (200, 30, 30), "object_blue": (30, 30, 200)}
    for obj_id, color in colors.items():
        good_dir = data_dir / "MVID" / obj_id / "train" / "good"
        good_dir.mkdir(parents=True)
        for view in range(3):
            arr = np.full((224, 224, 3), color, dtype=np.uint8)
            noise = rng.integers(-20, 20, arr.shape, dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(good_dir / f"{view:03d}.png")
    return data_dir, colors


@pytest.fixture
def embedding_client():
    try:
        from src.vision.embedding import get_embedding_client

        return get_embedding_client()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"facebook/dinov2-base unavailable in this environment: {exc}")


@pytest.fixture
def mvid_gallery(tmp_path, monkeypatch, embedding_client):
    from src.config import settings

    monkeypatch.setattr(settings, "chroma_host", None)
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma_mvid_test"))
    data_dir, colors = _make_gallery(tmp_path)
    report = mvid_ingest.ingest(data_dir=data_dir, force=True, batch_size=4)
    return data_dir, colors, report


# --------------------------------------------------------------------------- #
# Test 1 - dataset ingestion
# --------------------------------------------------------------------------- #
def test_ingestion_indexes_expected_images_and_metadata(mvid_gallery):
    data_dir, colors, report = mvid_gallery
    assert report["images_found"] == 6  # 2 objects x 3 views
    assert report["unique_object_ids"] == 2
    assert set(report["object_ids"]) == set(colors)
    assert report["indexed"] == 6
    assert report["skipped"] is False

    collection = mvid_ingest.get_mvid_collection()
    assert collection.count() == 6
    got = collection.get(ids=["object_red/000"], include=["metadatas"])
    meta = got["metadatas"][0]
    assert meta["object_id"] == "object_red"
    assert meta["view_id"] == "000"
    assert meta["dataset"] == "MVID"
    assert meta["source"] == "MVID"
    assert meta["type"] == "visual_candidate"


def test_ingestion_is_idempotent_and_skips_reembedding(mvid_gallery):
    # Calling ingest() again without --force must not recompute embeddings.
    data_dir, colors, _ = mvid_gallery
    second = mvid_ingest.ingest(data_dir=data_dir, force=False)
    assert second["skipped"] is True
    assert second["indexed"] == 6


# --------------------------------------------------------------------------- #
# Test 2 - same-object retrieval (real embeddings, real vector search)
# --------------------------------------------------------------------------- #
def test_same_object_views_rank_above_different_object(mvid_gallery):
    from src.rag.visual_retriever import search_by_image

    data_dir, colors, _ = mvid_gallery
    query_path = data_dir / "MVID" / "object_red" / "train" / "good" / "000.png"
    query_image = Image.open(query_path).convert("RGB")

    result = search_by_image(
        query_image, query_image_path=str(query_path), top_k=5, exclude_id="object_red/000"
    )

    assert result.embedding_model == "facebook/dinov2-base"
    assert len(result.candidates) > 0
    # The two other real "object_red" views must outrank every "object_blue" view.
    red_scores = [c.similarity for c in result.candidates if c.object_id == "object_red"]
    blue_scores = [c.similarity for c in result.candidates if c.object_id == "object_blue"]
    assert len(red_scores) == 2  # both other red views recovered
    assert min(red_scores) > max(blue_scores)
    assert all(c.match_type == "visual_candidate" for c in result.candidates)
    assert all(c.dataset == "MVID" for c in result.candidates)


# --------------------------------------------------------------------------- #
# Test 3 - different-object query does not force a same-object claim
# --------------------------------------------------------------------------- #
def test_different_object_query_ranks_correctly(mvid_gallery):
    from src.rag.visual_retriever import search_by_image

    data_dir, colors, _ = mvid_gallery
    query_path = data_dir / "MVID" / "object_blue" / "train" / "good" / "001.png"
    query_image = Image.open(query_path).convert("RGB")

    result = search_by_image(
        query_image, query_image_path=str(query_path), top_k=5, exclude_id="object_blue/001"
    )
    # Top-1 result must be the correct object, not an arbitrary/forced one.
    assert result.candidates[0].object_id == "object_blue"


# --------------------------------------------------------------------------- #
# Test 4 - empty/invalid image
# --------------------------------------------------------------------------- #
def test_search_by_image_rejects_unindexed_collection(tmp_path, monkeypatch, embedding_client):
    from src.config import settings
    from src.rag.visual_retriever import search_by_image

    monkeypatch.setattr(settings, "chroma_host", None)
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma_empty"))

    blank = Image.new("RGB", (64, 64), (0, 0, 0))
    with pytest.raises(RuntimeError, match="empty"):
        search_by_image(blank, query_image_path="blank.png", top_k=5)


def test_visual_retrieval_node_degrades_gracefully_without_index(tmp_path, monkeypatch):
    """image entry path, no MVID index built - must not crash the pipeline
    or fabricate results; state.visual_search stays None."""
    from src.agents.visual_retrieval_agent import visual_retrieval_node
    from src.config import settings
    from src.models.schemas import AgentState, PerceptionResult

    monkeypatch.setattr(settings, "chroma_host", None)
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma_missing"))

    bad_image = tmp_path / "not_an_image.png"
    bad_image.write_bytes(b"not a real png")
    state = AgentState(job_id="vr-1", image_path=str(bad_image), perception=PerceptionResult(usable_frame_count=1))

    result = visual_retrieval_node(state)
    assert "visual_search" not in result or result.get("visual_search") is None
    assert result["current_node"] == "visual_retrieval"
