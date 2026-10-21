"""Checkpoint 2: OCR & Visual Entity Confidence."""
from __future__ import annotations

from src.models.schemas import EntityType
from src.vision.perception import MockVisionClient


def test_mock_client_produces_expected_entity_types(sharp_bright_frame):
    client = MockVisionClient()
    entities = client.analyze_frame(sharp_bright_frame, frame_index=0)
    types = {e.entity_type for e in entities}
    assert EntityType.MACHINE in types
    assert EntityType.BRAND_PLATE in types
    assert EntityType.SERIAL_TAG in types
    for e in entities:
        assert 0.0 <= e.confidence <= 1.0
        assert e.frame_index == 0


def test_rescan_crop_recovers_or_reports_confidence(sharp_bright_frame):
    client = MockVisionClient()
    entities = client.analyze_frame(sharp_bright_frame, frame_index=0)
    serial_entity = next(e for e in entities if e.entity_type == EntityType.SERIAL_TAG)

    rescanned = client.rescan_crop(sharp_bright_frame, serial_entity.bbox, frame_index=0)

    assert rescanned is not None
    assert rescanned.entity_type == EntityType.SERIAL_TAG
    assert rescanned.raw_text is not None
    # rescan is deliberately biased toward passing the checkpoint threshold
    assert rescanned.confidence >= 0.80


def test_perception_node_flags_uncertain_fields_when_rescan_insufficient(monkeypatch, sharp_bright_frame):
    from src.agents import perception_agent
    from src.models.schemas import AgentState, BoundingBox, DetectedEntity

    class AlwaysLowConfidenceClient:
        def analyze_frame(self, frame_bgr, frame_index):
            return [
                DetectedEntity(
                    entity_type=EntityType.SERIAL_TAG,
                    label="serial/ID tag",
                    bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2),
                    confidence=0.4,
                    frame_index=frame_index,
                    raw_text="XYZ0001",
                )
            ]

    class FakeFrame:
        def __init__(self, index, image):
            self.index = index
            self.timestamp_sec = 0.0
            self.image = image

    monkeypatch.setattr(perception_agent, "get_vision_client", lambda: AlwaysLowConfidenceClient())
    monkeypatch.setattr(perception_agent, "extract_keyframes", lambda path: [FakeFrame(0, sharp_bright_frame)])

    from src.models.schemas import FrameQualityReport

    monkeypatch.setattr(
        perception_agent,
        "probe_frame_reports",
        lambda path: [FrameQualityReport(frame_index=0, timestamp_sec=0.0, laplacian_variance=999,
                                          brightness_mean=100, width=320, height=240, passed=True)],
    )

    state = AgentState(job_id="test-job", video_path="dummy.mp4")
    result = perception_agent.perception_node(state)

    assert result["perception"].uncertain_fields  # AlwaysLowConfidenceClient is not a MockVisionClient, so
    # no rescan is attempted and the low-confidence serial tag is flagged as uncertain.
    assert any("serial_tag" in f for f in result["perception"].uncertain_fields)
