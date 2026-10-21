"""Input validation for the video ingestion pipeline (extension/existence
checks that run before Checkpoint 1's per-frame quality gate)."""
from __future__ import annotations

import pytest

from src.vision.video_processor import VideoValidationError, validate_video_file


def test_missing_file_raises(tmp_path):
    with pytest.raises(VideoValidationError, match="not found"):
        validate_video_file(tmp_path / "does_not_exist.mp4")


def test_unsupported_extension_raises(tmp_path):
    bad_file = tmp_path / "clip.wmv"
    bad_file.write_bytes(b"not a real video")
    with pytest.raises(VideoValidationError, match="Unsupported"):
        validate_video_file(bad_file)


@pytest.mark.parametrize("suffix", [".mp4", ".avi", ".mov", ".mkv"])
def test_supported_extensions_pass_validation(tmp_path, suffix):
    f = tmp_path / f"clip{suffix}"
    f.write_bytes(b"placeholder bytes")
    validated = validate_video_file(f)
    assert validated == f
