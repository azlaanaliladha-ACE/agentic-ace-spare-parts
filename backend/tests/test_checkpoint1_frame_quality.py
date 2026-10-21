"""Checkpoint 1: Video Frame Quality Check."""
from __future__ import annotations

import numpy as np

from src.vision.video_processor import assess_frame_quality


def test_sharp_bright_frame_passes(sharp_bright_frame):
    report = assess_frame_quality(sharp_bright_frame, frame_index=0, timestamp_sec=0.0)
    assert report.passed
    assert report.reason is None


def test_blurry_flat_frame_fails(blurry_frame):
    report = assess_frame_quality(blurry_frame, frame_index=0, timestamp_sec=0.0)
    assert not report.passed
    assert "blurry" in report.reason


def test_dark_frame_fails(dark_frame):
    report = assess_frame_quality(dark_frame, frame_index=0, timestamp_sec=0.0)
    assert not report.passed
    assert "dark" in report.reason


def test_sharp_portrait_frame_passes_on_longer_edge():
    # Regression: a vertical phone video (e.g. 478x850, narrower than tall)
    # must not fail resolution just because its *width* is the short edge.
    frame = np.zeros((850, 478, 3), dtype=np.uint8)
    frame[::2, ::2] = 220
    frame[1::2, 1::2] = 200
    frame[:] += 40
    report = assess_frame_quality(frame, frame_index=0, timestamp_sec=0.0)
    assert report.passed, report.reason


def test_extract_keyframes_auto_discards_bad_frames(synthetic_video):
    from src.vision.video_processor import extract_keyframes, probe_frame_reports

    all_reports = probe_frame_reports(synthetic_video)
    kept = extract_keyframes(synthetic_video)

    assert len(all_reports) > 0
    # the synthetic video interleaves noisy (sharp) and flat (blurry) stretches
    assert any(r.passed for r in all_reports)
    assert any(not r.passed for r in all_reports)
    assert len(kept) == sum(1 for r in all_reports if r.passed)
