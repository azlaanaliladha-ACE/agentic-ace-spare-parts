"""Video ingestion pipeline: frame extraction, keyframe sampling and the
Checkpoint 1 (frame quality) verification gate.

Uses OpenCV. Blur is measured with the variance of the Laplacian - a low
variance means the frame lacks high-frequency edge content, i.e. it is out
of focus / motion-blurred.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import settings
from src.models.schemas import FrameQualityReport

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class ExtractedFrame:
    index: int
    timestamp_sec: float
    image: np.ndarray  # BGR, as returned by OpenCV
    quality: FrameQualityReport


class VideoValidationError(ValueError):
    pass


def validate_video_file(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise VideoValidationError(f"Video file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoValidationError(
            f"Unsupported video extension '{p.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return p


def _laplacian_variance(gray_frame: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


def _brightness_mean(gray_frame: np.ndarray) -> float:
    return float(gray_frame.mean())


def assess_frame_quality(
    frame_bgr: np.ndarray, frame_index: int, timestamp_sec: float
) -> FrameQualityReport:
    """Checkpoint 1: Video Frame Quality Check.

    Error conditions: too blurry, too dark/bright, or resolution too low.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    variance = _laplacian_variance(gray)
    brightness = _brightness_mean(gray)

    reasons: list[str] = []
    if variance < settings.blur_variance_threshold:
        reasons.append(
            f"blurry (laplacian_variance={variance:.1f} < {settings.blur_variance_threshold})"
        )
    if brightness < settings.brightness_min:
        reasons.append(f"too dark (brightness={brightness:.1f} < {settings.brightness_min})")
    elif brightness > settings.brightness_max:
        reasons.append(f"overexposed (brightness={brightness:.1f} > {settings.brightness_max})")
    # Compare against the frame's longer edge, not raw width - portrait
    # video (e.g. a phone held vertically) has its "width" as the short
    # edge, which would otherwise fail this gate even at high resolution.
    longer_edge = max(width, height)
    if longer_edge < settings.min_resolution_width:
        reasons.append(f"low resolution (longer edge={longer_edge} < {settings.min_resolution_width})")

    passed = len(reasons) == 0
    return FrameQualityReport(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        laplacian_variance=variance,
        brightness_mean=brightness,
        width=width,
        height=height,
        passed=passed,
        reason="; ".join(reasons) if reasons else None,
    )


def extract_keyframes(video_path: str | Path) -> list[ExtractedFrame]:
    """Samples frames at a fixed interval, runs Checkpoint 1 on each, and
    keeps only frames that pass quality gating (auto-discard correction).

    If a sampled frame fails, the correction strategy is to *skip forward*
    to the next sample point rather than block the whole pipeline - the UI
    layer surfaces `discarded_frame_count` / per-frame `reason`s so the user
    can be prompted to re-upload higher quality footage for that timestamp
    range if too many frames are discarded (see agents/perception_agent.py).
    """
    path = validate_video_file(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoValidationError(f"OpenCV could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps * settings.keyframe_sample_interval_sec)))

    kept: list[ExtractedFrame] = []
    all_reports: list[FrameQualityReport] = []
    frame_idx = 0
    sample_idx = 0

    while cap.isOpened() and len(kept) < settings.max_keyframes:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            quality = assess_frame_quality(frame, sample_idx, timestamp)
            all_reports.append(quality)
            if quality.passed:
                kept.append(
                    ExtractedFrame(index=sample_idx, timestamp_sec=timestamp, image=frame, quality=quality)
                )
            else:
                logger.info("Discarding frame %s @ %.2fs: %s", sample_idx, timestamp, quality.reason)
            sample_idx += 1
        frame_idx += 1

    cap.release()

    if not kept:
        logger.warning(
            "No frames passed the quality gate for %s (%d sampled, all discarded).",
            path,
            len(all_reports),
        )

    # Stash all reports (including discarded) on each kept frame's neighbours
    # by returning them alongside via module-level attribute isn't ideal;
    # callers that need the *full* report list should call `probe_frame_reports`.
    return kept


def probe_frame_reports(video_path: str | Path) -> list[FrameQualityReport]:
    """Runs Checkpoint 1 over every sampled frame and returns *all* reports
    (pass + fail), independent of `extract_keyframes`'s kept-frame list.
    Used by the API/UI to show the user exactly which timestamps were bad.
    """
    path = validate_video_file(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoValidationError(f"OpenCV could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps * settings.keyframe_sample_interval_sec)))

    reports: list[FrameQualityReport] = []
    frame_idx = 0
    sample_idx = 0
    while cap.isOpened() and sample_idx < settings.max_keyframes:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            reports.append(assess_frame_quality(frame, sample_idx, timestamp))
            sample_idx += 1
        frame_idx += 1
    cap.release()
    return reports
