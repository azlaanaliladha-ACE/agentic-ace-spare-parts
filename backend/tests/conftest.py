from __future__ import annotations

import numpy as np
import pytest

# 640x480 keeps every synthetic frame above settings.min_resolution_width (480px)
# so Checkpoint 1's resolution gate doesn't confound the blur/brightness assertions.
_HEIGHT, _WIDTH = 480, 640


@pytest.fixture
def sharp_bright_frame() -> np.ndarray:
    """A high-frequency checkerboard pattern -> high Laplacian variance, so
    it should pass the Checkpoint 1 blur gate."""
    frame = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    frame[::2, ::2] = 220
    frame[1::2, 1::2] = 200
    frame[:] += 40  # keep mean brightness inside the acceptable band
    return frame


@pytest.fixture
def blurry_frame() -> np.ndarray:
    """A flat, uniform frame -> ~zero Laplacian variance -> should fail
    the Checkpoint 1 blur gate."""
    frame = np.full((_HEIGHT, _WIDTH, 3), 128, dtype=np.uint8)
    return frame


@pytest.fixture
def dark_frame() -> np.ndarray:
    frame = np.full((_HEIGHT, _WIDTH, 3), 5, dtype=np.uint8)
    return frame


@pytest.fixture
def synthetic_video(tmp_path):
    """Writes a short synthetic .mp4 with a mix of sharp and blurry frames,
    for exercising the full extraction pipeline end to end.
    """
    import cv2

    path = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (_WIDTH, _HEIGHT))

    rng = np.random.default_rng(42)
    # Period (40 frames) deliberately does NOT divide the ~30-frame sample
    # stride the pipeline uses (1s @ 30fps), so the samples taken at
    # frames 0/30/60 land on a mix of noisy (sharp) and flat (blurry)
    # stretches instead of always hitting the same phase.
    for i in range(90):
        if i % 40 < 25:
            frame = rng.integers(0, 255, (_HEIGHT, _WIDTH, 3), dtype=np.uint8)
        else:
            frame = np.full((_HEIGHT, _WIDTH, 3), 128, dtype=np.uint8)  # blurry/flat stretch
        writer.write(frame)
    writer.release()
    return str(path)
