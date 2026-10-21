"""Generates 10 synthetic demo videos guaranteed to pass Checkpoint 1
(frame quality: sharp, well-lit, sufficient resolution) so a user can
verify the video-ingestion pipeline works before troubleshooting anything
downstream (vision-model classification, which is NOT deterministic).

Each video mixes: large bold high-contrast text (matching real catalog
entries where possible, so a real vision model has legible text to OCR),
a light noise texture (keeps Laplacian variance well above the blur
threshold across every sampled frame, not just the first), and a mix of
landscape/portrait orientations and resolutions to exercise the
orientation-aware resolution check.

Usage: python scripts/generate_demo_videos.py [output_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "SpareParts_Demo_Videos"

VIDEOS = [
    {
        "file": "01_jcb_4cx_boom_cylinder.mp4",
        "size": (960, 540),
        "bg": (60, 90, 150),
        "lines": ["JCB 4CX", "BOOM HYDRAULIC CYLINDER", "ROD 45mm - STANDARD DUTY"],
    },
    {
        "file": "02_jcb_serial_plate.mp4",
        "size": (960, 540),
        "bg": (40, 70, 40),
        "lines": ["JCB BACKHOE LOADER", "SERIAL  JCB4CX0512", "MODEL 4CX"],
    },
    {
        "file": "03_caterpillar_320_excavator.mp4",
        "size": (960, 540),
        "bg": (30, 110, 170),
        "lines": ["CATERPILLAR 320", "TRACKED EXCAVATOR", "ARM / BUCKET CYLINDER"],
    },
    {
        "file": "04_cat_330_boom_portrait.mp4",
        "size": (540, 960),
        "bg": (90, 60, 20),
        "lines": ["CAT 330", "BOOM CYLINDER", "ROD 110mm"],
    },
    {
        "file": "05_case_580_backhoe.mp4",
        "size": (960, 540),
        "bg": (120, 60, 60),
        "lines": ["CASE 580 SUPER N", "BACKHOE LOADER", "WET DISC BRAKE KIT"],
    },
    {
        "file": "06_parker_hydraulic_pump.mp4",
        "size": (960, 540),
        "bg": (80, 80, 20),
        "lines": ["PARKER PVP016", "VARIABLE PISTON PUMP", "SAE A SHAFT"],
    },
    {
        "file": "07_donaldson_filter_portrait.mp4",
        "size": (540, 960),
        "bg": (20, 100, 100),
        "lines": ["DONALDSON P550", "HYDRAULIC RETURN FILTER", "10 MICRON"],
    },
    {
        "file": "08_damaged_hydraulic_hose.mp4",
        "size": (960, 540),
        "bg": (25, 25, 130),
        "lines": ["DAMAGED COMPONENT", "HYDRAULIC HOSE - VISIBLE WEAR", "LEAK AT FITTING"],
    },
    {
        "file": "09_jcb_dipper_cylinder.mp4",
        "size": (960, 540),
        "bg": (70, 100, 30),
        "lines": ["JCB 3CX", "DIPPER ARM CYLINDER", "ROD 60mm"],
    },
    {
        "file": "10_checkpoint1_calibration.mp4",
        "size": (960, 540),
        "bg": (150, 150, 150),
        "lines": ["CHECKPOINT 1 CALIBRATION", "SHARP / BRIGHT / IN-RANGE", "SHOULD ALWAYS PASS"],
    },
]

FPS = 30.0
DURATION_SEC = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX


def render_frame(size_wh: tuple[int, int], bg: tuple[int, int, int], lines: list[str], rng: np.random.Generator) -> np.ndarray:
    w, h = size_wh
    frame = np.full((h, w, 3), bg, dtype=np.uint8)

    # Light noise texture: keeps Laplacian variance comfortably above the
    # blur threshold on every frame without obscuring the text.
    noise = rng.integers(-14, 14, (h, w, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # A machine-plate-style rectangle as a focal element.
    plate_w, plate_h = int(w * 0.72), int(h * 0.5)
    x0, y0 = (w - plate_w) // 2, (h - plate_h) // 2
    cv2.rectangle(frame, (x0, y0), (x0 + plate_w, y0 + plate_h), (235, 235, 235), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + plate_w, y0 + plate_h), (20, 20, 20), 4)

    scale = plate_w / 960.0
    y = y0 + int(plate_h * 0.32)
    for i, line in enumerate(lines):
        font_scale = (1.15 if i == 0 else 0.72) * scale * 1.4
        thickness = 3 if i == 0 else 2
        (tw, th), _ = cv2.getTextSize(line, FONT, font_scale, thickness)
        tx = x0 + (plate_w - tw) // 2
        cv2.putText(frame, line, (tx, y), FONT, font_scale, (15, 15, 15), thickness, cv2.LINE_AA)
        y += int(th * 1.9)

    return frame


def write_video(path: Path, size_wh: tuple[int, int], bg: tuple[int, int, int], lines: list[str]) -> None:
    w, h = size_wh
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    rng = np.random.default_rng(abs(hash(path.name)) % (2**32))
    for _ in range(int(FPS * DURATION_SEC)):
        writer.write(render_frame(size_wh, bg, lines, rng))
    writer.release()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in VIDEOS:
        path = OUT_DIR / spec["file"]
        write_video(path, spec["size"], spec["bg"], spec["lines"])
        print(f"wrote {path} ({spec['size'][0]}x{spec['size'][1]})")
    print(f"\n{len(VIDEOS)} demo videos in {OUT_DIR}")


if __name__ == "__main__":
    main()
