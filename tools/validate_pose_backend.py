"""Validate the ONNX pose backend on a real photo (needs onnxruntime).

Checks that the RTMO graph runs, outputs have the expected layout, and the
mapped landmarks land inside the frame. Also prints quick latency numbers::

    python tools/validate_pose_backend.py
    python tools/validate_pose_backend.py --image path/to/person.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.utils.config_loader import load_config  # noqa: E402
from src.vision.pose_backend_onnx import OnnxPoseBackend, MODEL_URL  # noqa: E402

SAMPLE_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=None,
                        help="Photo containing a person (default: downloads a sample)")
    parser.add_argument("--device", default="auto", choices=["auto", "dml", "cpu"])
    args = parser.parse_args()

    cfg = load_config()
    section = cfg.vision.pose.onnx
    section.device = args.device
    backend = OnnxPoseBackend(section)
    backend.ensure_session()
    print(f"backend: {backend.backend_name} | providers: {backend._session.get_providers()}")

    for o in backend._session.get_outputs():
        print(f"  model output: {o.name} {o.shape}")

    image_path = args.image
    if image_path is None:
        image_path = Path(".wheels/sample_people.jpg")
        if not image_path.is_file():
            from urllib.request import urlopen
            image_path.parent.mkdir(exist_ok=True)
            image_path.write_bytes(urlopen(SAMPLE_URL, timeout=60).read())
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Could not read {image_path}")
        return 2
    # Resize big photos to a webcam-like size for realistic latency.
    scale = 640.0 / max(frame.shape[:2])
    if scale < 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

    landmarks = backend.process(frame)
    if landmarks is None:
        print("NO PERSON DETECTED — check thresholds or the input photo")
        return 1

    h, w = frame.shape[:2]
    print(f"person detected, {len(landmarks)} mapped landmarks:")
    for idx, (x, y, vis) in sorted(landmarks.items()):
        inside = 0 <= x < w and 0 <= y < h
        flag = "" if inside else "  <-- OUTSIDE FRAME"
        print(f"  mp#{idx:<3} ({x:7.1f}, {y:7.1f}) vis={vis:.2f}{flag}")

    # Latency (warm).
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        backend.process(frame)
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"latency: {ms:.1f} ms/frame  (~{1000 / ms:.0f} FPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
