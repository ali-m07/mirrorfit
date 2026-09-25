"""Benchmark the real-time pipeline: pose backends, segmentation, renderer.

Measures per-stage latency on this machine so the right ``vision.pose.backend``
default can be chosen honestly, instead of guessing. Run it on the target
deployment hardware::

    python tools/benchmark.py               # all backends found on this machine
    python tools/benchmark.py --frames 60   # longer, steadier measurement
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
from src.utils.helpers import ClothingCatalog  # noqa: E402
from src.vision.clothing_renderer import ClothingRenderer  # noqa: E402
from src.vision.pose_estimator import (  # noqa: E402
    LM_LEFT_HIP, LM_LEFT_SHOULDER, LM_LEFT_WRIST, LM_RIGHT_HIP,
    LM_RIGHT_SHOULDER, LM_RIGHT_WRIST, PoseResult, Landmark,
)


def _synthetic_pose(w: int, h: int) -> PoseResult:
    """A believable frontal torso for exercising the render path."""
    pts = {
        LM_LEFT_SHOULDER: ((int(w * 0.33), int(h * 0.28)), 0.95),
        LM_RIGHT_SHOULDER: ((int(w * 0.67), int(h * 0.28)), 0.95),
        LM_LEFT_HIP: ((int(w * 0.38), int(h * 0.62)), 0.9),
        LM_RIGHT_HIP: ((int(w * 0.62), int(h * 0.62)), 0.9),
        LM_LEFT_WRIST: ((int(w * 0.33) - 30, int(h * 0.28) + 200), 0.8),
        LM_RIGHT_WRIST: ((int(w * 0.67) + 30, int(h * 0.28) + 200), 0.8),
    }
    landmarks = {idx: Landmark(x=float(x), y=float(y), visibility=vis)
                 for idx, ((x, y), vis) in pts.items()}
    return PoseResult(landmarks=landmarks, frame_size=(w, h), tracked=True)


def _timed(name: str, fn, frames: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(frames):
        fn()
    elapsed = time.perf_counter() - start
    ms = elapsed / frames * 1000.0
    print(f"  {name:<38} {ms:8.1f} ms/frame   {1.0 / (elapsed / frames):6.1f} FPS")
    return ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=40)
    args = parser.parse_args()

    cfg = load_config()
    w, h = int(cfg.camera.width), int(cfg.camera.height)
    # Photo-LIKE frames: pure noise inflates NMS work ~8x and skews every
    # number. A smooth gradient + soft blobs approximates a real webcam view.
    grad = np.tile(np.linspace(40, 160, h, dtype=np.uint8)[:, None, None], (1, w, 3))
    frames = []
    for i in range(4):
        f = grad + ((i * 7) % 25)
        cv2.circle(f, (w // 2, h // 2), int(h * 0.3), (90, 110, 150), -1)
        cv2.circle(f, (w // 2, int(h * 0.18)), int(h * 0.1), (140, 150, 170), -1)
        frames.append(np.ascontiguousarray(f, dtype=np.uint8))

    print(f"MirrorFit benchmark — {w}x{h}, {args.frames} frames per stage\n")

    # --- pose backends -------------------------------------------------
    import mediapipe as mp
    for complexity in (1, 0):
        pose = mp.solutions.pose.Pose(static_image_mode=False,
                                      model_complexity=complexity,
                                      smooth_landmarks=True)
        idx = {"i": 0}

        def run_mp(pose=pose, idx=idx):
            frame = frames[idx["i"] % len(frames)]
            idx["i"] += 1
            pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        _timed(f"MediaPipe Pose CPU (complexity={complexity})", run_mp, args.frames)
        pose.close()

    try:
        import onnxruntime as ort
        from src.vision.pose_backend_onnx import OnnxPoseBackend
        model_path = cfg.resolve_path(cfg.vision.pose.onnx.model_path)
        if not model_path.is_file():
            print(f"  ONNX model missing at {model_path} — run the app once "
                  "or set vision.pose.onnx.auto_download")
            model_path = None
        if model_path is not None:
            size = int(cfg.vision.pose.onnx.input_size)
            for device, providers in (
                ("CPU", ["CPUExecutionProvider"]),
                ("DirectML (GPU)",
                 ["DmlExecutionProvider", "CPUExecutionProvider"]),
            ):
                if not all(p in ort.get_available_providers() for p in providers):
                    continue
                try:
                    backend = OnnxPoseBackend(cfg.vision.pose.onnx)
                    import onnxruntime as _ort
                    backend._session = _ort.InferenceSession(str(model_path),
                                                             providers=providers)
                    backend._input_name = backend._session.get_inputs()[0].name
                    idx = {"i": 0}

                    def run_onnx(backend=backend, idx=idx):
                        frame = frames[idx["i"] % len(frames)]
                        idx["i"] += 1
                        backend.process(frame)

                    _timed(f"RTMO-s ONNX {device} (input {size}px)", run_onnx, args.frames)
                    backend.close()
                except Exception as exc:
                    print(f"  RTMO-s ONNX {device}: unavailable ({exc})")
    except ImportError:
        print("  onnxruntime not installed — skipping ONNX backends")

    # --- segmentation ---------------------------------------------------
    try:
        from src.vision.segmenter import PersonSegmenter
        seg = PersonSegmenter(cfg.vision.segmentation)
        idx = {"i": 0}

        def run_seg(idx={"i": 0}):
            frame = frames[idx["i"] % len(frames)]
            idx["i"] += 1
            seg.process(frame)

        seg_ms = _timed("Selfie segmentation (every frame)", run_seg, args.frames)
        interval = max(1, int(getattr(cfg.vision.segmentation, "interval", 1)))
        if interval > 1:
            print(f"  {'  -> amortised at interval=%d' % interval:<38}"
                  f" {seg_ms / interval:8.1f} ms/frame equivalent")
        seg.close()
    except Exception as exc:
        print(f"  Segmentation: unavailable ({exc})")

    # --- renderer -------------------------------------------------------
    catalog = ClothingCatalog(cfg.resolve_path(cfg.clothes.directory),
                              catalog_file=cfg.clothes.catalog_file)
    renderer = ClothingRenderer(cfg.rendering, cfg.vision.tracking.min_visibility)
    pose = _synthetic_pose(w, h)
    png_items = [it for it in catalog.items if it.path.suffix.lower() == ".png"]
    if png_items:
        garment = catalog.load_image(png_items[0])
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        def run_render():
            canvas[:] = 60
            renderer.render(canvas, garment, pose, png_items[0])

        _timed("Garment render (pose->composite)", run_render, args.frames)
    else:
        print("  Renderer: no garments in catalog")

    print("\nPick the fastest pose backend in config.yaml -> vision.pose.backend.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
