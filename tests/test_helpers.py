"""Tests for alpha compositing and small utilities."""

from __future__ import annotations

import numpy as np

from src.utils.helpers import (
    FPSCounter, alpha_blend, encode_jpeg, letterbox, timestamped_filename,
)


def _overlay(w: int, h: int, alpha: int = 255) -> np.ndarray:
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, :3] = (10, 20, 30)
    img[:, :, 3] = alpha
    return img


def test_alpha_blend_fully_opaque_center() -> None:
    bg = np.full((100, 100, 3), 200, dtype=np.uint8)
    alpha_blend(bg, _overlay(20, 20), 40, 40)
    assert (bg[50, 50] == [10, 20, 30]).all()
    assert bg[10, 10, 0] == 200  # untouched region


def test_alpha_blend_clips_at_edges() -> None:
    bg = np.full((50, 50, 3), 100, dtype=np.uint8)
    alpha_blend(bg, _overlay(40, 40), -20, -20)   # top-left overflow
    alpha_blend(bg, _overlay(40, 40), 30, 30)     # bottom-right overflow
    alpha_blend(bg, _overlay(40, 40), 1000, 0)    # fully outside — no-op
    assert bg.shape == (50, 50, 3)
    assert (bg[5, 5] == [10, 20, 30]).all()


def test_alpha_blend_transparent_is_noop() -> None:
    bg = np.full((30, 30, 3), 77, dtype=np.uint8)
    alpha_blend(bg, _overlay(10, 10, alpha=0), 5, 5)
    assert (bg == 77).all()


def test_timestamped_filename_is_safe() -> None:
    name = timestamped_filename("tryon", "", "png", extra="Navy Hoodie!")
    assert name.startswith("tryon_")
    assert name.endswith("_navy-hoodie.png")
    assert "/" not in name and "\\" not in name


def test_encode_jpeg_roundtrip() -> None:
    import cv2
    frame = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    data = encode_jpeg(frame, 90)
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == frame.shape


def test_letterbox_preserves_aspect() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = letterbox(frame, (200, 200))
    assert out.shape == (200, 200, 3)


def test_fps_counter_ticks() -> None:
    counter = FPSCounter()
    assert counter.tick() == 0.0
    assert counter.tick() >= 0.0
