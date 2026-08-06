"""Tests for garment placement and rendering."""

from __future__ import annotations

import numpy as np

from src.utils.config_loader import RenderingSection
from src.utils.helpers import ClothingItem
from src.vision.clothing_renderer import ClothingRenderer
from src.vision.pose_estimator import (
    Landmark, PoseResult, LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER,
    LM_LEFT_HIP, LM_RIGHT_HIP,
)
from pathlib import Path


def _pose(ls=(300, 200), rs=(500, 200), lh=(320, 480), rh=(480, 480),
          size=(800, 600)) -> PoseResult:
    return PoseResult(landmarks={
        LM_LEFT_SHOULDER: Landmark(*ls, 0.99),
        LM_RIGHT_SHOULDER: Landmark(*rs, 0.99),
        LM_LEFT_HIP: Landmark(*lh, 0.99),
        LM_RIGHT_HIP: Landmark(*rh, 0.99),
    }, frame_size=size)


def _garment(w: int = 200, h: int = 250) -> np.ndarray:
    g = np.zeros((h, w, 4), dtype=np.uint8)
    g[:, :, 1] = 180
    g[:, :, 3] = 255
    return g


def test_placement_scales_with_shoulder_width() -> None:
    renderer = ClothingRenderer(RenderingSection())
    near = renderer.compute_placement(_pose(ls=(300, 200), rs=(500, 200)),
                                      _garment().shape[:2])
    far = renderer.compute_placement(_pose(ls=(380, 200), rs=(420, 200)),
                                     _garment().shape[:2])
    assert near is not None and far is not None
    assert near.size[0] > far.size[0]


def test_placement_none_when_too_far() -> None:
    renderer = ClothingRenderer(RenderingSection())
    tiny = _pose(ls=(395, 200), rs=(405, 200))  # 10 px shoulder width
    assert renderer.compute_placement(tiny, _garment().shape[:2]) is None


def test_placement_tracks_shoulder_tilt() -> None:
    renderer = ClothingRenderer(RenderingSection())
    tilted = _pose(ls=(300, 180), rs=(500, 240))
    placement = renderer.compute_placement(tilted, _garment().shape[:2])
    assert placement is not None
    assert placement.angle_deg > 5  # follows the tilt


def test_rotation_is_clamped() -> None:
    cfg = RenderingSection(max_rotation_deg=20.0)
    renderer = ClothingRenderer(cfg)
    extreme = _pose(ls=(300, 100), rs=(500, 400))
    placement = renderer.compute_placement(extreme, _garment().shape[:2])
    assert placement is not None
    assert abs(placement.angle_deg) <= 20.0


def test_render_draws_opaque_pixels() -> None:
    renderer = ClothingRenderer(RenderingSection())
    renderer.config.shadow.enabled = False
    renderer.config.occlusion.use_arm_cutout = False
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    drawn = renderer.render(frame, _garment(), _pose())
    assert drawn
    assert (frame[:, :, 1] > 100).any()


def test_render_without_pose_is_noop() -> None:
    renderer = ClothingRenderer(RenderingSection())
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    assert not renderer.render(frame, _garment(), None)
    assert not frame.any()


def test_render_dress_extends_toward_hips() -> None:
    # A pose with a very long torso, where the dress rule exceeds the default
    # aspect-ratio height and actually engages. shoulders=200px wide, torso
    # drop=560px -> dress floor = 560*1.35 = 756 > 562 (aspect height).
    long_pose = _pose(ls=(300, 80), rs=(500, 80), lh=(320, 640), rh=(480, 640))
    renderer = ClothingRenderer(RenderingSection())
    item = ClothingItem(id="d", name="dress", filename="d.png",
                        path=Path("d.png"), category="dress")
    placement = renderer.compute_placement(long_pose, _garment().shape[:2], item)
    base = renderer.compute_placement(long_pose, _garment().shape[:2], None)
    assert placement is not None and base is not None
    assert placement.size[1] > base.size[1]
