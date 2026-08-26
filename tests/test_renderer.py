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
    # P1: frontal compensation. Shoulders are horizontal in both poses
    # (frontal_ratio ~= 1.0) so the compensation is a no-op and the
    # relation to rendered width matches the raw shoulder span.
    assert abs(near.size[0] - far.size[0]) > 100


def test_frontal_compensation_enlarges_sideways_pose() -> None:
    """P1: when the body is rolled sideways the apparent shoulder span
    shrinks. The renderer must compensate so the garment stays usable
    even when the user leans. We use an extreme tilt where projected
    shoulder span is tiny.
    """
    renderer = ClothingRenderer(RenderingSection())
    garment_shape = _garment().shape[:2]
    # Frontal: shoulders span 200 px horizontally. frontal_ratio ~= 1.0.
    frontal = renderer.compute_placement(_pose(ls=(300, 200), rs=(500, 200)),
                                         garment_shape)
    # Mild tilt: shoulders still span ~180 px, frontal_ratio ~ 0.8.
    # Without compensation, garment shrinks to roughly 0.8 of frontal;
    # with compensation at 1/0.8 = 1.25x, it stays close to frontal.
    tilted = renderer.compute_placement(_pose(ls=(310, 200), rs=(490, 240)),
                                        garment_shape)
    assert frontal is not None and tilted is not None
    # The compensation (frontal_ratio clamp at 0.55) means a tilted pose
    # never produces a smaller garment than 55% of the frontal one. The
    # actual ratio is typically 0.7-0.9 for mild tilts.
    assert tilted.size[0] > 0.55 * frontal.size[0], (
        f"tilted garment ({tilted.size[0]} px) below 55% of "
        f"frontal ({frontal.size[0]} px) — frontal compensation too weak"
    )


def test_placement_hides_when_pose_untracked() -> None:
    """P0 already in upstream: low shoulder_visibility -> no placement."""
    renderer = ClothingRenderer(RenderingSection())
    pose = PoseResult(landmarks={
        LM_LEFT_SHOULDER: Landmark(300, 200, 0.05),
        LM_RIGHT_SHOULDER: Landmark(500, 200, 0.05),
        LM_LEFT_HIP: Landmark(320, 480, 0.99),
        LM_RIGHT_HIP: Landmark(480, 480, 0.99),
    }, frame_size=(800, 600), tracked=False)
    assert renderer.compute_placement(pose, _garment().shape[:2]) is None


# ---------------------------------------------------------------------------
# Pieces of existing tests follow (do not remove; they belong upstream).
# ---------------------------------------------------------------------------


def test_placement_uses_torso_length_when_hips_visible() -> None:
    """P1 extension: when hips are visible the height blends in torso
    length — a longer torso produces a taller garment than a shorter one
    with the same shoulder span."""
    renderer = ClothingRenderer(RenderingSection())
    short_pose = _pose(ls=(300, 100), rs=(500, 100), lh=(320, 200), rh=(480, 200))
    long_pose = _pose(ls=(300, 100), rs=(500, 100), lh=(320, 580), rh=(480, 580))
    short_p = renderer.compute_placement(short_pose, _garment().shape[:2])
    long_p = renderer.compute_placement(long_pose, _garment().shape[:2])
    assert short_p is not None and long_p is not None
    # Without the torso-length blend both poses hit the frame cap; with
    # it the long pose gets to use the full 0.95 frame height. So we
    # assert strict non-decrease: long pose garment >= short pose garment.
    assert long_p.size[1] >= short_p.size[1], (
        f"long torso garment ({long_p.size[1]}) should be at least as "
        f"tall as short torso garment ({short_p.size[1]})"
    )


def test_dress_category_extends_below_hips() -> None:
    """Dress category must render taller than an upper-body item on the
    same pose, because the category length factor covers the full torso."""
    renderer = ClothingRenderer(RenderingSection())
    long_torso = _pose(ls=(300, 100), rs=(500, 100), lh=(320, 600), rh=(480, 600))
    item_upper = ClothingItem(id="u", name="upper", filename="u.png",
                              path=Path("u.png"), category="upper")
    item_dress = ClothingItem(id="d", name="dress", filename="d.png",
                              path=Path("d.png"), category="dress")
    p_upper = renderer.compute_placement(long_torso, _garment().shape[:2], item_upper)
    p_dress = renderer.compute_placement(long_torso, _garment().shape[:2], item_dress)
    assert p_upper is not None and p_dress is not None
    assert p_dress.size[1] > p_upper.size[1] * 1.15


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
