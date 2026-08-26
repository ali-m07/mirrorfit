"""Tests for the perspective-warp (homography) rendering path.

These tests cover the 3D-feel fix: when both shoulders and both hips are
tracked at high confidence, the renderer maps the garment box onto the
torso trapezoid using a 3x3 perspective transform. When any landmark is
unreliable, it falls back to the legacy affine-only transform.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.utils.config_loader import RenderingSection
from src.vision.clothing_renderer import ClothingRenderer
from src.vision.pose_estimator import (
    Landmark, PoseResult, LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER,
    LM_LEFT_HIP, LM_RIGHT_HIP,
)


def _pose(ls=(300, 200), rs=(500, 200), lh=(320, 480), rh=(480, 480),
          size=(800, 600)) -> PoseResult:
    return PoseResult(landmarks={
        LM_LEFT_SHOULDER: Landmark(*ls, 0.99),
        LM_RIGHT_SHOULDER: Landmark(*rs, 0.99),
        LM_LEFT_HIP: Landmark(*lh, 0.99),
        LM_RIGHT_HIP: Landmark(*rh, 0.99),
    }, frame_size=size)


def _garment(w=200, h=250):
    g = np.zeros((h, w, 4), dtype=np.uint8)
    g[:, :, 1] = 180
    g[:, :, 3] = 255
    return g


def test_homography_present_with_all_four_landmarks():
    """All four torso landmarks confident -> placement carries a 3x3 warp."""
    renderer = ClothingRenderer(RenderingSection())
    placement = renderer.compute_placement(_pose(), _garment().shape[:2])
    assert placement is not None
    assert placement.homography is not None
    assert placement.homography.shape == (3, 3)
    assert placement.homography[2, 2] == pytest.approx(1.0)


def test_homography_absent_when_hips_unreliable():
    """Hips below visibility threshold -> renderer falls back to affine."""
    pose = PoseResult(landmarks={
        LM_LEFT_SHOULDER: Landmark(300, 200, 0.99),
        LM_RIGHT_SHOULDER: Landmark(500, 200, 0.99),
        LM_LEFT_HIP: Landmark(320, 480, 0.05),
        LM_RIGHT_HIP: Landmark(480, 480, 0.05),
    }, frame_size=(800, 600))
    renderer = ClothingRenderer(RenderingSection())
    placement = renderer.compute_placement(pose, _garment().shape[:2])
    assert placement is not None
    assert placement.homography is None


def test_homography_maps_corners_to_landmarks():
    """The warp must map garment box corners to the tracked landmarks."""
    pose = _pose(ls=(300, 200), rs=(500, 200), lh=(320, 480), rh=(480, 480))
    renderer = ClothingRenderer(RenderingSection())
    placement = renderer.compute_placement(pose, _garment().shape[:2])
    assert placement is not None and placement.homography is not None
    gw, gh = placement.size
    H = placement.homography
    corners = np.array([[0, 0], [gw, 0], [gw, gh], [0, gh]],
                       dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    np.testing.assert_allclose(warped[0], pose.point(LM_LEFT_SHOULDER), atol=1.5)
    np.testing.assert_allclose(warped[1], pose.point(LM_RIGHT_SHOULDER), atol=1.5)
    np.testing.assert_allclose(warped[2], pose.point(LM_RIGHT_HIP), atol=1.5)
    np.testing.assert_allclose(warped[3], pose.point(LM_LEFT_HIP), atol=1.5)


def test_render_warps_garment_to_landmark_pixels():
    """The perspective warp must pull the garment onto the tracked body.

    We render with the four landmarks defining a NARROW trapezoid (10 px
    wide at the shoulders and 10 px wide at the hips). The resulting
    garment must be confined to a narrow vertical band on the frame —
    not smeared across the entire width as the legacy affine transform
    would.
    """
    # Narrow trapezoid: 30 px shoulder span, 30 px hip span — narrower
    # than the garment's intrinsic box, so a perspective warp will squash
    # the painted region while an affine transform will smear it wide.
    pose = _pose(ls=(385, 200), rs=(415, 200), lh=(385, 400), rh=(415, 400))
    renderer = ClothingRenderer(RenderingSection())
    renderer.config.shadow.enabled = False
    renderer.config.occlusion.use_arm_cutout = False
    placement = renderer.compute_placement(pose, _garment(200, 250).shape[:2])
    assert placement is not None and placement.homography is not None

    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    drawn = renderer.render(frame, _garment(200, 250), pose)
    assert drawn

    nonzero_per_col = (frame.max(axis=2) > 0).sum(axis=0)
    band_cols = np.where(nonzero_per_col > 0)[0]
    assert band_cols.size > 0, "no pixels were drawn at all"
    span = int(band_cols.max() - band_cols.min())
    # Working warp paints ~30-60 px wide (10 px landmark span +
    # interpolation kernel); broken affine would smear to >150 px.
    assert span < 100, (
        f"warped garment covers {span}px wide — perspective path not used"
    )


def test_render_falls_back_to_affine_when_no_homography():
    """When the homography is None, the legacy affine path must still draw."""
    pose_low_hips = PoseResult(landmarks={
        LM_LEFT_SHOULDER: Landmark(300, 200, 0.99),
        LM_RIGHT_SHOULDER: Landmark(500, 200, 0.99),
        LM_LEFT_HIP: Landmark(320, 480, 0.05),
        LM_RIGHT_HIP: Landmark(480, 480, 0.05),
    }, frame_size=(800, 600))
    renderer = ClothingRenderer(RenderingSection())
    renderer.config.shadow.enabled = False
    renderer.config.occlusion.use_arm_cutout = False
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    drawn = renderer.render(frame, _garment(), pose_low_hips)
    assert drawn
    assert (frame > 0).any()