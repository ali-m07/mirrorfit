"""Tests for body-mask integration with garment rendering.

The renderer accepts an optional float32 person mask and uses it to clip
the garment / mesh so it doesn't spill onto the background or face.
"""

from __future__ import annotations

import numpy as np

from src.utils.config_loader import RenderingSection
from src.vision.clothing_renderer import ClothingRenderer
from src.vision.pose_estimator import (
    Landmark, PoseResult, LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER,
    LM_LEFT_HIP, LM_RIGHT_HIP,
)


def _pose():
    return PoseResult(landmarks={
        LM_LEFT_SHOULDER: Landmark(280, 200, 0.99),
        LM_RIGHT_SHOULDER: Landmark(520, 200, 0.99),
        LM_LEFT_HIP: Landmark(280, 480, 0.99),
        LM_RIGHT_HIP: Landmark(520, 480, 0.99),
    }, frame_size=(800, 600))


def _body_mask():
    """A narrow vertical band mask (body shape simplified)."""
    mask = np.zeros((600, 800), dtype=np.float32)
    mask[100:550, 320:480] = 1.0
    return mask


def test_render_without_body_mask_draws_widely():
    """Without a body mask, the garment covers its full warped box."""
    pose = _pose()
    renderer = ClothingRenderer(RenderingSection())
    renderer.config.shadow.enabled = False
    g = np.zeros((250, 350, 4), dtype=np.uint8)
    g[:, :, :3] = 255
    g[:, :, 3] = 220
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    renderer.render(frame, g, pose)
    assert (frame > 0).any(axis=2).sum() > 0


def test_render_with_body_mask_clips_to_body():
    """With a narrow body mask, far fewer pixels get painted."""
    pose = _pose()
    renderer = ClothingRenderer(RenderingSection())
    renderer.config.shadow.enabled = False
    g = np.zeros((250, 350, 4), dtype=np.uint8)
    g[:, :, :3] = 255
    g[:, :, 3] = 220
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    renderer.render(frame, g, pose, body_mask=_body_mask())
    assert (frame > 0).any(axis=2).sum() > 0


def test_body_mask_reduces_painted_area():
    """Body-mask integration must clip the garment to the body contour."""
    pose = _pose()
    renderer = ClothingRenderer(RenderingSection())
    renderer.config.shadow.enabled = False
    g = np.zeros((250, 350, 4), dtype=np.uint8)
    g[:, :, :3] = 255
    g[:, :, 3] = 220
    frame_no = np.zeros((600, 800, 3), dtype=np.uint8)
    renderer.render(frame_no, g, pose)
    frame_with = np.zeros((600, 800, 3), dtype=np.uint8)
    renderer.render(frame_with, g, pose, body_mask=_body_mask())
    px_no = int((frame_no > 0).any(axis=2).sum())
    px_with = int((frame_with > 0).any(axis=2).sum())
    assert px_with < px_no * 0.85, (
        f"body mask removed {px_no - px_with} pixels (expected at least "
        f"15% reduction); without={px_no}, with={px_with}"
    )


def test_apply_body_mask_zero_outside():
    """Direct unit test: alpha outside the body must become zero."""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[20:80, 30:60] = 1.0     # body
    layer = np.zeros((100, 100, 4), dtype=np.uint8)
    layer[:, :, 3] = 255         # full opacity everywhere
    out = ClothingRenderer._apply_body_mask(layer, mask)
    # Body interior retains full alpha.
    assert out[50, 45, 3] == 255
    # Outside body, alpha collapses to zero.
    assert out[5, 5, 3] == 0
    assert out[95, 95, 3] == 0
    assert out[10, 90, 3] == 0


def test_render_mesh_skips_faces_outside_body():
    """Mesh renderer must drop faces whose centroids aren't on the body."""
    pose = _pose()
    renderer = ClothingRenderer(RenderingSection())
    # Hand-craft a single square quad 100x100 at the corner of the frame.
    faces = [(0, 1, 2), (0, 2, 3)]
    vertices = np.array([
        [0.0, 0.0, 0.0],   # TL
        [1.0, 0.0, 0.0],   # TR
        [1.0, 1.0, 0.0],   # BR
        [0.0, 1.0, 0.0],   # BL
    ], dtype=np.float32)

    # Pose that maps the quad to a corner (x=10..110, y=10..110) - entirely
    # outside the body mask.
    pose_corner = PoseResult(landmarks={
        LM_LEFT_SHOULDER: Landmark(0, 0, 0.99),
        LM_RIGHT_SHOULDER: Landmark(110, 0, 0.99),
        LM_LEFT_HIP: Landmark(0, 110, 0.99),
        LM_RIGHT_HIP: Landmark(110, 110, 0.99),
    }, frame_size=(600, 800))
    body_mask = np.zeros((600, 800), dtype=np.float32)
    body_mask[200:550, 300:520] = 1.0    # mask covers the centre only

    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    # Without body mask the mesh paints the quad.
    frame_no_mask = frame.copy()
    renderer.render_mesh(frame_no_mask, vertices, faces, pose_corner)
    painted_no_mask = (frame_no_mask > 0).any(axis=2).sum()

    # With body mask the quad is fully outside the body -> nothing drawn.
    frame_with_mask = frame.copy()
    renderer.render_mesh(frame_with_mask, vertices, faces, pose_corner,
                         body_mask=body_mask)
    painted_with_mask = (frame_with_mask > 0).any(axis=2).sum()

    assert painted_no_mask > 0, "baseline render expected to paint"
    assert painted_with_mask == 0, (
        f"expected mesh outside body to be skipped, got "
        f"{painted_with_mask} pixels painted"
    )


def test_apply_body_mask_returns_input_when_none():
    """Passing body_mask=None must leave the layer untouched."""
    layer = np.zeros((50, 50, 4), dtype=np.uint8)
    layer[:, :, 3] = 200
    out = ClothingRenderer._apply_body_mask(layer, None)
    assert (out == layer).all()
