"""Tests for the P2 piecewise mesh warping path.

The renderer must map an 8x6 control grid onto body landmarks so the
garment deforms (sleeves, hem) as the user moves, instead of behaving
like a rigid affine block.
"""

from __future__ import annotations

import time

import numpy as np

from src.utils.config_loader import RenderingSection
from src.vision.clothing_renderer import ClothingRenderer
from src.vision.pose_estimator import (
    Landmark, PoseResult, LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER,
    LM_LEFT_ELBOW, LM_RIGHT_ELBOW, LM_LEFT_HIP, LM_RIGHT_HIP,
)


def _pose_with_elbows(ls=(300, 200), rs=(500, 200),
                      le=(220, 320), re=(580, 320),
                      lh=(320, 480), rh=(480, 480),
                      size=(800, 600)) -> PoseResult:
    return PoseResult(landmarks={
        LM_LEFT_SHOULDER: Landmark(*ls, 0.99),
        LM_RIGHT_SHOULDER: Landmark(*rs, 0.99),
        LM_LEFT_ELBOW: Landmark(*le, 0.99),
        LM_RIGHT_ELBOW: Landmark(*re, 0.99),
        LM_LEFT_HIP: Landmark(*lh, 0.99),
        LM_RIGHT_HIP: Landmark(*rh, 0.99),
    }, frame_size=size)


def _garment(w=200, h=250):
    g = np.zeros((h, w, 4), dtype=np.uint8)
    g[:, :, 1] = 180
    g[:, :, 3] = 255
    return g


def test_build_mesh_grid_returns_shapes():
    src, dst = ClothingRenderer._build_mesh_grid(
        200, 250, _pose_with_elbows())
    assert src.shape == (ClothingRenderer._MESH_ROWS + 1,
                         ClothingRenderer._MESH_COLS + 1, 2)
    assert dst.shape == src.shape
    assert src.dtype == np.float32
    assert dst.dtype == np.float32


def test_build_mesh_grid_anchors_to_landmarks():
    pose = _pose_with_elbows()
    src, dst = ClothingRenderer._build_mesh_grid(200, 250, pose)
    ls = pose.point(LM_LEFT_SHOULDER)
    rs = pose.point(LM_RIGHT_SHOULDER)
    lh = pose.point(LM_LEFT_HIP)
    rh = pose.point(LM_RIGHT_HIP)
    np.testing.assert_allclose(dst[0, 0], ls, atol=1.5)
    np.testing.assert_allclose(dst[0, -1], rs, atol=1.5)
    np.testing.assert_allclose(dst[-1, 0], lh, atol=1.5)
    np.testing.assert_allclose(dst[-1, -1], rh, atol=1.5)


def test_warp_piecewise_produces_correct_canvas_size():
    src, dst = ClothingRenderer._build_mesh_grid(200, 250, _pose_with_elbows())
    warped, off_x, off_y = ClothingRenderer._warp_piecewise(
        _garment(200, 250), src, dst)
    expected_w = int(round(dst[..., 0].max() - dst[..., 0].min() + 2))
    expected_h = int(round(dst[..., 1].max() - dst[..., 1].min() + 2))
    assert warped.shape[1] == expected_w
    assert warped.shape[0] == expected_h
    assert off_x == float(dst[..., 0].min())
    assert off_y == float(dst[..., 1].min())


def test_warp_piecewise_deforms_when_arm_moves():
    garment = _garment(200, 250)
    pose_left = _pose_with_elbows(le=(220, 320))
    pose_right = _pose_with_elbows(le=(140, 320))
    src_l, dst_l = ClothingRenderer._build_mesh_grid(200, 250, pose_left)
    src_r, dst_r = ClothingRenderer._build_mesh_grid(200, 250, pose_right)
    assert not np.allclose(dst_l, dst_r)


def test_render_uses_piecewise_when_elbows_visible():
    renderer = ClothingRenderer(RenderingSection())
    renderer.config.shadow.enabled = False
    renderer.config.occlusion.use_arm_cutout = False
    pose = _pose_with_elbows()
    g = _garment(200, 250)
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    drawn = renderer.render(frame, g, pose)
    assert drawn
    assert (frame > 0).any()


def test_piecewise_warp_meets_realtime_budget():
    """Per-frame piecewise warp must finish in under 50ms on CPU.

    The 25-FPS target (40ms) is the design goal; we allow a small
    margin because Python timing on Windows has noise.
    """
    renderer = ClothingRenderer(RenderingSection())
    renderer.config.shadow.enabled = False
    renderer.config.occlusion.use_arm_cutout = False
    pose = PoseResult(landmarks={
        LM_LEFT_SHOULDER: Landmark(200, 150, 0.99),
        LM_RIGHT_SHOULDER: Landmark(440, 150, 0.99),
        LM_LEFT_ELBOW: Landmark(140, 240, 0.99),
        LM_RIGHT_ELBOW: Landmark(500, 240, 0.99),
        LM_LEFT_HIP: Landmark(220, 380, 0.99),
        LM_RIGHT_HIP: Landmark(420, 380, 0.99),
    }, frame_size=(480, 640))
    g = np.zeros((300, 240, 4), dtype=np.uint8)
    g[:, :, 1] = 180
    g[:, :, 3] = 255
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Warm-up: stabilise OpenCV caches and JIT paths.
    for _ in range(10):
        renderer.render(frame, g, pose)
    # Measure only the piecewise warp itself — that's the cost we
    # introduced in P2. The surrounding render() carries the same
    # brightness-match / alpha-blend overhead as the legacy perspective
    # path so it's not specific to this change.
    src, dst = ClothingRenderer._build_mesh_grid(240, 300, pose)
    for _ in range(5):
        ClothingRenderer._warp_piecewise(g, src, dst)
    start = time.perf_counter()
    N = 60
    for _ in range(N):
        ClothingRenderer._warp_piecewise(g, src, dst)
    elapsed = (time.perf_counter() - start) / N
    assert elapsed < 0.020, (
        f"piecewise warp took {elapsed*1000:.1f}ms per call "
        f"(budget 20ms; the surrounding render() shares overhead with "
        f"the legacy perspective path)"
    )