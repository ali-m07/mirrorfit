"""Tests for the cylindrical torso wrap (wrap contract).

Covers the three shared-contract surfaces:

- ``PoseResult.torso_yaw_deg``: signed yaw from shoulder squeeze + nose offset.
- ``ClothingRenderer._apply_torso_wrap``: remap geometry (shape, identity at
  yaw=0, monotone mapping, edge compression, far-side loss past the visible
  horizon), volume shading, and the guard rails.
- ``ClothingRenderer.compute_placement``: with ``wrap.enabled`` the garment
  box must NOT be divided by ``frontal_ratio`` (it narrows with yaw instead).

Poses are synthetic ``Landmark`` dicts — no camera, no mediapipe import.
"""

from __future__ import annotations

import math

import numpy as np

from src.utils.config_loader import RenderingSection
from src.vision.clothing_renderer import ClothingRenderer
from src.vision.pose_estimator import (
    Landmark, PoseResult, LM_NOSE, LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER,
)

# Shoulder-line length used by every synthetic pose (px). The yaw property
# derives purely from the squeeze ratio, so one constant keeps the geometry
# easy to reason about.
_SHOULDER_SPAN = 160.0
_FRAME = (640, 480)


def _wrap_renderer(enabled=True, strength=0.85, max_arc_deg=85.0,
                   shading=True, shading_depth=0.35) -> ClothingRenderer:
    """Renderer whose wrap section matches the contract defaults."""
    cfg = RenderingSection()
    cfg.wrap.enabled = enabled
    cfg.wrap.strength = strength
    cfg.wrap.max_arc_deg = max_arc_deg
    cfg.wrap.shading = shading
    cfg.wrap.shading_depth = shading_depth
    return ClothingRenderer(cfg)


def _yaw_pose(yaw_deg: float, nose_dx: float = 40.0) -> PoseResult:
    """Pose whose ``torso_yaw_deg`` evaluates to ~``yaw_deg``.

    Shoulders are squeezed so ``frontal_ratio == |cos(yaw)|`` (which makes
    the yaw magnitude the requested angle) and the nose is offset to the
    right of the shoulder midpoint (left for negative yaw) to set the sign.
    """
    ratio = abs(math.cos(math.radians(yaw_deg)))
    dx = ratio * _SHOULDER_SPAN
    dy = math.sqrt(max(0.0, 1.0 - ratio * ratio)) * _SHOULDER_SPAN
    mid_x, mid_y = 280.0, 200.0
    sign = -1.0 if yaw_deg < 0 else 1.0
    landmarks = {
        LM_LEFT_SHOULDER: Landmark(mid_x - dx / 2.0, mid_y - dy / 2.0, 0.99),
        LM_RIGHT_SHOULDER: Landmark(mid_x + dx / 2.0, mid_y + dy / 2.0, 0.99),
        LM_NOSE: Landmark(mid_x + sign * abs(nose_dx), mid_y - 80.0, 0.99),
    }
    return PoseResult(landmarks=landmarks, frame_size=_FRAME)


def _placement(renderer: ClothingRenderer, pose: PoseResult):
    """Real placement for ``pose``; garment must be built at its size."""
    placement = renderer.compute_placement(pose, (200, 400))
    assert placement is not None, "compute_placement returned None"
    return placement


def _gradient_garment(size) -> np.ndarray:
    """Opaque BGRA garment whose B channel encodes the column index."""
    w, h = size
    g = np.zeros((h, w, 4), dtype=np.uint8)
    g[:, :, 0] = np.linspace(0.0, 255.0, w)[None, :]
    g[:, :, 3] = 255
    return g


def _src_positions(out: np.ndarray) -> np.ndarray:
    """Recover the per-column source position sampled by a ramp garment.

    The B channel is a linear 0..255 ramp over source columns, so the value
    of an output column maps back to the (fractional) source column that was
    interpolated there.
    """
    w = out.shape[1]
    return out[:, :, 0].mean(axis=0).astype(np.float64) / 255.0 * (w - 1)


# ---------------------------------------------------------------------------
# 1) PoseResult.torso_yaw_deg
# ---------------------------------------------------------------------------

def test_torso_yaw_frontal_is_zero():
    """Level shoulders + centered nose: frontal_ratio 1.0 -> yaw ~ 0."""
    pose = _yaw_pose(0.0, nose_dx=0.0)
    assert abs(pose.torso_yaw_deg) < 5.0, (
        f"frontal pose should have ~0 yaw, got {pose.torso_yaw_deg:.2f}"
    )


def test_torso_yaw_squeezed_nose_right_is_positive():
    """frontal_ratio ~0.5 with the nose right of the shoulder mid: ~60 deg."""
    pose = _yaw_pose(60.0, nose_dx=40.0)
    assert abs(pose.frontal_ratio - 0.5) < 0.02
    assert pose.torso_yaw_deg > 0.0, "nose right of shoulder mid must be positive"
    assert pose.torso_yaw_deg >= 50.0, (
        f"expected ~60 deg yaw, got {pose.torso_yaw_deg:.2f}"
    )


def test_torso_yaw_negative_when_nose_left():
    """Mirrored pose: the sign flips with the nose offset."""
    pose = _yaw_pose(-60.0, nose_dx=40.0)
    assert pose.torso_yaw_deg < -50.0, (
        f"expected ~-60 deg yaw, got {pose.torso_yaw_deg:.2f}"
    )


def test_torso_yaw_missing_nose_is_zero():
    """Without a nose landmark the yaw must be exactly 0."""
    pose = _yaw_pose(60.0)
    del pose.landmarks[LM_NOSE]
    assert pose.torso_yaw_deg == 0.0


# ---------------------------------------------------------------------------
# 2) _apply_torso_wrap geometry
# ---------------------------------------------------------------------------

def test_wrap_output_shape_preserved():
    """Wrapped garment keeps the input BGRA shape."""
    renderer = _wrap_renderer()
    pose = _yaw_pose(40.0)
    placement = _placement(renderer, pose)
    g = _gradient_garment(placement.size)
    out = renderer._apply_torso_wrap(g, pose, placement)
    assert out.shape == g.shape


def test_wrap_flat_center_is_identity():
    """At yaw=0 the center output column samples the texture center."""
    renderer = _wrap_renderer(shading=False)
    pose = _yaw_pose(0.0, nose_dx=0.0)
    placement = _placement(renderer, pose)
    g = _gradient_garment(placement.size)
    out = renderer._apply_torso_wrap(g, pose, placement)
    w = g.shape[1]
    src = _src_positions(out)[w // 2]
    assert abs(src - w // 2) <= 1.5, (
        f"center column sampled src x={src:.1f}, expected ~{w // 2}"
    )


def test_wrap_horizontal_mapping_monotone():
    """Yawed: each output column must sample a non-decreasing src column."""
    renderer = _wrap_renderer(shading=False)   # shading would bias the ramp
    pose = _yaw_pose(40.0)
    placement = _placement(renderer, pose)
    g = _gradient_garment(placement.size)
    out = renderer._apply_torso_wrap(g, pose, placement)
    src = _src_positions(out)
    diffs = np.diff(src)
    # Tolerance: one ramp-quantization step (~255/(w-1) src px of wobble).
    assert diffs.min() >= -1.6, (
        f"horizontal mapping is not monotone: min step {diffs.min():.2f}"
    )
    # Smoothed steps must be clearly positive (quantization-tolerant).
    smoothed = np.convolve(diffs, np.ones(5) / 5.0, mode="valid")
    assert smoothed.min() > 0.0


def test_wrap_edges_compress_faster_than_center():
    """Yawed: src displacement per dest column grows toward the edges."""
    renderer = _wrap_renderer(shading=False)   # shading would bias the ramp
    pose = _yaw_pose(40.0)
    placement = _placement(renderer, pose)
    g = _gradient_garment(placement.size)
    out = renderer._apply_torso_wrap(g, pose, placement)
    src = _src_positions(out)
    w = len(src)
    center_delta = src[int(0.55 * w)] - src[int(0.45 * w)]
    edge_delta = src[w - 1] - src[int(0.90 * w)]
    assert center_delta > 0.0
    assert edge_delta > center_delta, (
        f"edges must compress faster: edge delta {edge_delta:.1f} px vs "
        f"center delta {center_delta:.1f} px"
    )


def test_wrap_strength_zero_is_identity():
    """strength=0 must collapse the wrap to the flat (unmodified) garment."""
    renderer = _wrap_renderer(strength=0.0, shading=False)
    pose = _yaw_pose(40.0)
    placement = _placement(renderer, pose)
    g = _gradient_garment(placement.size)
    out = renderer._apply_torso_wrap(g, pose, placement)
    max_diff = int(np.max(np.abs(out.astype(np.int16) - g.astype(np.int16))))
    assert max_diff <= 2, f"strength=0 should be identity, max diff {max_diff}"


# ---------------------------------------------------------------------------
# 3) Far-side texture loss when yawed
# ---------------------------------------------------------------------------

def _two_tone_garment(size) -> np.ndarray:
    """Opaque garment: left half pure blue, right half pure red (BGR).

    For positive yaw the source range caps before theta_tex = +A, so the
    RIGHT (far) half of the texture is the red one.
    """
    w, h = size
    g = np.zeros((h, w, 4), dtype=np.uint8)
    g[:, :, :3] = (255, 0, 0)          # blue (near side for positive yaw)
    g[:, w // 2:, :3] = (0, 0, 255)    # red (far side)
    g[:, :, 3] = 255
    return g


def _color_share(out: np.ndarray, bgr) -> float:
    match = np.all(out[:, :, :3] == np.array(bgr, dtype=np.uint8), axis=2)
    return float(match.sum()) / match.size


def test_wrap_far_side_texture_lost_when_yawed():
    """torso_yaw_deg >= 55 with A=85: the far half passes the horizon.

    Positive yaw compresses the source range before theta_tex = +A, so the
    right (far) half of the texture must shrink drastically versus the
    yaw=0 render, while the near (left) half stays visible.
    """
    renderer = _wrap_renderer(shading=False)
    flat = _yaw_pose(0.0, nose_dx=0.0)
    yawed = _yaw_pose(60.0)
    assert yawed.torso_yaw_deg >= 55.0

    out_flat = renderer._apply_torso_wrap(
        _two_tone_garment(_placement(renderer, flat).size), flat,
        _placement(renderer, flat))
    out_yawed = renderer._apply_torso_wrap(
        _two_tone_garment(_placement(renderer, yawed).size), yawed,
        _placement(renderer, yawed))

    far_flat = _color_share(out_flat, (0, 0, 255))    # red: right half
    far_yawed = _color_share(out_yawed, (0, 0, 255))
    near_flat = _color_share(out_flat, (255, 0, 0))   # blue: left half
    near_yawed = _color_share(out_yawed, (255, 0, 0))

    assert far_flat > 0.4, f"sanity: far half missing at yaw=0 ({far_flat:.2f})"
    assert far_yawed < far_flat * 0.5, (
        f"far-side texture should vanish when yawed: share {far_yawed:.3f} "
        f"vs {far_flat:.3f} at yaw=0"
    )
    assert near_yawed > near_flat * 0.8, (
        f"near-side texture must survive: {near_yawed:.3f} vs {near_flat:.3f}"
    )


# ---------------------------------------------------------------------------
# 4) Volume shading
# ---------------------------------------------------------------------------

def _solid_garment(size, bgr) -> np.ndarray:
    w, h = size
    g = np.zeros((h, w, 4), dtype=np.uint8)
    g[:, :, :3] = bgr
    g[:, :, 3] = 137
    return g


def test_wrap_shading_darkens_edges_alpha_untouched():
    """depth=0.35: edge columns darker than the center; alpha untouched.

    Uses a yawed pose: the |psi| < 0.5 deg guard rail returns the input
    unchanged near yaw=0, so shading only manifests once yawed.
    """
    renderer = _wrap_renderer()          # shading on, depth 0.35
    pose = _yaw_pose(40.0)
    placement = _placement(renderer, pose)
    g = _solid_garment(placement.size, (100, 100, 100))
    out = renderer._apply_torso_wrap(g, pose, placement)

    w = out.shape[1]
    center = out[:, w // 2 - 2: w // 2 + 3, :3].mean()
    per_col = out[:, :, :3].mean(axis=(0, 2))
    darkest_edge = min(per_col[:3].min(), per_col[-3:].min())
    assert center - darkest_edge > 10.0, (
        f"edge columns must be darker: center {center:.1f} vs darkest edge "
        f"{darkest_edge:.1f}"
    )
    assert np.array_equal(out[:, :, 3], g[:, :, 3]), "alpha must be untouched"


def test_wrap_shading_disabled_leaves_colors():
    """With shading off a uniform garment stays uniform."""
    renderer = _wrap_renderer(shading=False)
    pose = _yaw_pose(0.0, nose_dx=0.0)
    placement = _placement(renderer, pose)
    g = _solid_garment(placement.size, (100, 100, 100))
    out = renderer._apply_torso_wrap(g, pose, placement)
    per_col = out[:, :, :3].mean(axis=(0, 2))
    assert per_col.max() - per_col.min() <= 2.0, (
        "shading off must not darken edges"
    )


def test_wrap_disabled_returns_input():
    """Guard rail: wrap.enabled=False returns the garment unchanged."""
    renderer = _wrap_renderer(enabled=False)
    pose = _yaw_pose(60.0)
    placement = _placement(renderer, pose)
    g = _two_tone_garment(placement.size)
    out = renderer._apply_torso_wrap(g, pose, placement)
    assert np.array_equal(out, g)


# ---------------------------------------------------------------------------
# 5) compute_placement: no frontal divide when wrap is enabled
# ---------------------------------------------------------------------------

def test_compute_placement_skips_frontal_divide_when_wrap_enabled():
    """With wrap enabled the box uses shoulder_w directly (~0.7 ratio)."""
    # frontal_ratio == 0.7 exactly -> old code widens by /0.7.
    renderer = _wrap_renderer()
    pose = _yaw_pose(math.degrees(math.acos(0.7)), nose_dx=0.0)
    assert abs(pose.frontal_ratio - 0.7) < 0.01
    placement = _placement(renderer, pose)

    assert placement.shoulder_width_px <= _SHOULDER_SPAN * 1.02, (
        f"shoulder width must not be divided by frontal_ratio "
        f"({placement.shoulder_width_px:.1f} vs {_SHOULDER_SPAN:.1f})"
    )
    expected_w = _SHOULDER_SPAN * renderer.config.shoulder_width_factor
    assert abs(placement.size[0] - expected_w) <= 4, (
        f"placement width {placement.size[0]} should be ~{expected_w:.0f} "
        f"(not the frontal-compensated ~{expected_w / 0.7:.0f})"
    )


def test_compute_placement_keeps_divide_when_wrap_disabled():
    """Contract keeps the legacy compensation when the wrap is off."""
    renderer = _wrap_renderer(enabled=False)
    pose = _yaw_pose(math.degrees(math.acos(0.7)), nose_dx=0.0)
    placement = _placement(renderer, pose)
    assert placement.shoulder_width_px > _SHOULDER_SPAN * 1.25, (
        f"wrap disabled should keep the frontal divide "
        f"({placement.shoulder_width_px:.1f})"
    )
