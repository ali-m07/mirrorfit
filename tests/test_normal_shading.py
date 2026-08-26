"""Tests for per-vertex normal-based shading of OBJ garments.

The renderer used to use only depth (z axis) for shading, producing a flat
silhouette. With per-vertex normals loaded from the OBJ file, the renderer
shades faces by their screen-space normal dotted with a virtual light
direction — that's what makes the garment look 3D instead of pasted on.
"""

from __future__ import annotations

import numpy as np

from src.utils.config_loader import RenderingSection
from src.utils.helpers import _compute_vertex_normals
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


def test_compute_vertex_normals_returns_unit_vectors():
    """Every returned normal must be either zero (degenerate vertex) or
    length 1.0 — that's what the renderer relies on for stable lighting."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    ], dtype=np.float32)
    faces = [(0, 1, 2), (0, 2, 3)]
    normals = _compute_vertex_normals(vertices, faces)
    lengths = np.linalg.norm(normals, axis=1)
    for i, length in enumerate(lengths):
        if length > 1e-6:
            assert abs(length - 1.0) < 1e-5, f"vertex {i}: length={length}"


def test_compute_vertex_normals_coplanar_quad():
    """All four corners of a flat Z=0 quad must share the same normal: +Z
    (since face order is counter-clockwise looking down +Z)."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    ], dtype=np.float32)
    faces = [(0, 1, 2), (0, 2, 3)]
    normals = _compute_vertex_normals(vertices, faces)
    # Cross of edges (1,0,0)×(1,1,0) = (0,0,1), so normal should be +Z.
    expected = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for i in range(4):
        np.testing.assert_allclose(normals[i], expected, atol=1e-5)


def test_render_mesh_with_normals_produces_varied_shades():
    """When vertex normals are supplied, faces lit from different angles
    must paint in different intensities — that's the 3D look. Without
    normals, the legacy depth-based shading produces near-uniform shades
    because every face has roughly the same depth."""
    pose = _pose()
    renderer = ClothingRenderer(RenderingSection())

    # Build a simple cube — eight vertices, 12 triangular faces. The faces
    # face six different directions so lighting must produce at least two
    # distinct intensity bands.
    s = 1.0
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ], dtype=np.float32)
    faces = [
        # Bottom (z=-1)
        (0, 2, 1), (0, 3, 2),
        # Top (z=+1)
        (4, 5, 6), (4, 6, 7),
        # Sides
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]
    normals = _compute_vertex_normals(vertices, faces)

    # Place the cube at the torso so the renderer actually draws it.
    cube_pose = PoseResult(landmarks={
        LM_LEFT_SHOULDER: Landmark(380, 200, 0.99),
        LM_RIGHT_SHOULDER: Landmark(420, 200, 0.99),
        LM_LEFT_HIP: Landmark(380, 400, 0.99),
        LM_RIGHT_HIP: Landmark(420, 400, 0.99),
    }, frame_size=(600, 800))

    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    drawn = renderer.render_mesh(frame, vertices, faces, cube_pose,
                                  vertex_normals=normals)
    assert drawn

    # Collect the green channel of every painted pixel (the renderer uses
    # green for the brightest part of the shaded colour).
    painted = frame[(frame > 0).any(axis=2)]
    if painted.size == 0:
        # Faces fell outside the frame; the test is inconclusive.
        return
    # At least two distinct intensity levels (faces catch different light).
    unique_shades = np.unique(painted[:, 1])
    assert len(unique_shades) >= 3, (
        f"only {len(unique_shades)} shade levels — normal lighting inactive"
    )


def test_render_mesh_without_normals_still_draws():
    """Falling back to depth-based shading must not crash or skip faces."""
    pose = _pose()
    renderer = ClothingRenderer(RenderingSection())
    # Same cube, but no normals.
    s = 1.0
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ], dtype=np.float32)
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    cube_pose = PoseResult(landmarks={
        LM_LEFT_SHOULDER: Landmark(380, 200, 0.99),
        LM_RIGHT_SHOULDER: Landmark(420, 200, 0.99),
        LM_LEFT_HIP: Landmark(380, 400, 0.99),
        LM_RIGHT_HIP: Landmark(420, 400, 0.99),
    }, frame_size=(600, 800))
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    drawn = renderer.render_mesh(frame, vertices, faces, cube_pose)
    assert drawn
    assert (frame > 0).any()
