"""Garment rendering: pose-aligned warp, alpha compositing, shadow & occlusion.

Pipeline per frame::

    garment (BGRA)
      -> scale to shoulder width / torso length
      -> rotate to match the shoulder-line angle (perspective tilt)
      -> optional scene-brightness matching (lighting)
      -> optional drop shadow (same affine transform)
      -> optional forearm occlusion cutout
      -> premultiplied alpha blend onto the frame

All measurements derive from the *smoothed* pose, so the garment inherits the
jitter suppression applied upstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.config_loader import RenderingSection
from src.utils.helpers import ClothingItem
from src.utils.logger import get_logger
from src.vision.pose_estimator import (
    PoseResult, LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER, LM_LEFT_ELBOW,
    LM_RIGHT_ELBOW, LM_LEFT_WRIST, LM_RIGHT_WRIST, LM_LEFT_HIP, LM_RIGHT_HIP,
)

logger = get_logger(__name__)


@dataclass
class GarmentPlacement:
    """Where and how the garment will be drawn this frame."""

    center: Tuple[float, float]
    size: Tuple[int, int]          # (width, height) before rotation
    angle_deg: float
    transform: np.ndarray          # 2x3 affine matrix garment->frame
    shoulder_width_px: float


class ClothingRenderer:
    """Renders transparent PNG garments onto video frames, pose-aligned."""

    # Blending weight of the torso length into the garment scale. The scale is
    # derived from BOTH shoulder width and shoulder->hip length so the garment
    # keeps a believable aspect instead of only stretching horizontally.
    _TORSO_SCALE_BLEND = 0.35

    # Fraction of torso length used to lift the garment top above the shoulder
    # line. Tunable per category in _category_neck_factor().
    _NECK_TORSO_FACTOR = 0.13

    # Angle EMA gain per frame (lower = smoother rotation).
    _ANGLE_SMOOTHING = 0.35

    def __init__(self, config: RenderingSection, min_visibility: float = 0.55) -> None:
        self.config = config
        self.min_visibility = min_visibility
        self._angle_state: Optional[float] = None  # EMA of the garment angle

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    @staticmethod
    def _category_neck_factor(category: str) -> float:
        """Per-category tweak of how far the collar sits above the shoulders."""
        return {
            "dress": 0.16,   # dresses anchor slightly higher (neckline)
            "long": 0.15,
            "upper": 0.13,   # tees / hoodies / jackets
        }.get(category, 0.13)

    @staticmethod
    def _category_length_factor(category: str) -> float:
        """Vertical size multiplier per category when hips are available."""
        return {
            "dress": 1.55,   # cover torso + extend well below hips
            "long": 1.35,
            "upper": 1.15,   # hem just below the belt line
        }.get(category, 1.15)

    def _smooth_angle(self, angle: float) -> float:
        """EMA on the rotation angle — kills frame-to-frame wobble."""
        if self._angle_state is None:
            self._angle_state = angle
            return angle
        self._angle_state += self._ANGLE_SMOOTHING * (angle - self._angle_state)
        return self._angle_state

    def compute_placement(self, pose: PoseResult,
                          garment_shape: Tuple[int, int],
                          item: Optional[ClothingItem] = None) -> Optional[GarmentPlacement]:
        """Compute the affine placement for the garment, or None to skip."""
        gh, gw = garment_shape
        if gw <= 0 or gh <= 0:
            return None
        ls = pose.point(LM_LEFT_SHOULDER)
        rs = pose.point(LM_RIGHT_SHOULDER)
        if ls is None or rs is None:
            return None
        if not pose.tracked or pose.shoulder_visibility < self.min_visibility:
            return None

        frame_w, frame_h = pose.frame_size
        shoulder_w = pose.shoulder_width_px
        if (shoulder_w < self.config.min_shoulder_width_px
                or not np.isfinite(shoulder_w)):
            return None

        # Width from the shoulder span, clamped hard against the frame size
        # so a bad frame can never blow the garment up into a blob.
        anchor_width = item.anchor_width if item else 1.0
        target_w = shoulder_w * self.config.shoulder_width_factor * anchor_width
        target_w = float(min(target_w, frame_w * 0.85))

        # Preserve the asset aspect ratio, but use the observed torso length
        # when hips are reliable. This makes dresses/long garments extend
        # naturally instead of behaving like every other upper-body item.
        aspect_h = gh * (target_w / gw)
        category = (item.category if item else "upper").lower()
        torso_h = pose.torso_length_px * self._category_length_factor(category)
        if pose.hip_visibility >= self.min_visibility and torso_h > 0:
            target_h = max(aspect_h, aspect_h * (1 - self._TORSO_SCALE_BLEND)
                           + torso_h * self._TORSO_SCALE_BLEND)
            if category in {"dress", "long"}:
                target_h = max(target_h, torso_h)
        else:
            target_h = aspect_h
        # Long garments are allowed to extend below the visible frame; the
        # compositor clips them safely. A shared 95% cap made dress and upper
        # garments identical for long torsos.
        height_cap = frame_h * (1.20 if category in {"dress", "long"} else 0.95)
        target_h = float(min(target_h, height_cap))
        if target_w < 8 or target_h < 8:
            return None

        # Anchor the garment TOP just above the shoulder midpoint, then derive
        # the centre from it — this keeps the collar at the neck and stops the
        # vertical drift that used to push the garment to the bottom.
        mid_x = (ls[0] + rs[0]) / 2.0
        mid_y = (ls[1] + rs[1]) / 2.0
        neck_lift = max(shoulder_w * self.config.neck_offset_factor,
                        pose.torso_length_px * self._category_neck_factor(category)
                        if pose.torso_length_px > 0 else 0.0)
        anchor_top = (item.anchor_top if item else 0.0) * target_h
        top_y = mid_y - neck_lift - anchor_top
        center_x = mid_x
        center_y = top_y + target_h / 2.0
        if not (np.isfinite(center_x) and np.isfinite(center_y)):
            return None

        # Rotation follows the shoulder line (clamped for stability).
        angle = pose.torso_angle_deg if self.config.perspective_tilt else 0.0
        if not np.isfinite(angle):
            angle = 0.0
        angle = float(np.clip(angle, -self.config.max_rotation_deg,
                              self.config.max_rotation_deg))
        angle = self._smooth_angle(angle)

        # Affine transform: garment space -> frame space.
        transform = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), angle, 1.0)
        transform[0, 2] += center_x - target_w / 2.0
        transform[1, 2] += center_y - target_h / 2.0

        return GarmentPlacement(
            center=(center_x, center_y),
            size=(int(round(target_w)), int(round(target_h))),
            angle_deg=angle,
            transform=transform,
            shoulder_width_px=shoulder_w,
        )

    # ------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------

    def render_mesh(self, frame: np.ndarray, vertices: np.ndarray, faces,
                    pose: Optional[PoseResult], item: Optional[ClothingItem] = None) -> bool:
        """Project a low-poly garment mesh onto the tracked torso.

        This is a lightweight CPU renderer for OBJ assets: it preserves mesh
        depth ordering and produces real 3D silhouette geometry without adding
        a heavyweight OpenGL dependency to the desktop app.
        """
        if pose is None or not pose.tracked or pose.shoulder_visibility < self.min_visibility:
            return False
        sm, hm = pose.shoulder_mid, pose.hip_mid
        if sm is None or hm is None or not len(vertices):
            return False
        ls, rs = pose.point(LM_LEFT_SHOULDER), pose.point(LM_RIGHT_SHOULDER)
        if ls is None or rs is None:
            return False
        v = np.asarray(vertices, dtype=np.float32)
        xmin, xmax = float(v[:, 0].min()), float(v[:, 0].max())
        ymin, ymax = float(v[:, 1].min()), float(v[:, 1].max())
        zmin, zmax = float(v[:, 2].min()), float(v[:, 2].max())
        span_x, span_y = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
        torso = max(float(np.hypot(hm[0]-sm[0], hm[1]-sm[1])), 1.0)
        shoulder = max(float(np.hypot(rs[0]-ls[0], rs[1]-ls[1])), 1.0)
        # Build an orthonormal body basis. The mesh's highest point is
        # anchored at the shoulder line and its lowest point is constrained to
        # the hip/torso region; this prevents the hoodie from sitting on the
        # user's head when an OBJ has a tall hood.
        hx, hy = (rs[0]-ls[0]) / shoulder, (rs[1]-ls[1]) / shoulder
        vx, vy = -hy, hx
        if vx * (hm[0]-sm[0]) + vy * (hm[1]-sm[1]) < 0:
            vx, vy = -vx, -vy
        # The supplied block-out hoodie is intentionally fitted to the upper
        # torso. Keep its silhouette inside the shoulders instead of letting
        # the sleeves balloon beyond the tracked body.
        target_width = shoulder * 1.16
        target_height = min(torso * 0.98, pose.frame_size[1] * 0.62)
        # Neckline sits a little above the shoulder midpoint.
        anchor_x = sm[0] - vx * torso * 0.07
        anchor_y = sm[1] - vy * torso * 0.07
        scale_x, scale_y = target_width / span_x, target_height / span_y
        projected = []
        for x, y, z in v:
            px = (x - (xmin+xmax)/2) * scale_x
            py = (ymax - y) * scale_y
            # Small depth contribution gives a convincing volume cue without
            # changing the garment's body anchor.
            depth = (z - (zmin+zmax)/2) * scale_x * 0.06
            projected.append((int(anchor_x + hx * px + vx * py + hx * depth),
                              int(anchor_y + hy * px + vy * py + hy * depth), float(z)))
        for a, b, c in sorted(faces, key=lambda f: sum(projected[i][2] for i in f)):
            pts = np.array([[projected[i][0], projected[i][1]] for i in (a,b,c)], dtype=np.int32)
            depth = sum(projected[i][2] for i in (a,b,c)) / 3
            shade = int(np.clip(135 + 55 * (depth-zmin) / max(zmax-zmin, 1e-6), 80, 210))
            cv2.fillConvexPoly(frame, pts, (shade//3, shade//2, shade), lineType=cv2.LINE_AA)
        return True

    def _match_scene_brightness(self, garment: np.ndarray, frame: np.ndarray,
                                placement: GarmentPlacement) -> np.ndarray:
        """Tone the garment towards the scene's local exposure."""
        if not self.config.lighting.enabled or not self.config.lighting.match_scene_brightness:
            return garment
        h, w = frame.shape[:2]
        x, y = placement.center
        # Sample the frame region where the garment will land.
        half_w = int(placement.size[0] * 0.35)
        half_h = int(placement.size[1] * 0.30)
        x1, x2 = int(max(0, x - half_w)), int(min(w, x + half_w))
        y1, y2 = int(max(0, y - half_h)), int(min(h, y + half_h))
        if x2 <= x1 or y2 <= y1:
            return garment
        scene_luma = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).mean()

        opaque = garment[:, :, 3] > 10
        if not opaque.any():
            return garment
        garment_luma = cv2.cvtColor(garment[:, :, :3], cv2.COLOR_BGR2GRAY)[opaque].mean()
        if garment_luma < 1e-3:
            return garment
        gain = float(np.clip(scene_luma / garment_luma, 0.75, 1.3))
        out = garment.copy()
        out[:, :, :3] = np.clip(garment[:, :, :3].astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return out

    def _build_shadow(self, garment: np.ndarray) -> Optional[np.ndarray]:
        """Soft drop shadow derived from the garment alpha channel."""
        if not self.config.shadow.enabled:
            return None
        alpha = garment[:, :, 3]
        k = max(3, int(self.config.shadow.blur_radius) | 1)
        blurred = cv2.GaussianBlur(alpha, (k, k), 0)
        shadow = np.zeros_like(garment)
        shadow[:, :, 3] = (blurred.astype(np.float32) * self.config.shadow.opacity).astype(np.uint8)
        return shadow

    def _apply_arm_occlusion(self, garment: np.ndarray, pose: PoseResult,
                             placement: GarmentPlacement) -> np.ndarray:
        """Cut the garment alpha where the person's forearms cross the torso.

        The arms should visually pass *in front of* the garment. We rebuild the
        arm silhouette in garment space using the inverse affine transform.
        """
        if not self.config.occlusion.use_arm_cutout:
            return garment
        gw, gh = placement.size[0], placement.size[1]
        if gw <= 0 or gh <= 0:
            return garment

        inv = cv2.invertAffineTransform(placement.transform)
        mask = np.zeros((gh, gw), dtype=np.uint8)

        def to_garment(p: Tuple[float, float]) -> Tuple[int, int]:
            gx = inv[0, 0] * p[0] + inv[0, 1] * p[1] + inv[0, 2]
            gy = inv[1, 0] * p[0] + inv[1, 1] * p[1] + inv[1, 2]
            return int(gx), int(gy)

        for shoulder, elbow, wrist in (
            (LM_LEFT_SHOULDER, LM_LEFT_ELBOW, LM_LEFT_WRIST),
            (LM_RIGHT_SHOULDER, LM_RIGHT_ELBOW, LM_RIGHT_WRIST),
        ):
            pts = [pose.point(i) for i in (shoulder, elbow, wrist)]
            if any(p is None for p in pts):
                continue
            # Only occlude when the wrist is raised to torso height (arms crossed
            # or hand near chest) — otherwise arms hang beside the garment.
            wrist_y = pts[2][1]  # type: ignore[index]
            shoulder_y = pts[0][1]  # type: ignore[index]
            if wrist_y > shoulder_y + placement.shoulder_width_px * 0.9:
                continue
            poly = np.array([to_garment(p) for p in pts if p is not None], dtype=np.int32)
            thickness = max(6, int(placement.shoulder_width_px * 0.16))
            cv2.polylines(mask, [poly], isClosed=False, color=255,
                          thickness=thickness, lineType=cv2.LINE_AA)

        dil = max(1, int(self.config.occlusion.arm_mask_dilation))
        if dil > 1:
            kernel = np.ones((dil, dil), np.uint8)
            mask = cv2.dilate(mask, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        out = garment.copy()
        out[:, :, 3] = cv2.subtract(out[:, :, 3], mask)
        return out

    # ------------------------------------------------------------------
    # Compositing
    # ------------------------------------------------------------------

    @staticmethod
    def _blend_warped(frame: np.ndarray, warped_bgra: np.ndarray) -> None:
        """Premultiplied-style alpha blend of a full-frame BGRA layer."""
        alpha = (warped_bgra[:, :, 3:4].astype(np.float32)) / 255.0
        rgb = warped_bgra[:, :, :3].astype(np.float32)
        frame[:] = (rgb * alpha + frame.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    def render(self, frame: np.ndarray, garment_bgra: np.ndarray,
               pose: Optional[PoseResult],
               item: Optional[ClothingItem] = None) -> bool:
        """Draw the garment onto ``frame`` (in place). Returns True if drawn."""
        if pose is None or garment_bgra is None or garment_bgra.size == 0:
            return False

        placement = self.compute_placement(pose, garment_bgra.shape[:2], item)
        if placement is None:
            return False

        gw, gh = placement.size
        if gw < 8 or gh < 8:
            return False

        resized = cv2.resize(garment_bgra, (gw, gh), interpolation=cv2.INTER_AREA)
        resized = self._match_scene_brightness(resized, frame, placement)
        resized = self._apply_arm_occlusion(resized, pose, placement)

        h, w = frame.shape[:2]

        # --- translate-only fast path (angle ~ 0): direct ROI blending ------
        # warpAffine on the FULL frame every frame is slow and smears edges;
        # when there is no rotation we can paste the resized garment directly.
        if abs(placement.angle_deg) < 0.5:
            cx, cy = placement.center
            x0 = int(round(cx - gw / 2.0))
            y0 = int(round(cy - gh / 2.0))
            x1, y1 = x0 + gw, y0 + gh
            # Clip to frame bounds; adjust the garment crop correspondingly.
            fx0, fy0 = max(0, x0), max(0, y0)
            fx1, fy1 = min(w, x1), min(h, y1)
            if fx1 <= fx0 or fy1 <= fy0:
                return False
            gx0, gy0 = fx0 - x0, fy0 - y0
            crop = resized[gy0:gy0 + (fy1 - fy0), gx0:gx0 + (fx1 - fx0)]
            shadow = self._build_shadow(resized)
            if shadow is not None:
                shadow_crop = shadow[gy0:gy0 + (fy1 - fy0), gx0:gx0 + (fx1 - fx0)]
                shadow_roi = frame[fy0:fy1, fx0:fx1]
                self._blend_warped(shadow_roi, shadow_crop)
            roi = frame[fy0:fy1, fx0:fx1]
            alpha = (crop[:, :, 3:4].astype(np.float32)) / 255.0
            rgb = crop[:, :, :3].astype(np.float32)
            roi[:] = (rgb * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            return True

        shadow = self._build_shadow(resized)
        if shadow is not None:
            shadow_tf = placement.transform.copy()
            shadow_tf[0, 2] += self.config.shadow.offset_x
            shadow_tf[1, 2] += self.config.shadow.offset_y
            warped_shadow = cv2.warpAffine(shadow, shadow_tf, (w, h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT)
            self._blend_warped(frame, warped_shadow)

        warped = cv2.warpAffine(resized, placement.transform, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        self._blend_warped(frame, warped)
        return True
