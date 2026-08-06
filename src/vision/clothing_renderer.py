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

    def __init__(self, config: RenderingSection) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def compute_placement(self, pose: PoseResult,
                          garment_shape: Tuple[int, int],
                          item: Optional[ClothingItem] = None) -> Optional[GarmentPlacement]:
        """Compute the affine placement for the garment, or None to skip."""
        gh, gw = garment_shape
        ls = pose.point(LM_LEFT_SHOULDER)
        rs = pose.point(LM_RIGHT_SHOULDER)
        if ls is None or rs is None:
            return None

        min_vis = 0.0  # visibility already filtered by the engine
        shoulder_w = pose.shoulder_width_px
        if shoulder_w < self.config.min_shoulder_width_px:
            return None  # person too far from camera to dress convincingly

        # Garment width from shoulder span (with per-asset fine-tuning).
        anchor_width = item.anchor_width if item else 1.0
        target_w = shoulder_w * self.config.shoulder_width_factor * anchor_width
        scale = target_w / max(gw, 1)
        target_h = gh * scale

        # Extend long garments (dresses) towards the hips when visible.
        category = (item.category if item else "upper").lower()
        hip_mid = pose.hip_mid
        if category in {"dress", "long"} and hip_mid is not None:
            shoulder_mid_y = (ls[1] + rs[1]) / 2.0
            torso_len = abs(hip_mid[1] - shoulder_mid_y)
            if torso_len > 10:
                target_h = max(target_h, torso_len * 1.35)

        # Neck anchor: slightly above the shoulder midpoint.
        mid_x = (ls[0] + rs[0]) / 2.0
        mid_y = (ls[1] + rs[1]) / 2.0
        neck_offset = shoulder_w * self.config.neck_offset_factor
        anchor_top = (item.anchor_top if item else 0.0) * target_h
        center_x = mid_x
        center_y = (mid_y - neck_offset - anchor_top) + target_h / 2.0

        # Rotation follows the shoulder line (clamped for stability).
        angle = pose.torso_angle_deg if self.config.perspective_tilt else 0.0
        angle = float(np.clip(angle, -self.config.max_rotation_deg,
                              self.config.max_rotation_deg))

        # Affine transform: garment space -> frame space.
        # Build by mapping the resized garment's center to the target center.
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
