"""MediaPipe Pose wrapper with temporal landmark smoothing.

Responsibilities:
- Run MediaPipe Pose on BGR frames.
- Convert normalised landmarks to pixel coordinates.
- Expose a small, named API for the landmarks the try-on pipeline needs.
- Smooth landmarks with an exponential moving average to suppress jitter.
- Hold the last-good pose briefly when detection flickers (graceful degradation).

This class is NOT thread-safe — MediaPipe graphs must be driven from a single
thread. The engine's worker thread owns the instance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.utils.config_loader import VisionSection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# MediaPipe BlazePose landmark indices (kept as constants so the rest of the
# codebase never depends on mediapipe's enum objects at runtime).
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_NOSE = 0
LM_LEFT_EAR = 7
LM_RIGHT_EAR = 8


@dataclass
class Landmark:
    x: float          # pixel coordinates
    y: float
    visibility: float


@dataclass
class PoseResult:
    """Smoothed pose for one frame, in pixel coordinates."""

    landmarks: Dict[int, Landmark]
    frame_size: Tuple[int, int]                     # (width, height)
    tracked: bool = True
    timestamp: float = field(default_factory=time.monotonic)

    def point(self, index: int) -> Optional[Tuple[float, float]]:
        lm = self.landmarks.get(index)
        return (lm.x, lm.y) if lm else None

    def visible(self, index: int, min_vis: float) -> bool:
        lm = self.landmarks.get(index)
        return lm is not None and lm.visibility >= min_vis

    # -- derived body measurements ----------------------------------------

    @property
    def shoulder_mid(self) -> Optional[Tuple[float, float]]:
        l, r = self.point(LM_LEFT_SHOULDER), self.point(LM_RIGHT_SHOULDER)
        if l is None or r is None:
            return None
        return (l[0] + r[0]) / 2.0, (l[1] + r[1]) / 2.0

    @property
    def hip_mid(self) -> Optional[Tuple[float, float]]:
        l, r = self.point(LM_LEFT_HIP), self.point(LM_RIGHT_HIP)
        if l is None or r is None:
            return None
        return (l[0] + r[0]) / 2.0, (l[1] + r[1]) / 2.0

    @property
    def shoulder_width_px(self) -> float:
        l, r = self.point(LM_LEFT_SHOULDER), self.point(LM_RIGHT_SHOULDER)
        if l is None or r is None:
            return 0.0
        return float(np.hypot(r[0] - l[0], r[1] - l[1]))

    @property
    def torso_angle_deg(self) -> float:
        """Rotation of the shoulder line relative to horizontal."""
        l, r = self.point(LM_LEFT_SHOULDER), self.point(LM_RIGHT_SHOULDER)
        if l is None or r is None:
            return 0.0
        return float(np.degrees(np.arctan2(r[1] - l[1], r[0] - l[0])))

    # -- richer body metrics for garment fitting ---------------------------

    def visibility(self, index: int) -> float:
        """Raw visibility of one landmark (0.0 when absent)."""
        lm = self.landmarks.get(index)
        return float(lm.visibility) if lm else 0.0

    @property
    def shoulder_visibility(self) -> float:
        """Worse of the two shoulder visibilities — the fitting bottleneck."""
        return min(self.visibility(LM_LEFT_SHOULDER),
                   self.visibility(LM_RIGHT_SHOULDER))

    @property
    def hip_visibility(self) -> float:
        return min(self.visibility(LM_LEFT_HIP),
                   self.visibility(LM_RIGHT_HIP))

    @property
    def torso_length_px(self) -> float:
        """Shoulder-mid to hip-mid distance; 0 when hips are not visible."""
        sm, hm = self.shoulder_mid, self.hip_mid
        if sm is None or hm is None:
            return 0.0
        return float(np.hypot(hm[0] - sm[0], hm[1] - sm[1]))

    @property
    def hip_width_px(self) -> float:
        l, r = self.point(LM_LEFT_HIP), self.point(LM_RIGHT_HIP)
        if l is None or r is None:
            return 0.0
        return float(np.hypot(r[0] - l[0], r[1] - l[1]))

    @property
    def frontal_ratio(self) -> float:
        """How frontal the torso is: 1.0 = facing camera, ->0 = sideways.

        Derived from how horizontal the shoulder line projects in the image;
        used to compensate the apparent shoulder-width shrink when turning.
        """
        l, r = self.point(LM_LEFT_SHOULDER), self.point(LM_RIGHT_SHOULDER)
        if l is None or r is None:
            return 1.0
        w = float(np.hypot(r[0] - l[0], r[1] - l[1]))
        if w < 1e-6:
            return 1.0
        return float(abs(r[0] - l[0]) / w)


class _EMAFilter:
    """Per-landmark exponential moving average with visibility-weighted gain."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._state: Dict[int, Tuple[float, float]] = {}

    def update(self, index: int, x: float, y: float, visibility: float) -> Tuple[float, float]:
        prev = self._state.get(index)
        if prev is None:
            self._state[index] = (x, y)
            return x, y
        # Low-confidence landmarks move less: jittery joints stay calmer.
        gain = self.alpha * (0.5 + 0.5 * visibility)
        nx = prev[0] + gain * (x - prev[0])
        ny = prev[1] + gain * (y - prev[1])
        self._state[index] = (nx, ny)
        return nx, ny

    def snapshot(self, index: int) -> Optional[Tuple[float, float]]:
        return self._state.get(index)

    def clear(self) -> None:
        self._state.clear()


class PoseEstimator:
    """Single-owner MediaPipe Pose driver."""

    def __init__(self, config: VisionSection) -> None:
        self.config = config
        self._pose = None
        self._mp = None
        self._filter = _EMAFilter(config.tracking.smoothing_alpha)
        self._last_good: Optional[PoseResult] = None
        self._missing_frames = 0

    def _init_graph(self) -> None:
        """Create the MediaPipe graph on first use (lazy).

        Lazy construction keeps the engine startable in environments without
        mediapipe installed (headless tests, CI) — the caller simply replaces
        or disables the estimator before the first frame.
        """
        if self._pose is not None:
            return
        import mediapipe as mp  # local import: keeps module importable without mediapipe

        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=int(self.config.pose.model_complexity),
            smooth_landmarks=bool(self.config.pose.smooth_landmarks),
            enable_segmentation=bool(self.config.pose.enable_segmentation),
            min_detection_confidence=float(self.config.pose.min_detection_confidence),
            min_tracking_confidence=float(self.config.pose.min_tracking_confidence),
        )
        logger.info("MediaPipe Pose initialised (complexity=%d)", self.config.pose.model_complexity)

    # ------------------------------------------------------------------

    def process(self, frame_bgr: np.ndarray) -> Optional[PoseResult]:
        """Estimate pose for one BGR frame. Returns None when tracking is lost."""
        if self._pose is None:
            self._init_graph()
        if self._pose is None:
            return None
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            self._missing_frames += 1
            # Graceful degradation: briefly reuse the last good pose so the
            # garment doesn't pop on/off during a one-frame detection drop.
            if (self._last_good is not None
                    and self._missing_frames <= self.config.tracking.max_missing_frames
                    and (time.monotonic() - self._last_good.timestamp)
                    <= self.config.tracking.lost_pose_timeout_s):
                stale = PoseResult(landmarks=self._last_good.landmarks,
                                   frame_size=self._last_good.frame_size,
                                   tracked=False)
                return stale
            self._filter.clear()
            return None

        self._missing_frames = 0
        landmarks: Dict[int, Landmark] = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            sx, sy = self._filter.update(idx, lm.x * w, lm.y * h, lm.visibility)
            landmarks[idx] = Landmark(x=sx, y=sy, visibility=float(lm.visibility))

        pose = PoseResult(landmarks=landmarks, frame_size=(w, h), tracked=True)
        self._last_good = pose
        return pose

    def draw_debug(self, frame_bgr: np.ndarray, pose: PoseResult) -> None:
        """Draw the pose skeleton for debugging (in place)."""
        if self._mp is None:
            return
        h, w = pose.frame_size[1], pose.frame_size[0]
        connections = self._mp.solutions.pose.POSE_CONNECTIONS
        for a, b in connections:
            pa, pb = pose.point(a), pose.point(b)
            if pa is None or pb is None:
                continue
            if not (0 <= pa[0] < w and 0 <= pa[1] < h and 0 <= pb[0] < w and 0 <= pb[1] < h):
                continue
            cv2.line(frame_bgr, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     (80, 220, 120), 2, cv2.LINE_AA)
        for idx in (LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER, LM_LEFT_HIP, LM_RIGHT_HIP):
            p = pose.point(idx)
            if p:
                cv2.circle(frame_bgr, (int(p[0]), int(p[1])), 5, (60, 130, 255), -1, cv2.LINE_AA)

    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
