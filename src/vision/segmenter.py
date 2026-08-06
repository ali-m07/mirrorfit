"""Person segmentation (MediaPipe Selfie Segmentation) + Virtual Studio.

Produces a float mask in [0, 1] for the person in the frame and composites
virtual backgrounds (gradient / image / blur) behind them — the "Virtual
Studio" mode used in retail kiosks and branded experiences.

Like :class:`PoseEstimator`, this wraps a MediaPipe graph and must be driven
from a single thread.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.utils.config_loader import StudioSection, SegmentationSection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PersonSegmenter:
    """Single-owner MediaPipe Selfie Segmentation driver."""

    def __init__(self, config: SegmentationSection) -> None:
        self.config = config
        self._segmenter = None

    def _init_graph(self) -> None:
        """Lazy construction — see PoseEstimator for the rationale."""
        if self._segmenter is not None:
            return
        import mediapipe as mp  # local import keeps module importable without mediapipe

        self._segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=int(self.config.model_selection))
        logger.info("MediaPipe Selfie Segmentation initialised (model=%d)",
                    self.config.model_selection)

    def process(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return a float32 mask in [0,1], same HxW as the frame (1 = person)."""
        if self._segmenter is None:
            self._init_graph()
        if self._segmenter is None:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._segmenter.process(rgb)
        if result.segmentation_mask is None:
            return None
        return result.segmentation_mask.astype(np.float32)

    def close(self) -> None:
        if self._segmenter is not None:
            self._segmenter.close()
            self._segmenter = None

    def __enter__(self) -> "PersonSegmenter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class StudioCompositor:
    """Background replacement used by the Virtual Studio mode."""

    def __init__(self, config: StudioSection) -> None:
        self.config = config
        self._bg_image: Optional[np.ndarray] = None
        if config.mode == "image":
            path = Path(config.background_image)
            if path.is_file():
                self._bg_image = cv2.imread(str(path))
            else:
                logger.warning("Studio background image not found: %s — falling back to gradient",
                               path)

    def set_mode(self, mode: str) -> None:
        if mode in {"gradient", "image", "blur"}:
            self.config.mode = mode

    def _gradient_background(self, h: int, w: int) -> np.ndarray:
        top = np.array(self.config.gradient_top, dtype=np.float32)
        bottom = np.array(self.config.gradient_bottom, dtype=np.float32)
        t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        gradient = top[None, None, :] * (1 - t) + bottom[None, None, :] * t
        return np.repeat(gradient.astype(np.uint8), w, axis=1)

    def background_for(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.config.mode == "image" and self._bg_image is not None:
            return cv2.resize(self._bg_image, (w, h), interpolation=cv2.INTER_AREA)
        if self.config.mode == "blur":
            k = max(3, int(self.config.blur_strength) | 1)  # force odd kernel
            return cv2.GaussianBlur(frame, (k, k), 0)
        return self._gradient_background(h, w)

    def composite(self, frame_bgr: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
        """Replace the background with the studio backdrop, keeping the person."""
        if person_mask is None:
            return frame_bgr
        background = self.background_for(frame_bgr)
        # Feather the mask edges so hair and shoulders don't cut out harshly.
        soft = cv2.GaussianBlur(person_mask, (5, 5), 0)
        soft = np.clip((soft - (self.config.segmentation_threshold - 0.25)) / 0.5, 0.0, 1.0)
        alpha = soft[:, :, None]
        out = frame_bgr.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)
        return out.astype(np.uint8)
