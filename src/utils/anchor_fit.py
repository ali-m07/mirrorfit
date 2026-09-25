"""Automatic anatomical anchor fitting from the garment silhouette itself.

Stage 2 of the ingestion pipeline. Instead of a human typing
``anchor_top`` / ``anchor_width`` numbers, the garment's own alpha mask is
analysed:

    alpha mask ──► top-width profile ──► shoulder line (widest upper row)
               ──► neckline (center of the top run)
               ──► anchor_top  = shoulder line as fraction of height
               ──► anchor_width = shoulder span as fraction of width
               ──► vertical span / width ratio (drives category sanity)

The renderer already maps ``anchor_top``/``anchor_width`` onto tracked body
landmarks, so a mask-derived fit "just works" for every garment shape —
boxy tees, raglan hoodies, A-line dresses — with zero manual numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AnchorFit:
    """Anchors derived from the garment mask, plus fit diagnostics."""

    anchor_top: float          # neckline y as a fraction of image height (0..0.4)
    anchor_width: float        # shoulder span as a fraction of image width (0.3..1.2)
    shoulder_row: int          # detected shoulder-line row (pixels)
    neckline_x: float          # neckline center x (pixels)
    confidence: float          # mask quality heuristic (0..1)


def fit_anchors_from_mask(bgra: np.ndarray) -> AnchorFit:
    """Analyse an RGBA cutout and return pose-anchoring metadata.

    Method (all derived from the mask, no constants tuned to one garment):
    1. Build the per-row opaque width profile.
    2. The shoulder line is the widest row inside the top 45% of the
       garment (sleeves stick out at shoulder height, so that row is the
       widest for any upper-body garment).
    3. The neckline is the horizontal center of the topmost opaque run on
       the row above the shoulders (the collar dip of the garment).
    """
    alpha = bgra[:, :, 3]
    h, w = alpha.shape[:2]
    binary = (alpha > 8).astype(np.uint8)
    total = float(binary.sum())
    if total < 100:
        return AnchorFit(anchor_top=0.0, anchor_width=1.0,
                         shoulder_row=h // 4, neckline_x=w / 2.0, confidence=0.0)

    widths = binary.sum(axis=1).astype(np.float32)

    # Shoulder line: widest row in the top 45% of the garment.
    search_end = max(1, int(h * 0.45))
    shoulder_row = int(np.argmax(widths[:search_end]))

    # Neckline: the topmost row that has content, its horizontal center.
    rows_with_content = np.nonzero(binary.any(axis=1))[0]
    top_row = int(rows_with_content[0])
    run = np.nonzero(binary[top_row])[0]
    neckline_x = float((run.min() + run.max()) / 2.0)

    # anchor_top: how far below the image top the shoulder line sits. After a
    # tight crop a normal garment has its shoulder line close to the top; the
    # renderer lifts the garment by this fraction of its height.
    anchor_top = float(np.clip(shoulder_row / float(h), 0.0, 0.4))

    # anchor_width: the shoulder span relative to the image width. This is
    # exactly what the renderer scales onto the tracked shoulder distance.
    anchor_width = float(np.clip(widths[shoulder_row] / float(w), 0.3, 1.2))

    # Confidence: a clean single-blob garment has its shoulder row in the top
    # ~35% and occupies a plausible share of the image.
    position_ok = shoulder_row <= h * 0.35
    coverage = total / float(h * w)
    size_ok = 0.15 <= coverage <= 0.85
    confidence = 1.0 if (position_ok and size_ok) else 0.5
    logger.debug("anchor fit: shoulder_row=%d/%d anchor_top=%.3f width=%.3f conf=%.2f",
                 shoulder_row, h, anchor_top, anchor_width, confidence)
    return AnchorFit(anchor_top=anchor_top, anchor_width=anchor_width,
                     shoulder_row=shoulder_row, neckline_x=neckline_x,
                     confidence=confidence)
