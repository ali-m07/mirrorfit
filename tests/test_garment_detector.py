"""Tests for the YOLOS garment detector: output decoding + class filtering.

The ONNX session is mocked — no model download, no inference.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from src.vision.garment_detector import (
    _CLASS_TO_CATEGORY, _softmax, GarmentDetection, GarmentDetector,
)


def test_category_map_supported_set() -> None:
    det = GarmentDetection(box=(0, 0, 10, 10), score=0.9, class_id=1,
                           label="top", category="upper")
    assert det.supported
    pants = GarmentDetection(box=(0, 0, 10, 10), score=0.9, class_id=6,
                             label="pants", category="lower")
    assert not pants.supported
    # Every mapped class resolves to a known category string.
    assert set(_CLASS_TO_CATEGORY.values()) <= {"upper", "jacket", "dress",
                                                "long", "lower"}


def test_softmax_numerically_stable() -> None:
    logits = np.array([[1000.0, 1001.0, 999.0]])
    probs = _softmax(logits)
    assert np.isfinite(probs).all()
    assert abs(probs.sum() - 1.0) < 1e-5
    assert probs.argmax() == 1


class _FakeSession:
    """Mimics an ORT session: 100 queries, 47 classes; one dominant query."""

    def __init__(self, garment_query: int, cls: int, logit: float,
                 box: Tuple[float, float, float, float]) -> None:
        q, ncls = 100, 47
        self.logits = np.full((1, q, ncls), -10.0, dtype=np.float32)
        self.logits[0, garment_query, cls] = logit          # dominant class
        self.boxes = np.zeros((1, q, 4), dtype=np.float32)
        self.boxes[0, garment_query] = box                  # cxcywh normalised

    def get_inputs(self):
        class In:  # noqa: D401
            name = "pixel_values"
            shape = [1, 3, 512, 864]
        return [In()]

    def run(self, _names, _feed) -> List[np.ndarray]:
        return [self.logits, self.boxes]


def _detector_with(session: _FakeSession) -> GarmentDetector:
    det = GarmentDetector(auto_download=False)
    det._session = session
    det._input_hw = (512, 864)
    return det


def test_detect_picks_garment_and_ignores_parts() -> None:
    # Query 3 = "sleeve" (a garment PART, must be ignored) with a huge logit,
    # query 7 = a jacket (kept): the detector must return the jacket.
    session = _FakeSession(garment_query=3, cls=32, logit=10.99,
                           box=(0.5, 0.5, 0.9, 0.9))
    session.logits[0, 7, 4] = 8.0                      # class 4 = jacket
    session.boxes[0, 7] = (0.5, 0.4, 0.4, 0.5)

    det = _detector_with(session)
    result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result is not None
    assert result.class_id == 4 and result.category == "jacket"
    # Normalised box (0.5,0.4,0.4,0.5) on a 640x480 frame -> pixels.
    x0, y0, x1, y1 = result.box
    assert abs(x0 - (0.5 - 0.2) * 640) <= 2
    assert abs(y0 - (0.4 - 0.25) * 480) <= 2
    assert abs(x1 - (0.5 + 0.2) * 640) <= 2
    assert abs(y1 - (0.4 + 0.25) * 480) <= 2


def test_detect_returns_none_when_only_accessories() -> None:
    session = _FakeSession(garment_query=1, cls=13, logit=10.0,  # glasses
                           box=(0.5, 0.5, 0.3, 0.3))
    det = _detector_with(session)
    assert det.detect(np.zeros((480, 640, 3), dtype=np.uint8)) is None


def test_detect_respects_confidence_threshold() -> None:
    # logit -9 among -10s -> softmax ~0.055, below the 0.5 gate.
    session = _FakeSession(garment_query=5, cls=1, logit=-9.0,
                           box=(0.5, 0.5, 0.4, 0.4))
    det = _detector_with(session)
    assert det.detect(np.zeros((480, 640, 3), dtype=np.uint8)) is None
