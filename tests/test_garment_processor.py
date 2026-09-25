"""Headless tests for the garment auto-processor and catalog merge helper.

Detection is disabled (``detect=False``) in these tests: the YOLOS model is
validated separately in test_garment_detector.py with a mocked session.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.utils.anchor_fit import fit_anchors_from_mask
from src.utils.garment_processor import (
    _has_usable_alpha, process_garment, update_catalog_json,
)


def _synthetic_photo(width: int = 240, height: int = 320) -> np.ndarray:
    """A 'product photo': colored garment-ish blob on a plain background."""
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.ellipse(image, (width // 2, height // 2), (width // 3, height // 3),
                0, 0, 360, (40, 60, 200), -1)
    cv2.rectangle(image, (width // 2 - width // 3, int(height * 0.25)),
                  (width // 2 + width // 3, int(height * 0.55)), (40, 60, 200), -1)
    return image


def test_grabcut_produces_opaque_cutout() -> None:
    cutout = process_garment(_synthetic_photo(), method="auto", detect=False)
    assert cutout.bgra.ndim == 3 and cutout.bgra.shape[2] == 4
    coverage = float((cutout.bgra[:, :, 3] > 8).mean())
    assert 0.05 < coverage < 1.0          # something found, background gone
    assert cutout.method == "grabcut"     # rembg not installed in CI/venv


def test_crop_is_tight_to_content() -> None:
    cutout = process_garment(_synthetic_photo(), detect=False)
    h, w = cutout.bgra.shape[:2]
    # Background rows/cols must be gone: content fills the crop.
    ys, xs = np.nonzero(cutout.bgra[:, :, 3] > 8)
    assert xs.min() <= 8 and ys.min() <= 8
    assert w - 1 - xs.max() <= 8 and h - 1 - ys.max() <= 8
    assert 0.55 <= cutout.anchor_width <= 1.0


def test_anchors_fit_from_silhouette() -> None:
    """A tee-like mask: shoulders widest near the top, neckline centered."""
    mask = np.zeros((300, 200), dtype=np.uint8)
    cv2.rectangle(mask, (60, 40), (140, 280), 255, -1)    # torso
    cv2.rectangle(mask, (20, 40), (180, 90), 255, -1)     # sleeve band (widest)
    bgra = np.zeros((300, 200, 4), dtype=np.uint8)
    bgra[:, :, 3] = mask
    fit = fit_anchors_from_mask(bgra)
    # Shoulder line detected at the sleeve band, ~1/7 down the garment.
    assert 35 <= fit.shoulder_row <= 50
    assert 0.1 <= fit.anchor_top <= 0.2
    # Shoulder span = 160 px over a 200 px image.
    assert 0.75 <= fit.anchor_width <= 0.85
    assert 90 <= fit.neckline_x <= 110


def test_dress_silhouette_yields_higher_anchor_top_than_tee() -> None:
    """Long garments carry their shoulder line relatively higher (width peak
    sits closer to the top because the lower half is a constant-width skirt).
    Both must stay inside the renderer's 0..0.4 anchor_top range."""
    def make(hem_width: int) -> np.ndarray:
        mask = np.zeros((400, 200), dtype=np.uint8)
        cv2.rectangle(mask, (60, 40), (140, 200), 255, -1)
        cv2.rectangle(mask, (20, 40), (180, 90), 255, -1)
        cv2.rectangle(mask, (hem_width[0], 200), (hem_width[1], 390), 255, -1)
        bgra = np.zeros((400, 200, 4), dtype=np.uint8)
        bgra[:, :, 3] = mask
        return bgra

    fit_tee = fit_anchors_from_mask(make((70, 130)))
    fit_dress = fit_anchors_from_mask(make((10, 190)))
    assert 0.0 <= fit_dress.anchor_top <= 0.4
    assert fit_dress.anchor_width >= fit_tee.anchor_width  # wide hem -> wider span


def test_transparent_png_passes_through() -> None:
    rgba = np.zeros((100, 80, 4), dtype=np.uint8)
    cv2.rectangle(rgba, (10, 20), (70, 90), (10, 20, 200, 255), -1)
    cutout = process_garment(rgba, detect=False)
    assert cutout.method == "passthrough"
    assert _has_usable_alpha(rgba)
    # Content preserved, cropped to the blob with a small margin.
    assert cutout.bgra.shape[0] < 100


def test_empty_mask_raises() -> None:
    blank = np.full((64, 64, 3), 200, dtype=np.uint8)
    with pytest.raises(ValueError):
        process_garment(blank, detect=False)


def test_update_catalog_json_merges(tmp_path) -> None:
    catalog = tmp_path / "catalog.json"
    update_catalog_json(catalog, "a.png", {"id": "a", "name": "A"})
    update_catalog_json(catalog, "b.png", {"id": "b", "name": "B"})
    update_catalog_json(catalog, "a.png", {"id": "a2", "name": "A2"})
    entries = json.loads(catalog.read_text(encoding="utf-8"))["items"]
    assert [e["filename"] for e in entries] == ["b.png", "a.png"]
    assert entries[1]["name"] == "A2"
