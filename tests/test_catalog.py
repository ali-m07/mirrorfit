"""Tests for the clothing catalog."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils.helpers import ClothingCatalog


def _write_png(path: Path, channels: int = 4) -> None:
    if channels == 4:
        img = np.zeros((20, 20, 4), dtype=np.uint8)
        img[:, :, 3] = 255
    else:
        img = np.zeros((20, 20, channels), dtype=np.uint8)
    cv2.imwrite(str(path), img)


@pytest.fixture()
def clothes_dir(tmp_path: Path) -> Path:
    d = tmp_path / "clothes"
    d.mkdir()
    _write_png(d / "red_tee.png")
    _write_png(d / "blue_hoodie.png", channels=3)  # alpha added at load
    (d / "catalog.json").write_text(json.dumps({"items": [
        {"filename": "red_tee.png", "name": "Red Tee", "category": "upper",
         "anchor_width": 0.95},
    ]}), encoding="utf-8")
    return d


def test_catalog_discovers_pngs(clothes_dir: Path) -> None:
    catalog = ClothingCatalog(clothes_dir)
    assert len(catalog) == 2
    assert {i.filename for i in catalog.items} == {"red_tee.png", "blue_hoodie.png"}


def test_catalog_applies_metadata(clothes_dir: Path) -> None:
    catalog = ClothingCatalog(clothes_dir)
    item = catalog.find("red-tee")
    assert item is not None
    assert item.name == "Red Tee"
    assert item.anchor_width == pytest.approx(0.95)


def test_catalog_load_image_normalises_to_bgra(clothes_dir: Path) -> None:
    catalog = ClothingCatalog(clothes_dir)
    for item in catalog.items:
        img = catalog.load_image(item)
        assert img.shape[2] == 4


def test_catalog_empty_directory_raises_on_get(tmp_path: Path) -> None:
    catalog = ClothingCatalog(tmp_path / "empty")
    with pytest.raises(IndexError):
        catalog.get(0)


def test_catalog_cyclic_index(clothes_dir: Path) -> None:
    catalog = ClothingCatalog(clothes_dir)
    first = catalog.get(0)
    assert catalog.get(len(catalog)).id == first.id
