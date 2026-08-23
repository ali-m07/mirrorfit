"""API contract tests using FastAPI's TestClient with a synthetic-camera engine.

These run fully headless: no webcam, no MediaPipe.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import json

import cv2
import numpy as np

from src.api.routes import router
from src.core.tryon_engine import TryOnEngine
from tests.conftest import SyntheticCamera


@pytest.fixture()
def client(config):
    app = FastAPI()
    app.state.config = config
    app.state.engine = TryOnEngine(config, camera=SyntheticCamera().start(),
                                   enable_vision=False)
    app.include_router(router)
    with TestClient(app) as c:
        yield c
        if app.state.engine.is_running:
            app.state.engine.stop()
@pytest.fixture()
def catalog_client(config, tmp_path):
    """API client backed by an isolated temporary clothing catalog."""
    clothes_dir = tmp_path / "clothes"
    clothes_dir.mkdir()

    # Create a small valid RGBA PNG.
    image = np.zeros((64, 64, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    garment_path = clothes_dir / "test_hoodie.png"
    cv2.imwrite(str(garment_path), image)

    # Metadata entry for the test garment.
    (clothes_dir / "catalog.json").write_text(
        json.dumps({
            "items": [
                {
                    "id": "test-hoodie",
                    "filename": "test_hoodie.png",
                    "name": "Test Hoodie",
                    "category": "upper",
                    "description": "Temporary test garment",
                    "anchor_top": 0.1,
                    "anchor_width": 0.9,
                }
            ]
        }),
        encoding="utf-8",
    )

    config.clothes.directory = str(clothes_dir)

    app = FastAPI()
    app.state.config = config
    app.state.engine = TryOnEngine(
        config,
        camera=SyntheticCamera().start(),
        enable_vision=False,
    )
    app.include_router(router)

    with TestClient(app) as c:
        yield c

        if app.state.engine.is_running:
            app.state.engine.stop()

def test_health(client) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["engine_running"] is False  # not started yet


def test_clothes_listing(client) -> None:
    res = client.get("/clothes")
    assert res.status_code == 200
    body = res.json()
    assert body["count"] >= 5
    assert any(i["id"] == "navy-hoodie" for i in body["items"])


def test_full_session_flow(client) -> None:
    # Controls before a session are rejected with 409.
    assert client.post("/tryon/screenshot", json={}).status_code == 409

    res = client.post("/tryon/start", json={})
    assert res.status_code == 200
    session_id = res.json()["session_id"]

    # Change cloth by id.
    res = client.post("/tryon/change-cloth", json={"cloth_id": "navy-hoodie"})
    assert res.status_code == 200
    assert res.json()["data"]["cloth"]["id"] == "navy-hoodie"

    # Unknown garment -> 404.
    assert client.post("/tryon/change-cloth",
                       json={"cloth_id": "nope"}).status_code == 404

    # Mode + studio toggles.
    assert client.post("/tryon/mode", json={"mode": "full"}).status_code == 200
    assert client.post("/tryon/studio", json={"enabled": True}).status_code == 200

    status = client.get("/status").json()
    assert status["mode"] == "full"
    assert status["studio_enabled"] is True

    res = client.post("/tryon/stop", json={})
    assert res.status_code == 200
    assert res.json()["data"]["session"]["session_id"] == session_id


def test_change_cloth_cycle(client) -> None:
    client.post("/tryon/start", json={})
    first = client.get("/status").json()["current_cloth"]["id"]
    client.post("/tryon/change-cloth", json={"direction": "next"})
    second = client.get("/status").json()["current_cloth"]["id"]
    assert first != second

def test_update_cloth(catalog_client) -> None:
    res = catalog_client.patch(
        "/clothes/test-hoodie",
        json={"name": "Premium Test Hoodie"},
    )

    assert res.status_code == 200

    body = res.json()
    assert body["ok"] is True
    assert body["message"] == "Garment updated"

    cloth = body["data"]["cloth"]
    assert cloth["id"] == "test-hoodie"
    assert cloth["name"] == "Premium Test Hoodie"
    assert cloth["category"] == "upper"
    assert cloth["anchor_top"] == pytest.approx(0.1)
    assert cloth["anchor_width"] == pytest.approx(0.9)

def test_update_cloth_partial_preserves_metadata(catalog_client) -> None:
    res = catalog_client.patch(
        "/clothes/test-hoodie",
        json={"anchor_top": 0.25},
    )

    assert res.status_code == 200

    cloth = res.json()["data"]["cloth"]

    assert cloth["name"] == "Test Hoodie"
    assert cloth["category"] == "upper"
    assert cloth["description"] == "Temporary test garment"
    assert cloth["anchor_top"] == pytest.approx(0.25)
    assert cloth["anchor_width"] == pytest.approx(0.9)

def test_update_unknown_cloth(catalog_client) -> None:
    res = catalog_client.patch(
        "/clothes/no-such-garment",
        json={"name": "Whatever"},
    )

    assert res.status_code == 404

def test_delete_cloth(catalog_client) -> None:
    res = catalog_client.delete("/clothes/test-hoodie")

    assert res.status_code == 200

    body = res.json()
    assert body["ok"] is True
    assert body["message"] == "Garment deleted"
    assert body["data"]["id"] == "test-hoodie"
    assert body["data"]["filename"] == "test_hoodie.png"

    # Catalog was reloaded and the garment is gone.
    clothes = catalog_client.get("/clothes").json()
    assert not any(
        item["id"] == "test-hoodie"
        for item in clothes["items"]
    )

def test_delete_cloth_removes_asset(catalog_client) -> None:
    engine = catalog_client.app.state.engine
    garment_path = engine.catalog.directory / "test_hoodie.png"

    assert garment_path.exists()

    res = catalog_client.delete("/clothes/test-hoodie")

    assert res.status_code == 200
    assert not garment_path.exists()

def test_delete_unknown_cloth(catalog_client) -> None:
    res = catalog_client.delete("/clothes/no-such-garment")

    assert res.status_code == 404

def test_update_cloth_rejects_invalid_anchor(catalog_client) -> None:
    res = catalog_client.patch(
        "/clothes/test-hoodie",
        json={"anchor_top": 2.0},
    )

    assert res.status_code == 422

def test_update_cloth_rejects_invalid_category(catalog_client) -> None:
    res = catalog_client.patch(
        "/clothes/test-hoodie",
        json={"category": "pants"},
    )

    assert res.status_code == 422