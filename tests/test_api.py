"""API contract tests using FastAPI's TestClient with a synthetic-camera engine.

These run fully headless: no webcam, no MediaPipe.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
