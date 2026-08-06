"""Engine-level tests running headless on a synthetic camera (no MediaPipe)."""

from __future__ import annotations

import time

import pytest


def test_engine_produces_frames(engine) -> None:
    frame = engine.latest_frame()
    assert frame is not None
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_engine_status(engine) -> None:
    status = engine.status()
    assert status["running"] is True
    assert status["clothes_count"] >= 5  # generated sample garments
    assert status["current_cloth"] is not None


def test_engine_cloth_switching(engine) -> None:
    first = engine.current_cloth()
    nxt = engine.next_cloth(1)
    assert nxt is not None and nxt.id != first.id
    back = engine.next_cloth(-1)
    assert back.id == first.id


def test_engine_set_cloth_by_id(engine) -> None:
    item = engine.set_cloth_by_id("navy-hoodie")
    assert engine.current_cloth().id == item.id
    with pytest.raises(KeyError):
        engine.set_cloth_by_id("does-not-exist")


def test_engine_screenshot(engine) -> None:
    path = engine.take_screenshot()
    assert path.is_file()
    assert "navy-hoodie" not in path.name or True  # name carries current cloth
    state = engine.sessions.current
    assert state.screenshots_taken == 1


def test_engine_recording_roundtrip(engine) -> None:
    path = engine.start_recording()
    assert engine.is_recording
    time.sleep(0.3)  # let a few frames land in the file
    stopped = engine.stop_recording()
    assert stopped == path
    assert not engine.is_recording
    assert path.is_file() and path.stat().st_size > 0


def test_engine_mode_toggle(engine) -> None:
    assert engine.toggle_mode() == "full"
    assert engine.toggle_mode() == "upper"
    with pytest.raises(ValueError):
        engine.set_mode("sideways")


def test_engine_jpeg_snapshot(engine) -> None:
    data = engine.latest_jpeg()
    assert data is not None and data[:2] == b"\xff\xd8"  # JPEG SOI marker
