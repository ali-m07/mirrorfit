"""Shared fixtures: synthetic camera and engine for headless tests."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config  # noqa: E402


class SyntheticCamera:
    """Drop-in WebcamStream replacement producing animated noise frames."""

    def __init__(self, width: int = 320, height: int = 240) -> None:
        self._w, self._h = width, height
        self._running = False
        self._tick = 0

    # WebcamStream-compatible surface ---------------------------------

    def start(self) -> "SyntheticCamera":
        self._running = True
        return self

    def stop(self) -> None:
        self._running = False

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._running:
            return False, None
        self._tick += 1
        rng = np.random.default_rng(self._tick)
        frame = rng.integers(0, 255, (self._h, self._w, 3), dtype=np.uint8)
        return True, frame

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._w, self._h

    @property
    def is_running(self) -> bool:
        return self._running


@pytest.fixture()
def config(tmp_path, monkeypatch):
    cfg = load_config()
    # Route all file output into a temp dir.
    cfg.output.screenshots_dir = str(tmp_path / "screenshots")
    cfg.output.recordings_dir = str(tmp_path / "recordings")
    cfg.logging.file = str(tmp_path / "logs" / "test.log")
    cfg.camera.width, cfg.camera.height = 320, 240
    return cfg


@pytest.fixture()
def engine(config):
    """Engine on synthetic frames with real render/UI but no vision graphs.

    ``enable_vision=False`` keeps the pipeline fully headless and fast: the
    loop captures, composites the HUD and serves frames without MediaPipe.
    """
    from src.core.tryon_engine import TryOnEngine

    eng = TryOnEngine(config, camera=SyntheticCamera().start(), enable_vision=False)
    eng.start()
    # Wait for at least one processed frame.
    deadline = time.time() + 5
    while time.time() < deadline and eng.latest_frame() is None:
        time.sleep(0.02)
    yield eng
    eng.stop()
