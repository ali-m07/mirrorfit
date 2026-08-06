"""Threaded webcam capture.

``cv2.VideoCapture.read()`` blocks for up to a full frame period. Running it
on a dedicated thread and publishing into a small queue lets the processing
loop consume frames at its own pace without ever stalling the camera driver,
which is the single biggest latency win in a real-time AR pipeline.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.config_loader import CameraSection
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BACKENDS = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,        # Windows DirectShow
    "msmf": cv2.CAP_MSMF,          # Windows Media Foundation
    "v4l2": cv2.CAP_V4L2,          # Linux
    "avfoundation": cv2.CAP_AVFOUNDATION,  # macOS
}


class WebcamStream:
    """Background-thread camera with a tiny latest-frame queue.

    Usage::

        with WebcamStream(cfg.camera) as cam:
            ok, frame = cam.read()
    """

    def __init__(self, config: CameraSection) -> None:
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max(1, config.buffer_size))
        self._mirror = config.mirror
        self._opened = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "WebcamStream":
        if self._thread and self._thread.is_alive():
            return self
        backend = _BACKENDS.get(str(self.config.backend).lower(), cv2.CAP_ANY)
        self._cap = cv2.VideoCapture(self.config.index, backend)
        if not self._cap.isOpened():
            # Fallback: retry with the default backend before giving up.
            logger.warning("Camera %d failed on backend '%s'; retrying with default",
                           self.config.index, self.config.backend)
            self._cap = cv2.VideoCapture(self.config.index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.config.index}. "
                "Check that no other application is using the webcam.")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise driver-side latency

        actual = self.resolution
        logger.info("Camera %d opened at %dx%d", self.config.index, *actual)

        self._stop.clear()
        self._thread = threading.Thread(target=self._grab_loop, name="webcam-grab", daemon=True)
        self._thread.start()
        if not self._opened.wait(timeout=5.0):
            logger.warning("Camera opened but no frame received within 5 s")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera released")

    def __enter__(self) -> "WebcamStream":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _grab_loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            if self._mirror:
                frame = cv2.flip(frame, 1)
            # Keep only the freshest frame: drop one if the queue is full.
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)
            self._opened.set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the latest frame; ``(False, None)`` on timeout."""
        try:
            return True, self._queue.get(timeout=timeout)
        except queue.Empty:
            return False, None

    @property
    def resolution(self) -> Tuple[int, int]:
        if not self._cap:
            return self.config.width, self.config.height
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.config.width)
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.config.height)
        return w, h

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
