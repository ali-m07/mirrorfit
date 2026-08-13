"""The try-on engine: capture -> pose -> segment -> render -> publish.

Runs the full pipeline on a dedicated worker thread (MediaPipe graphs are
single-owner), while exposing a small thread-safe control surface used by both
the desktop app (``main.py``) and the FastAPI service:

- ``latest_frame()`` / ``latest_jpeg()`` for display and streaming
- ``next_cloth()`` / ``set_cloth_by_id()`` for garment switching
- ``take_screenshot()`` / ``start_recording()`` / ``stop_recording()``
- ``toggle_studio()`` / ``set_mode()`` for Virtual Studio and body modes
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.camera.webcam import WebcamStream
from src.core.session_manager import SessionManager, SessionState
from src.ui.overlay import UIOverlay
from src.utils.config_loader import AppConfig
from src.utils.helpers import (
    ClothingCatalog, ClothingItem, FPSCounter, encode_jpeg, ensure_dirs,
    timestamped_filename,
)
from src.utils.logger import get_logger
from src.vision.clothing_renderer import ClothingRenderer
from src.vision.pose_estimator import PoseEstimator, PoseResult
from src.vision.segmenter import PersonSegmenter, StudioCompositor

logger = get_logger(__name__)

MODE_UPPER = "upper"
MODE_FULL = "full"

_INSTRUCTIONS = [
    "[N] next garment   [B] previous",
    "[S] screenshot      [R] record on/off",
    "[V] virtual studio  [M] upper/full body",
    "[D] pose debug      [Q] quit",
]


class TryOnEngine:
    """Owns the real-time pipeline. One instance per process/session."""

    def __init__(self, config: AppConfig,
                 camera: Optional[WebcamStream] = None,
                 draw_ui: bool = True,
                 enable_vision: bool = True) -> None:
        self.config = config
        self.draw_ui = draw_ui
        # When False, MediaPipe graphs are never constructed: the engine still
        # captures, renders the HUD, records, and serves the API — used for
        # headless demos, CI and tests.
        self.enable_vision = enable_vision

        self.catalog = ClothingCatalog(
            config.resolve_path(config.clothes.directory),
            catalog_file=config.clothes.catalog_file)
        self.sessions = SessionManager()

        # External camera injection keeps the engine testable.
        self._camera = camera
        self._owns_camera = camera is None

        self._pose: Optional[PoseEstimator] = None
        self._segmenter: Optional[PersonSegmenter] = None
        self._studio: Optional[StudioCompositor] = None
        self._renderer = ClothingRenderer(config.rendering, config.vision.tracking.min_visibility)
        self._ui = UIOverlay(config.ui)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = threading.Event()

        # Thread-safe publication of the latest composed frame.
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._last_processed: Optional[np.ndarray] = None  # clean, no UI
        self._last_pose: Optional[PoseResult] = None
        self._last_garment_drawn = False

        # Recording
        self._rec_lock = threading.Lock()
        self._writer: Optional[cv2.VideoWriter] = None
        self._recording = False
        self._recording_path: Optional[Path] = None

        # Toasts (transient on-screen messages)
        self._toast_lock = threading.Lock()
        self._toast_text: str = ""
        self._toast_until: float = 0.0

        self._fps = FPSCounter()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "TryOnEngine":
        if self._running.is_set():
            return self
        cfg = self.config
        if self._owns_camera:
            self._camera = WebcamStream(cfg.camera).start()
        if self.enable_vision:
            self._pose = PoseEstimator(cfg.vision)
            self._segmenter = PersonSegmenter(cfg.vision.segmentation)
            self._studio = StudioCompositor(cfg.studio)

        ensure_dirs(cfg.resolve_path(cfg.output.screenshots_dir),
                    cfg.resolve_path(cfg.output.recordings_dir))

        self.sessions.start_session(metadata={
            "clothes_available": len(self.catalog),
            "resolution": list(self._camera.resolution) if self._camera else None,
        })
        self._sync_default_cloth()

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="tryon-engine", daemon=True)
        self._thread.start()
        self._running.set()
        logger.info("TryOnEngine started (%d garment(s) loaded)", len(self.catalog))
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._stop_recording_internal(finalize=True)
        if self._pose:
            self._pose.close()
        if self._segmenter:
            self._segmenter.close()
        if self._owns_camera and self._camera:
            self._camera.stop()
        self.sessions.end_session()
        self._running.clear()
        logger.info("TryOnEngine stopped")

    def __enter__(self) -> "TryOnEngine":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    # ------------------------------------------------------------------
    # Main loop (single owner of all MediaPipe graphs)
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        assert self._camera is not None
        while not self._stop.is_set():
            ok, frame = self._camera.read(timeout=0.5)
            if not ok or frame is None:
                continue
            try:
                self._process_frame(frame)
            except Exception:  # pragma: no cover - defensive: never kill the loop
                logger.exception("Frame processing failed")
        logger.debug("Engine loop exited")

    def _process_frame(self, frame: np.ndarray) -> None:
        state = self.sessions.current

        # 1. Pose
        pose = self._pose.process(frame) if self._pose else None
        self._last_pose = pose

        # 2. Virtual Studio (background replacement)
        studio_on = bool(state and state.studio_enabled)
        if studio_on and self._segmenter and self._studio:
            mask = self._segmenter.process(frame)
            if mask is not None:
                frame = self._studio.composite(frame, mask)

        # 3. Garment rendering
        garment_drawn = False
        if state and state.cloth_index is not None and len(self.catalog) > 0:
            try:
                item = self.catalog.get(state.cloth_index)
                if item.path.suffix.lower() == ".obj":
                    vertices, faces = self.catalog.load_mesh(item)
                    garment_drawn = self._renderer.render_mesh(frame, vertices, faces, pose, item)
                else:
                    garment = self.catalog.load_image(item)
                    garment_drawn = self._renderer.render(frame, garment, pose, item)
                if garment_drawn:
                    state.frames_dressed += 1
            except (FileNotFoundError, ValueError) as exc:
                logger.error("Garment render failed: %s", exc)

        state.frames_processed += 1 if state else 0  # type: ignore[union-attr]

        # Keep a clean copy (no HUD) for screenshots/recordings.
        with self._frame_lock:
            self._last_processed = frame.copy()

        # 4. Recording happens on the clean frame (no HUD overlay in videos).
        self._write_recording(frame)

        # 5. HUD (skipped for API-driven headless streaming if disabled).
        display = frame
        if self.draw_ui:
            display = frame.copy()
            fps = self._fps.tick()
            self._draw_hud(display, state, pose, garment_drawn)

        with self._frame_lock:
            self._latest_frame = display
        self._last_garment_drawn = garment_drawn

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray, state: Optional[SessionState],
                  pose: Optional[PoseResult], garment_drawn: bool) -> None:
        cloth_name: Optional[str] = None
        if state and state.cloth_index is not None and len(self.catalog) > 0:
            cloth_name = self.catalog.get(state.cloth_index).name

        if len(self.catalog) == 0:
            self._ui.draw_empty_catalog(frame)

        tracking = pose is not None and pose.tracked
        self._ui.draw_header(
            frame, cloth_name=cloth_name, fps=self._fps.fps,
            recording=self.is_recording,
            studio_on=bool(state and state.studio_enabled),
            tracking=tracking)

        mode = (state.mode if state else MODE_UPPER)
        self._ui.draw_instructions(frame, _INSTRUCTIONS + [f"Mode: {mode} body"])

        if self.config.ui.show_landmarks and pose and self._pose:
            self._pose.draw_debug(frame, pose)

        with self._toast_lock:
            if time.monotonic() < self._toast_until and self._toast_text:
                self._ui.draw_toast(frame, self._toast_text)

    def _toast(self, message: str, seconds: float = 2.0) -> None:
        with self._toast_lock:
            self._toast_text = message
            self._toast_until = time.monotonic() + seconds

    # ------------------------------------------------------------------
    # Garment control (thread-safe)
    # ------------------------------------------------------------------

    def _sync_default_cloth(self) -> None:
        state = self.sessions.current
        if state is None or len(self.catalog) == 0:
            return
        default_id = self.config.clothes.default_item
        if default_id:
            item = self.catalog.find(default_id)
            if item:
                state.cloth_index = self.catalog.items.index(item)
                return
        state.cloth_index = 0

    def next_cloth(self, step: int = 1) -> Optional[ClothingItem]:
        state = self.sessions.current
        if state is None or len(self.catalog) == 0:
            return None
        state.next_cloth(len(self.catalog), step)
        item = self.catalog.get(state.cloth_index or 0)
        self._toast(item.name)
        logger.info("Garment changed -> %s", item.id, extra={"cloth": item.id})
        return item

    def set_cloth_by_id(self, cloth_id: str) -> ClothingItem:
        item = self.catalog.find(cloth_id)
        if item is None:
            raise KeyError(f"Unknown clothing item: {cloth_id!r}")
        state = self.sessions.current
        if state is None:
            raise RuntimeError("No active session")
        state.cloth_index = self.catalog.items.index(item)
        self._toast(item.name)
        return item

    def current_cloth(self) -> Optional[ClothingItem]:
        state = self.sessions.current
        if state is None or state.cloth_index is None or len(self.catalog) == 0:
            return None
        return self.catalog.get(state.cloth_index)

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------

    def toggle_studio(self) -> bool:
        state = self.sessions.current
        if state is None:
            return False
        state.studio_enabled = not state.studio_enabled
        self._toast(f"Virtual Studio {'ON' if state.studio_enabled else 'OFF'}")
        return state.studio_enabled

    def set_mode(self, mode: str) -> str:
        if mode not in {MODE_UPPER, MODE_FULL}:
            raise ValueError(f"mode must be '{MODE_UPPER}' or '{MODE_FULL}'")
        state = self.sessions.current
        if state is not None:
            state.mode = mode
            # Full-body framing: narrower relative garment scale reads better.
            factor = 2.25 if mode == MODE_UPPER else 1.9
            self._renderer.config.shoulder_width_factor = factor
            self._toast(f"Mode: {mode} body")
        return mode

    def toggle_mode(self) -> str:
        state = self.sessions.current
        current = state.mode if state else MODE_UPPER
        return self.set_mode(MODE_FULL if current == MODE_UPPER else MODE_UPPER)

    def toggle_debug(self) -> bool:
        self.config.ui.show_landmarks = not self.config.ui.show_landmarks
        return self.config.ui.show_landmarks

    # ------------------------------------------------------------------
    # Capture: screenshots + recording
    # ------------------------------------------------------------------

    def take_screenshot(self) -> Path:
        with self._frame_lock:
            source = self._last_processed
            frame = None if source is None else source.copy()
        if frame is None:
            raise RuntimeError("No frame available yet — is the camera running?")

        item = self.current_cloth()
        out_dir = self.config.resolve_path(self.config.output.screenshots_dir)
        ensure_dirs(out_dir)
        name = timestamped_filename("tryon", "", self.config.output.screenshot_format,
                                    extra=item.name if item else "")
        path = out_dir / name
        params = []
        if name.lower().endswith((".jpg", ".jpeg")):
            params = [cv2.IMWRITE_JPEG_QUALITY, self.config.output.jpeg_quality]
        if not cv2.imwrite(str(path), frame, params):
            raise RuntimeError(f"cv2.imwrite failed for {path}")
        state = self.sessions.current
        if state:
            state.screenshots_taken += 1
        self._toast("Screenshot saved")
        logger.info("Screenshot saved: %s", path)
        return path

    def start_recording(self) -> Path:
        with self._rec_lock:
            if self._recording:
                assert self._recording_path is not None
                return self._recording_path
            with self._frame_lock:
                ref = self._last_processed
            if ref is None:
                raise RuntimeError("No frame available yet — is the camera running?")
            h, w = ref.shape[:2]
            out_dir = self.config.resolve_path(self.config.output.recordings_dir)
            ensure_dirs(out_dir)
            path = out_dir / timestamped_filename("tryon", "session", "mp4")
            fourcc = cv2.VideoWriter_fourcc(*self.config.output.recording_codec)
            writer = cv2.VideoWriter(str(path), fourcc, self.config.output.recording_fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {path}")
            self._writer = writer
            self._recording = True
            self._recording_path = path
        self._toast("Recording started")
        logger.info("Recording started: %s", path)
        return path

    def _stop_recording_internal(self, finalize: bool) -> Optional[Path]:
        with self._rec_lock:
            if not self._recording:
                return None
            if self._writer is not None:
                self._writer.release()
            path = self._recording_path
            self._writer = None
            self._recording = False
            self._recording_path = None
        state = self.sessions.current
        if state:
            state.recordings_made += 1
        if finalize:
            logger.info("Recording saved: %s", path)
        return path

    def stop_recording(self) -> Path:
        path = self._stop_recording_internal(finalize=True)
        if path is None:
            raise RuntimeError("Not currently recording")
        self._toast("Recording saved")
        return path

    def toggle_recording(self) -> Optional[Path]:
        """Start if idle; stop and return the file path if recording."""
        return self.stop_recording() if self.is_recording else self.start_recording()

    def _write_recording(self, frame: np.ndarray) -> None:
        with self._rec_lock:
            if self._recording and self._writer is not None:
                self._writer.write(frame)

    @property
    def is_recording(self) -> bool:
        with self._rec_lock:
            return self._recording

    # ------------------------------------------------------------------
    # Frame access for API streaming
    # ------------------------------------------------------------------

    def latest_frame(self, with_ui: bool = True) -> Optional[np.ndarray]:
        with self._frame_lock:
            source = self._latest_frame if with_ui else self._last_processed
            return None if source is None else source.copy()

    def latest_jpeg(self, quality: Optional[int] = None,
                    with_ui: bool = True) -> Optional[bytes]:
        frame = self.latest_frame(with_ui=with_ui)
        if frame is None:
            return None
        q = quality if quality is not None else self.config.api.stream_jpeg_quality
        return encode_jpeg(frame, q)

    # ------------------------------------------------------------------

    def status(self) -> Dict[str, object]:
        state = self.sessions.current
        item = self.current_cloth()
        return {
            "running": self.is_running,
            "recording": self.is_recording,
            "studio_enabled": bool(state and state.studio_enabled),
            "mode": state.mode if state else MODE_UPPER,
            "tracking": bool(self._last_pose and self._last_pose.tracked
                              and self._last_pose.shoulder_visibility >= self.config.vision.tracking.min_visibility
                              and self._last_garment_drawn),
            "current_cloth": item.to_dict() if item else None,
            "clothes_count": len(self.catalog),
            "fps": round(self._fps.fps, 1),
            "session": state.to_dict() if state else None,
        }
