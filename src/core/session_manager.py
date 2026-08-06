"""Session lifecycle and state tracking.

A *session* is one continuous try-on experience: camera open -> camera closed.
The session manager owns the mutable per-session state (current garment, mode,
studio toggle, counters) and provides the audit trail a commercial deployment
needs (durations, engagement counters).

State objects are mutated only by the engine and guarded implicitly by the
engine's single-writer design; API routes read snapshots via ``to_dict()``.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SessionState:
    session_id: str
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    cloth_index: Optional[int] = 0
    mode: str = "upper"                # upper | full
    studio_enabled: bool = False
    frames_processed: int = 0
    frames_dressed: int = 0
    screenshots_taken: int = 0
    recordings_made: int = 0
    cloth_switches: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.ended_at is None

    @property
    def duration_s(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return max(0.0, end - self.started_at)

    def next_cloth(self, catalog_size: int, step: int = 1) -> None:
        if catalog_size <= 0:
            self.cloth_index = None
            return
        current = self.cloth_index or 0
        self.cloth_index = (current + step) % catalog_size
        self.cloth_switches += 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "active": self.is_active,
            "duration_s": round(self.duration_s, 2),
            "mode": self.mode,
            "studio_enabled": self.studio_enabled,
            "cloth_index": self.cloth_index,
            "frames_processed": self.frames_processed,
            "frames_dressed": self.frames_dressed,
            "screenshots_taken": self.screenshots_taken,
            "recordings_made": self.recordings_made,
            "cloth_switches": self.cloth_switches,
            "metadata": dict(self.metadata),
        }


class SessionManager:
    """Creates, tracks and closes sessions. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Optional[SessionState] = None
        self._history: List[SessionState] = []

    def start_session(self, metadata: Optional[Dict[str, object]] = None) -> SessionState:
        with self._lock:
            if self._current is not None and self._current.is_active:
                return self._current
            state = SessionState(session_id=str(uuid.uuid4()),
                                 metadata=dict(metadata or {}))
            self._current = state
        logger.info("Session started: %s", state.session_id,
                    extra={"session_id": state.session_id})
        return state

    def end_session(self) -> Optional[SessionState]:
        with self._lock:
            if self._current is None or not self._current.is_active:
                return None
            self._current.ended_at = time.time()
            self._history.append(self._current)
            ended = self._current
            self._current = None
        logger.info("Session ended: %s (%.1fs, %d screenshots, %d recordings)",
                    ended.session_id, ended.duration_s,
                    ended.screenshots_taken, ended.recordings_made)
        return ended

    @property
    def current(self) -> Optional[SessionState]:
        with self._lock:
            return self._current

    @property
    def history(self) -> List[SessionState]:
        with self._lock:
            return list(self._history)
