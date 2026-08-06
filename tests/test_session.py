"""Tests for session lifecycle."""

from __future__ import annotations

import time

from src.core.session_manager import SessionManager


def test_session_start_and_end() -> None:
    mgr = SessionManager()
    state = mgr.start_session(metadata={"source": "test"})
    assert state.is_active
    assert mgr.current is state
    ended = mgr.end_session()
    assert ended is state
    assert not ended.is_active
    assert mgr.current is None
    assert len(mgr.history) == 1


def test_double_start_returns_same_session() -> None:
    mgr = SessionManager()
    a = mgr.start_session()
    b = mgr.start_session()
    assert a is b


def test_cloth_cycling_counts_switches() -> None:
    mgr = SessionManager()
    state = mgr.start_session()
    state.next_cloth(catalog_size=3, step=1)
    state.next_cloth(catalog_size=3, step=1)
    assert state.cloth_index == 2
    state.next_cloth(catalog_size=3, step=1)
    assert state.cloth_index == 0          # wraps
    assert state.cloth_switches == 3
    state.next_cloth(catalog_size=3, step=-1)
    assert state.cloth_index == 2          # backwards wraps too


def test_duration_is_positive() -> None:
    mgr = SessionManager()
    state = mgr.start_session()
    time.sleep(0.01)
    assert state.duration_s > 0


def test_to_dict_shape() -> None:
    mgr = SessionManager()
    state = mgr.start_session()
    data = state.to_dict()
    for key in ("session_id", "active", "duration_s", "mode",
                "studio_enabled", "cloth_switches"):
        assert key in data
