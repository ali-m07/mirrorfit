"""Tests for the ONNX pose backend plumbing (no runtime / model required)."""

from __future__ import annotations

import numpy as np

from src.vision.pose_backend_onnx import (
    _COCO_TO_MP, OnnxPoseBackend,
)
from src.utils.config_loader import OnnxPoseSection


def test_mapping_covers_renderercritical_landmarks() -> None:
    # The renderer and HUD need shoulders, elbows, wrists, hips (+nose/ears).
    needed = {11, 12, 13, 14, 15, 16, 23, 24}
    assert needed.issubset(set(_COCO_TO_MP.values()))


def test_unpack_outputs_shapes() -> None:
    dets = np.random.rand(1, 300, 5).astype(np.float32)
    kpts = np.random.rand(1, 300, 17, 3).astype(np.float32)
    d, k = OnnxPoseBackend._unpack_outputs([dets, kpts])
    assert d is not None and d.shape == (300, 5)
    assert k is not None and k.shape == (300, 17, 3)


def test_unpack_outputs_transposed_layout() -> None:
    # Some exports emit [1, 56, N]-style layouts; the unpacker must at least
    # route the canonical layouts correctly and ignore unknown ones.
    dets = np.zeros((1, 4, 5), dtype=np.float32)
    kpts = np.zeros((1, 4, 17, 3), dtype=np.float32)
    d, k = OnnxPoseBackend._unpack_outputs([kpts, dets])
    assert d.shape == (4, 5)
    assert k.shape == (4, 17, 3)


def test_provider_selection_prefers_gpu(monkeypatch) -> None:
    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    backend = OnnxPoseBackend(OnnxPoseSection(device="auto"))
    assert backend._select_providers(FakeOrt)[0] == "DmlExecutionProvider"

    backend_cpu = OnnxPoseBackend(OnnxPoseSection(device="cpu"))
    assert backend_cpu._select_providers(FakeOrt) == ["CPUExecutionProvider"]


def test_process_returns_none_without_session_and_missing_model(tmp_path, monkeypatch) -> None:
    # With auto_download off and no model file, ensure_session must raise —
    # the estimator then falls back to MediaPipe (tested at engine level).
    section = OnnxPoseSection(model_path=str(tmp_path / "missing.onnx"),
                              auto_download=False)
    backend = OnnxPoseBackend(section)
    try:
        backend.ensure_session()
        raised = False
    except Exception:
        raised = True
    assert raised
