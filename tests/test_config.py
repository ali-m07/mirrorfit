"""Tests for configuration loading and overrides."""

from __future__ import annotations

from pathlib import Path

from src.utils.config_loader import AppConfig, load_config


def test_defaults_are_sane() -> None:
    cfg = AppConfig()
    assert cfg.camera.width == 1280
    assert cfg.vision.tracking.smoothing_alpha > 0
    assert cfg.api.port == 8000


def test_yaml_overrides(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "camera:\n  width: 640\n  index: 2\n"
        "rendering:\n  shoulder_width_factor: 3.0\n",
        encoding="utf-8")
    cfg = load_config(config_file)
    assert cfg.camera.width == 640
    assert cfg.camera.index == 2
    assert cfg.rendering.shoulder_width_factor == 3.0
    assert cfg.camera.height == 720  # untouched defaults survive


def test_env_override(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("camera:\n  index: 0\n", encoding="utf-8")
    monkeypatch.setenv("ARTRYON__CAMERA__INDEX", "3")
    monkeypatch.setenv("ARTRYON__STUDIO__ENABLED", "true")
    cfg = load_config(config_file)
    assert cfg.camera.index == 3
    assert cfg.studio.enabled is True


def test_missing_file_falls_back_to_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.api.port == 8000
