"""Configuration loading with YAML + environment-variable overrides.

The whole platform reads a single :class:`AppConfig` instance. Values come
from ``config.yaml`` and can be overridden with environment variables using
double underscores for nesting, e.g.::

    ARTRYON__CAMERA__INDEX=1   ->  cfg.camera.index = 1

This gives twelve-factor-style deployability for the API service while keeping
local desktop runs zero-config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml

ENV_PREFIX = "ARTRYON"
T = TypeVar("T")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Typed configuration schema
# ---------------------------------------------------------------------------

@dataclass
class AppSection:
    name: str = "AR Virtual Try-On"
    version: str = "1.0.0"
    window_title: str = "AR Virtual Try-On Studio"
    organization: str = "SnapStyle Labs"


@dataclass
class CameraSection:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    buffer_size: int = 2
    backend: str = "auto"          # auto | dshow | msmf | v4l2 | avfoundation


@dataclass
class PoseSection:
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    smooth_landmarks: bool = True
    enable_segmentation: bool = False


@dataclass
class SegmentationSection:
    model_selection: int = 1


@dataclass
class TrackingSection:
    min_visibility: float = 0.6
    smoothing_alpha: float = 0.45
    max_missing_frames: int = 5
    lost_pose_timeout_s: float = 0.8


@dataclass
class VisionSection:
    pose: PoseSection = field(default_factory=PoseSection)
    segmentation: SegmentationSection = field(default_factory=SegmentationSection)
    tracking: TrackingSection = field(default_factory=TrackingSection)


@dataclass
class ShadowSection:
    enabled: bool = True
    opacity: float = 0.25
    blur_radius: int = 9
    offset_x: int = 6
    offset_y: int = 12


@dataclass
class LightingSection:
    enabled: bool = True
    match_scene_brightness: bool = True


@dataclass
class OcclusionSection:
    use_arm_cutout: bool = True
    arm_mask_dilation: int = 7


@dataclass
class RenderingSection:
    shoulder_width_factor: float = 2.25
    neck_offset_factor: float = 0.14
    perspective_tilt: bool = True
    max_rotation_deg: float = 35.0
    min_shoulder_width_px: int = 24
    shadow: ShadowSection = field(default_factory=ShadowSection)
    lighting: LightingSection = field(default_factory=LightingSection)
    occlusion: OcclusionSection = field(default_factory=OcclusionSection)


@dataclass
class StudioSection:
    enabled: bool = False
    mode: str = "gradient"       # gradient | image | blur
    background_image: str = "assets/studio/backdrop.png"
    gradient_top: List[int] = field(default_factory=lambda: [24, 26, 38])
    gradient_bottom: List[int] = field(default_factory=lambda: [64, 78, 104])
    blur_strength: int = 35
    segmentation_threshold: float = 0.5


@dataclass
class ThemeSection:
    accent: List[int] = field(default_factory=lambda: [255, 178, 36])
    panel_bg: List[int] = field(default_factory=lambda: [18, 20, 28])
    text: List[int] = field(default_factory=lambda: [245, 245, 245])


@dataclass
class UISection:
    show_fps: bool = True
    show_instructions: bool = True
    show_cloth_name: bool = True
    show_landmarks: bool = False
    font_scale: float = 0.65
    theme: ThemeSection = field(default_factory=ThemeSection)
    panel_opacity: float = 0.72


@dataclass
class OutputSection:
    screenshots_dir: str = "output/screenshots"
    recordings_dir: str = "output/recordings"
    recording_fps: int = 24
    recording_codec: str = "mp4v"
    screenshot_format: str = "png"
    jpeg_quality: int = 92


@dataclass
class APISection:
    host: str = "0.0.0.0"
    port: int = 8000
    stream_jpeg_quality: int = 80
    stream_max_fps: int = 20
    websocket_path: str = "/ws/stream"


@dataclass
class ClothesSection:
    directory: str = "assets/clothes"
    catalog_file: str = "catalog.json"
    default_item: Optional[str] = None


@dataclass
class LoggingSection:
    level: str = "INFO"
    file: str = "logs/tryon.log"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 3
    format: str = "json"        # json | plain


@dataclass
class AppConfig:
    """Root configuration object shared by every component."""

    app: AppSection = field(default_factory=AppSection)
    camera: CameraSection = field(default_factory=CameraSection)
    vision: VisionSection = field(default_factory=VisionSection)
    rendering: RenderingSection = field(default_factory=RenderingSection)
    studio: StudioSection = field(default_factory=StudioSection)
    ui: UISection = field(default_factory=UISection)
    output: OutputSection = field(default_factory=OutputSection)
    api: APISection = field(default_factory=APISection)
    clothes: ClothesSection = field(default_factory=ClothesSection)
    logging: LoggingSection = field(default_factory=LoggingSection)

    # -- convenience ------------------------------------------------------

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve ``path`` against the project root when it is relative."""
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _coerce(value: Any, current: Any) -> Any:
    """Best-effort coercion of an env-var string to the type of ``current``."""
    if isinstance(current, bool):
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(str(value).strip())
        except ValueError:
            return current
    if isinstance(current, float):
        try:
            return float(str(value).strip())
        except ValueError:
            return current
    return value


def _apply_dict(instance: Any, data: Dict[str, Any]) -> None:
    """Recursively apply a mapping onto a dataclass instance."""
    known = {f.name: f for f in fields(instance)}
    for key, value in data.items():
        if key not in known:
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_dict(current, value)
        elif current is not None and isinstance(value, str) and not isinstance(current, str):
            setattr(instance, key, _coerce(value, current))
        else:
            setattr(instance, key, value)


def _apply_env_overrides(instance: Any, path: tuple[str, ...] = ()) -> None:
    """Walk the dataclass tree applying ``ARTRYON__SECTION__KEY`` overrides."""
    for f in fields(instance):
        current = getattr(instance, f.name)
        new_path = path + (f.name.upper(),)
        if is_dataclass(current):
            _apply_env_overrides(current, new_path)
            continue
        env_key = f"{ENV_PREFIX}__{'__'.join(new_path)}"
        if env_key in os.environ:
            setattr(instance, f.name, _coerce(os.environ[env_key], current))


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML, then apply environment overrides.

    Missing or malformed files fall back to built-in defaults so the system
    always starts — degradation is logged, never fatal at import time.
    """
    cfg = AppConfig()
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if cfg_path.is_file():
        try:
            with cfg_path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            if isinstance(raw, dict):
                _apply_dict(cfg, raw)
        except (yaml.YAMLError, OSError):  # pragma: no cover - defensive
            pass
    _apply_env_overrides(cfg)
    return cfg


__all__ = ["AppConfig", "load_config", "DEFAULT_CONFIG_PATH", "PROJECT_ROOT"]
