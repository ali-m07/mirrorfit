"""Pydantic schemas for the public API (v1).

Versioned request/response contracts — the same models a white-label SaaS
deployment would publish to partners.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Garments
# ---------------------------------------------------------------------------

class ClothingItemSchema(BaseModel):
    id: str
    name: str
    filename: str
    category: str = "upper"
    description: str = ""
    anchor_top: float = 0.0
    anchor_width: float = 1.0


class ClothesResponse(BaseModel):
    count: int
    items: List[ClothingItemSchema]


# ---------------------------------------------------------------------------
# Session control
# ---------------------------------------------------------------------------

class StartSessionRequest(BaseModel):
    camera_index: Optional[int] = Field(
        default=None, description="Override camera index for this session")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionInfo(BaseModel):
    session_id: str
    started_at: float
    ended_at: Optional[float] = None
    active: bool
    duration_s: float
    mode: str
    studio_enabled: bool
    cloth_index: Optional[int] = None
    frames_processed: int = 0
    frames_dressed: int = 0
    screenshots_taken: int = 0
    recordings_made: int = 0
    cloth_switches: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class ChangeClothRequest(BaseModel):
    cloth_id: Optional[str] = Field(
        default=None, description="Garment id or filename. Omit to cycle to next.")
    direction: Literal["next", "previous"] = "next"


class RecordRequest(BaseModel):
    action: Literal["start", "stop", "toggle"] = "toggle"


class ModeRequest(BaseModel):
    mode: Literal["upper", "full"]


class StudioRequest(BaseModel):
    enabled: bool


class FileResponse(BaseModel):
    ok: bool
    path: str
    filename: str


class MessageResponse(BaseModel):
    ok: bool
    message: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str
    engine_running: bool
    uptime_s: float


class StatusResponse(BaseModel):
    running: bool
    recording: bool
    studio_enabled: bool
    mode: str
    tracking: bool
    current_cloth: Optional[ClothingItemSchema] = None
    clothes_count: int
    fps: float
    session: Optional[Dict[str, Any]] = None
