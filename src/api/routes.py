"""FastAPI route definitions.

The router is engine-agnostic: it resolves the :class:`TryOnEngine` from
``request.app.state.engine`` so the same router works under ``uvicorn`` and
under tests with a stubbed engine.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    ChangeClothRequest, ClothesResponse, ClothingItemSchema, FileResponse,
    HealthResponse, MessageResponse, ModeRequest, RecordRequest,
    SessionInfo, StartSessionRequest, StatusResponse, StudioRequest,
)
from src.core.tryon_engine import TryOnEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()
_STARTED_AT = time.time()


def _engine(request: Request) -> TryOnEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    return engine


def _require_running(engine: TryOnEngine) -> None:
    if not engine.is_running:
        raise HTTPException(status_code=409,
                            detail="No active session — POST /tryon/start first")


def _item_schema(item) -> ClothingItemSchema:
    return ClothingItemSchema(
        id=item.id, name=item.name, filename=item.filename,
        category=item.category, description=item.description,
        anchor_top=item.anchor_top, anchor_width=item.anchor_width)


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    engine = getattr(request.app.state, "engine", None)
    cfg = request.app.state.config
    return HealthResponse(app=cfg.app.name, version=cfg.app.version,
                          engine_running=bool(engine and engine.is_running),
                          uptime_s=round(time.time() - _STARTED_AT, 2))


@router.get("/status", response_model=StatusResponse, tags=["meta"])
def status(request: Request) -> StatusResponse:
    engine = _engine(request)
    data = engine.status()
    item = data.pop("current_cloth", None)
    return StatusResponse(
        **data,
        current_cloth=ClothingItemSchema(**item) if item else None)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

@router.get("/clothes", response_model=ClothesResponse, tags=["catalog"])
def list_clothes(request: Request) -> ClothesResponse:
    engine = _engine(request)
    items = [_item_schema(i) for i in engine.catalog.items]
    return ClothesResponse(count=len(items), items=items)


# ---------------------------------------------------------------------------
# Session + control
# ---------------------------------------------------------------------------

@router.post("/tryon/start", response_model=SessionInfo, tags=["session"])
def start_session(request: Request, body: StartSessionRequest | None = None) -> SessionInfo:
    engine = _engine(request)
    if body and body.camera_index is not None:
        engine.config.camera.index = body.camera_index
    if not engine.is_running:
        try:
            engine.start()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    state = engine.sessions.current
    assert state is not None
    return SessionInfo(**state.to_dict())


@router.post("/tryon/stop", response_model=MessageResponse, tags=["session"])
def stop_session(request: Request) -> MessageResponse:
    engine = _engine(request)
    ended = engine.sessions.end_session()
    engine.stop()
    return MessageResponse(ok=True, message="Session stopped",
                           data={"session": ended.to_dict() if ended else {}})


@router.post("/tryon/change-cloth", response_model=MessageResponse, tags=["control"])
def change_cloth(request: Request, body: ChangeClothRequest) -> MessageResponse:
    engine = _engine(request)
    _require_running(engine)
    try:
        if body.cloth_id:
            item = engine.set_cloth_by_id(body.cloth_id)
        else:
            item = engine.next_cloth(1 if body.direction == "next" else -1)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=409, detail="Clothing catalog is empty")
    return MessageResponse(ok=True, message=f"Now wearing: {item.name}",
                           data={"cloth": _item_schema(item).model_dump()})


@router.post("/tryon/screenshot", response_model=FileResponse, tags=["control"])
def screenshot(request: Request) -> FileResponse:
    engine = _engine(request)
    _require_running(engine)
    try:
        path = engine.take_screenshot()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return FileResponse(ok=True, path=str(path), filename=Path(path).name)


@router.post("/tryon/record", response_model=MessageResponse, tags=["control"])
def record(request: Request, body: RecordRequest) -> MessageResponse:
    engine = _engine(request)
    _require_running(engine)
    try:
        if body.action == "start" or (body.action == "toggle" and not engine.is_recording):
            path = engine.start_recording()
            return MessageResponse(ok=True, message="Recording started",
                                   data={"path": str(path), "recording": True})
        path = engine.stop_recording()
        return MessageResponse(ok=True, message="Recording saved",
                               data={"path": str(path), "recording": False})
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/tryon/mode", response_model=MessageResponse, tags=["control"])
def set_mode(request: Request, body: ModeRequest) -> MessageResponse:
    engine = _engine(request)
    _require_running(engine)
    mode = engine.set_mode(body.mode)
    return MessageResponse(ok=True, message=f"Mode set to {mode}", data={"mode": mode})


@router.post("/tryon/studio", response_model=MessageResponse, tags=["control"])
def set_studio(request: Request, body: StudioRequest) -> MessageResponse:
    engine = _engine(request)
    _require_running(engine)
    state = engine.sessions.current
    assert state is not None
    state.studio_enabled = body.enabled
    return MessageResponse(ok=True,
                           message=f"Virtual Studio {'enabled' if body.enabled else 'disabled'}",
                           data={"studio_enabled": body.enabled})


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

@router.get("/stream.mjpeg", tags=["streaming"])
def mjpeg_stream(request: Request) -> StreamingResponse:
    """Multipart MJPEG stream — drop-in <img src="/stream.mjpeg"> for demos."""
    engine = _engine(request)
    _require_running(engine)
    boundary = "tryonframe"
    min_interval = 1.0 / max(1, request.app.state.config.api.stream_max_fps)

    def generate():
        last = 0.0
        while True:
            now = time.monotonic()
            if now - last < min_interval:
                time.sleep(min_interval - (now - last))
            last = time.monotonic()
            jpeg = engine.latest_jpeg()
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield (f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                   f"Content-Length: {len(jpeg)}\r\n\r\n").encode() + jpeg + b"\r\n"

    return StreamingResponse(
        generate(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """Binary JPEG frames pushed at the configured stream rate."""
    engine = getattr(websocket.app.state, "engine", None)
    if engine is None or not engine.is_running:
        await websocket.close(code=1013, reason="Engine not running")
        return
    await websocket.accept()
    cfg = websocket.app.state.config
    interval = 1.0 / max(1, cfg.api.stream_max_fps)
    logger.info("WebSocket client connected: %s", websocket.client)
    try:
        while True:
            jpeg = engine.latest_jpeg()
            if jpeg is not None:
                await websocket.send_bytes(jpeg)
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:  # pragma: no cover - defensive
        logger.exception("WebSocket stream error")
        await websocket.close(code=1011)
