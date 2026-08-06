"""FastAPI application entry point.

Run::

    python api_server.py            # uses config.yaml api.host / api.port
    uvicorn api_server:app --reload # development

The TryOnEngine is created at startup but the *session* is only started via
POST /tryon/start — so the service can boot on machines without a camera
(e.g. CI) and still answer /health and /clothes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.core.tryon_engine import TryOnEngine
from src.utils.config_loader import PROJECT_ROOT, load_config
from src.utils.logger import configure_logging, get_logger

cfg = load_config()
configure_logging(level=cfg.logging.level,
                  log_file=str(cfg.resolve_path(cfg.logging.file)),
                  fmt=cfg.logging.format,
                  max_bytes=cfg.logging.max_bytes,
                  backup_count=cfg.logging.backup_count)
logger = get_logger(__name__)

app = FastAPI(
    title=cfg.app.name,
    version=cfg.app.version,
    description="Real-time AR virtual clothing try-on — desktop-grade pipeline, "
                "served as a white-label-ready HTTP/WebSocket API.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Permissive CORS for demos/white-label embedding; restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.config = cfg
app.state.engine = TryOnEngine(cfg, draw_ui=True)  # UI baked into streamed frames

app.include_router(router)

# Static demo frontend (plain HTML + JS) served at /demo
_web_dir = PROJECT_ROOT / "web"
if _web_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_web_dir)), name="static")

    @app.get("/demo", include_in_schema=False)
    def demo_page() -> FileResponse:
        return FileResponse(str(_web_dir / "index.html"))


@app.on_event("shutdown")
def _shutdown() -> None:
    engine: TryOnEngine = app.state.engine
    if engine.is_running:
        engine.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="AR Virtual Try-On API server")
    parser.add_argument("--host", default=cfg.api.host)
    parser.add_argument("--port", type=int, default=cfg.api.port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn
    logger.info("Starting API on http://%s:%d (docs: /docs, demo: /demo)",
                args.host, args.port)
    uvicorn.run("api_server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
