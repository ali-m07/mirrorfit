"""Centralised, production-style logging for the whole platform.

- JSON or plain-text formatting (chosen via ``config.yaml``)
- Rotating file handler + coloured console handler
- ``get_logger(__name__)`` is the only import callers need
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_CONFIGURED = False
_LOG_FILE: Optional[Path] = None


class JSONFormatter(logging.Formatter):
    """One JSON object per line — ingestible by ELK / CloudWatch / Loki."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key in ("session_id", "cloth", "fps"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


class ColourFormatter(logging.Formatter):
    """Human-friendly console output with ANSI colours."""

    COLOURS = {
        "DEBUG": "\033[90m",
        "INFO": "\033[36m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        colour = self.COLOURS.get(record.levelname, self.RESET)
        record.msg = f"{colour}{record.getMessage()}{self.RESET}"
        record.args = ()
        return super().format(record)


def configure_logging(level: str = "INFO",
                      log_file: Optional[str] = None,
                      fmt: str = "plain",
                      max_bytes: int = 5 * 1024 * 1024,
                      backup_count: int = 3) -> None:
    """Configure the root logger once; subsequent calls are no-ops."""
    global _CONFIGURED, _LOG_FILE
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        console.setFormatter(JSONFormatter())
    else:
        console.setFormatter(ColourFormatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
                                             datefmt="%H:%M:%S"))
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        file_handler.setFormatter(JSONFormatter() if fmt == "json"
                                  else logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        root.addHandler(file_handler)
        _LOG_FILE = path

    # MediaPipe / absl are noisy at INFO — keep them at WARNING.
    logging.getLogger("absl").setLevel(logging.WARNING)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, auto-configuring with defaults if needed."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
