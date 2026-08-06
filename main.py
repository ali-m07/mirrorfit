"""Desktop application entry point — the retail-kiosk experience.

Run::

    python main.py

Keyboard controls (also shown on-screen):
    N / B   next / previous garment
    S       screenshot (timestamped, with garment name)
    R       start/stop video recording
    V       toggle Virtual Studio (background replacement)
    M       toggle upper-body / full-body mode
    D       toggle pose debug skeleton
    Q / Esc quit
"""

from __future__ import annotations

import sys

import cv2

from src.core.tryon_engine import TryOnEngine
from src.utils.config_loader import load_config
from src.utils.logger import configure_logging, get_logger


def main() -> int:
    cfg = load_config()
    configure_logging(level=cfg.logging.level,
                      log_file=str(cfg.resolve_path(cfg.logging.file)),
                      fmt=cfg.logging.format,
                      max_bytes=cfg.logging.max_bytes,
                      backup_count=cfg.logging.backup_count)
    logger = get_logger("main")
    logger.info("Starting %s v%s", cfg.app.name, cfg.app.version)

    window = cfg.app.window_title
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        engine = TryOnEngine(cfg, draw_ui=True).start()
    except RuntimeError as exc:
        logger.error("Startup failed: %s", exc)
        return 2

    with engine:
        logger.info("Desktop app running — press Q in the video window to quit")
        while True:
            frame = engine.latest_frame()
            if frame is not None:
                cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("n"), ord("N")):
                engine.next_cloth(1)
            elif key in (ord("b"), ord("B")):
                engine.next_cloth(-1)
            elif key in (ord("s"), ord("S")):
                try:
                    path = engine.take_screenshot()
                    logger.info("Screenshot: %s", path)
                except RuntimeError as exc:
                    logger.warning("Screenshot failed: %s", exc)
            elif key in (ord("r"), ord("R")):
                try:
                    result = engine.toggle_recording()
                    logger.info("Recording toggled -> %s", result)
                except RuntimeError as exc:
                    logger.warning("Recording failed: %s", exc)
            elif key in (ord("v"), ord("V")):
                logger.info("Virtual Studio: %s", engine.toggle_studio())
            elif key in (ord("m"), ord("M")):
                logger.info("Mode: %s", engine.toggle_mode())
            elif key in (ord("d"), ord("D")):
                logger.info("Pose debug: %s", engine.toggle_debug())

    cv2.destroyAllWindows()
    logger.info("Goodbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
