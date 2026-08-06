"""On-screen UI: translucent panels, cloth info, FPS, instructions, badges.

Designed to look like a retail kiosk rather than a debug view: soft rounded
panels, a consistent accent colour, and everything data-driven from config.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.utils.config_loader import UISection

Color = Tuple[int, int, int]  # BGR

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _rounded_rect(img: np.ndarray, x: int, y: int, w: int, h: int,
                  radius: int, color: Color, opacity: float) -> None:
    """Draw a translucent rounded rectangle (in place)."""
    h_img, w_img = img.shape[:2]
    x2, y2 = min(x + w, w_img), min(y + h, h_img)
    x, y = max(x, 0), max(y, 0)
    if x2 <= x or y2 <= y:
        return
    roi = img[y:y2, x:x2]
    overlay = np.zeros_like(roi)
    rh, rw = roi.shape[:2]
    r = max(0, min(radius, rh // 2, rw // 2))
    if r > 0:
        cv2.rectangle(overlay, (r, 0), (rw - r, rh), color, -1)
        cv2.rectangle(overlay, (0, r), (rw, rh - r), color, -1)
        for cx, cy in ((r, r), (rw - r, r), (r, rh - r), (rw - r, rh - r)):
            cv2.circle(overlay, (cx, cy), r, color, -1)
    else:
        overlay[:] = color
    cv2.addWeighted(overlay, opacity, roi, 1.0 - opacity, 0, dst=roi)


def _text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float,
          color: Color, thickness: int = 1) -> None:
    cv2.putText(img, text, org, _FONT, scale, (0, 0, 0), thickness + 2,
                cv2.LINE_AA)  # subtle drop shadow for legibility
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


class UIOverlay:
    """Draws the HUD on top of the composed frame."""

    def __init__(self, config: UISection) -> None:
        self.config = config

    @property
    def _accent(self) -> Color:
        return tuple(int(c) for c in self.config.theme.accent)  # type: ignore[return-value]

    @property
    def _panel(self) -> Color:
        return tuple(int(c) for c in self.config.theme.panel_bg)  # type: ignore[return-value]

    @property
    def _text_color(self) -> Color:
        return tuple(int(c) for c in self.config.theme.text)  # type: ignore[return-value]

    # ------------------------------------------------------------------

    def draw_header(self, frame: np.ndarray, cloth_name: Optional[str],
                    fps: float, recording: bool, studio_on: bool,
                    tracking: bool) -> None:
        c = self.config
        s = c.font_scale
        title_parts: List[str] = []
        if c.show_cloth_name and cloth_name:
            title_parts.append(cloth_name)
        if studio_on:
            title_parts.append("Virtual Studio")
        title = "  |  ".join(title_parts) if title_parts else "AR Virtual Try-On"

        (tw, th), _ = cv2.getTextSize(title, _FONT, s + 0.15, 2)
        pad = 14
        _rounded_rect(frame, 12, 12, tw + pad * 2 + 26, th + pad * 2, 12,
                      self._panel, c.panel_opacity)
        # Accent chip
        cv2.rectangle(frame, (12 + pad - 6, 12 + pad - 2),
                      (12 + pad - 2, 12 + pad + th + 4), self._accent, -1)
        _text(frame, title, (12 + pad + 12, 12 + pad + th - 2), s + 0.15,
              self._text_color, 2)

        # Status indicators, right-aligned
        h, w = frame.shape[:2]
        x_cursor = w - 20
        if recording:
            label = "REC"
            (rw_, rh_), _ = cv2.getTextSize(label, _FONT, s, 2)
            _rounded_rect(frame, x_cursor - rw_ - 34, 16, rw_ + 34, rh_ + 18, 9,
                          (30, 30, 200), 0.85)
            cv2.circle(frame, (x_cursor - rw_ - 16, 25 + rh_ // 2), 6, (255, 255, 255), -1)
            _text(frame, label, (x_cursor - rw_ - 2, 22 + rh_), s, (255, 255, 255), 2)
            x_cursor -= rw_ + 50
        if c.show_fps:
            label = f"{fps:4.1f} FPS"
            (fw, fh), _ = cv2.getTextSize(label, _FONT, s, 1)
            _rounded_rect(frame, x_cursor - fw - 20, 16, fw + 20, fh + 18, 9,
                          self._panel, c.panel_opacity)
            _text(frame, label, (x_cursor - fw - 10, 22 + fh), s, self._accent, 1)
            x_cursor -= fw + 36

        # Tracking indicator dot (top-left under header)
        dot_color = (90, 220, 90) if tracking else (60, 60, 220)
        cv2.circle(frame, (26, 12 + th + pad * 2 + 14), 6, dot_color, -1, cv2.LINE_AA)
        _text(frame, "tracking" if tracking else "find your pose",
              (40, 12 + th + pad * 2 + 19), s - 0.1, self._text_color, 1)

    def draw_instructions(self, frame: np.ndarray, lines: List[str]) -> None:
        if not self.config.show_instructions or not lines:
            return
        s = self.config.font_scale - 0.1
        h, w = frame.shape[:2]
        line_h = int(26 * (s + 0.4))
        box_h = line_h * len(lines) + 22
        box_w = max(cv2.getTextSize(t, _FONT, s, 1)[0][0] for t in lines) + 28
        x, y = 12, h - box_h - 12
        _rounded_rect(frame, x, y, box_w, box_h, 12, self._panel, self.config.panel_opacity)
        for i, line in enumerate(lines):
            _text(frame, line, (x + 14, y + 20 + i * line_h), s, self._text_color, 1)

    def draw_toast(self, frame: np.ndarray, message: str) -> None:
        """Centered transient message (e.g. 'Screenshot saved')."""
        h, w = frame.shape[:2]
        s = self.config.font_scale + 0.1
        (tw, th), _ = cv2.getTextSize(message, _FONT, s, 2)
        x = (w - tw) // 2
        _rounded_rect(frame, x - 18, 70, tw + 36, th + 26, 12, self._panel, 0.85)
        _text(frame, message, (x, 70 + th + 6), s, self._accent, 2)

    def draw_empty_catalog(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        msg = "No clothing items found — add transparent PNGs to assets/clothes/"
        s = self.config.font_scale
        (tw, th), _ = cv2.getTextSize(msg, _FONT, s, 1)
        _rounded_rect(frame, (w - tw) // 2 - 18, h // 2 - th, tw + 36, th + 32,
                      12, (40, 40, 160), 0.85)
        _text(frame, msg, ((w - tw) // 2, h // 2 + th // 2 + 6), s, (255, 255, 255), 1)
