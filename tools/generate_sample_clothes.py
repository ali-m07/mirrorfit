"""Generate demo garment PNGs (transparent BGRA) so the platform runs out-of-the-box.

Brands will replace these with photographed/rendered garments — see the
"Preparing Clothing Assets" section in README.md. These procedural samples
exist so every feature (switching, scaling, shadow, occlusion) is testable
immediately after install.

Run::

    python tools/generate_sample_clothes.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "clothes"

Color = Tuple[int, int, int]  # BGR


def _canvas(w: int = 700, h: int = 800) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _rounded(body: np.ndarray, tl: Tuple[int, int], br: Tuple[int, int],
             color: Color, radius: int = 60) -> None:
    x1, y1 = tl
    x2, y2 = br
    cv2.rectangle(body, (x1 + radius, y1), (x2 - radius, y2), (*color, 255), -1)
    cv2.rectangle(body, (x1, y1 + radius), (x2, y2 - radius), (*color, 255), -1)
    for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(body, (cx, cy), radius, (*color, 255), -1)


def _add_sleeves(img: np.ndarray, color: Color, long: bool = False) -> None:
    # Sleeves angled down-and-out from the shoulders.
    sleeve_len = 260 if long else 150
    for sign in (-1, 1):
        shoulder = (350 + sign * 210, 210)
        cuff = (350 + sign * (210 + sleeve_len // 2), 210 + sleeve_len)
        width = 120 if long else 130
        cv2.line(img, shoulder, cuff, (*color, 255), width, cv2.LINE_AA)
        cv2.circle(img, cuff, width // 2, (*color, 255), -1, cv2.LINE_AA)


def _neckline(img: np.ndarray, style: str = "crew") -> None:
    # Cut the neckline out of the alpha channel.
    if style == "crew":
        cv2.ellipse(img, (350, 165), (95, 55), 0, 0, 360, (0, 0, 0, 0), -1, cv2.LINE_AA)
    else:  # v-neck
        pts = np.array([[255, 150], [445, 150], [350, 280]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (0, 0, 0, 0), cv2.LINE_AA)


def _shading(img: np.ndarray) -> None:
    # Cheap fabric depth: darker edges + soft vertical highlight.
    alpha = img[:, :, 3]
    edges = cv2.Canny(alpha, 50, 150)
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8))
    dark = img[:, :, :3].astype(np.float32) * 0.78
    mask = edges > 0
    img[:, :, :3][mask] = dark[mask].astype(np.uint8)
    h, w = alpha.shape
    highlight = np.linspace(-14, 14, w, dtype=np.float32)[None, :]
    bgr = img[:, :, :3].astype(np.float32)
    bgr += highlight[:, :, None]
    img[:, :, :3] = np.clip(bgr, 0, 255).astype(np.uint8)
    img[:, :, 3] = alpha


def _text(img: np.ndarray, text: str, y: int, scale: float = 1.4,
          color: Color = (245, 245, 245)) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
    cv2.putText(img, text, ((img.shape[1] - tw) // 2, y),
                cv2.FONT_HERSHEY_DUPLEX, scale, (*color, 255), 2, cv2.LINE_AA)


def _hem(img: np.ndarray, color: Color) -> None:
    cv2.line(img, (160, 740), (540, 740), tuple(int(c * 0.7) for c in color) + (255,),
             10, cv2.LINE_AA)


def tshirt(name: str, color: Color, label: str) -> np.ndarray:
    img = _canvas()
    _add_sleeves(img, color, long=False)
    _rounded(img, (160, 150), (540, 745), color)
    _neckline(img, "crew")
    _text(img, label, 470)
    _hem(img, color)
    _shading(img)
    return img


def hoodie(name: str, color: Color, label: str) -> np.ndarray:
    img = _canvas(h=850)
    _add_sleeves(img, color, long=True)
    _rounded(img, (150, 160), (550, 790), color, radius=80)
    # Hood
    cv2.ellipse(img, (350, 170), (130, 90), 0, 180, 360, tuple(int(c * 0.85) for c in color) + (255,), -1, cv2.LINE_AA)
    _neckline(img, "crew")
    # Kangaroo pocket
    cv2.rectangle(img, (230, 600), (470, 740), tuple(int(c * 0.88) for c in color) + (255,), -1)
    cv2.line(img, (230, 600), (470, 600), tuple(int(c * 0.6) for c in color) + (255,), 6, cv2.LINE_AA)
    _text(img, label, 420)
    _hem(img, color)
    _shading(img)
    return img


def jacket(name: str, color: Color, label: str) -> np.ndarray:
    img = _canvas(h=850)
    _add_sleeves(img, color, long=True)
    _rounded(img, (150, 160), (550, 800), color, radius=40)
    # Open front: split the body with a zipper gap.
    cv2.rectangle(img, (330, 150), (370, 800), (0, 0, 0, 0), -1)
    cv2.line(img, (330, 160), (330, 795), (200, 200, 200, 255), 5, cv2.LINE_AA)
    cv2.line(img, (370, 160), (370, 795), (200, 200, 200, 255), 5, cv2.LINE_AA)
    # Collar
    pts_l = np.array([[330, 150], [230, 190], [330, 260]], dtype=np.int32)
    pts_r = np.array([[370, 150], [470, 190], [370, 260]], dtype=np.int32)
    collar = tuple(int(c * 0.75) for c in color) + (255,)
    cv2.fillPoly(img, [pts_l], collar, cv2.LINE_AA)
    cv2.fillPoly(img, [pts_r], collar, cv2.LINE_AA)
    _text(img, label, 430, scale=1.1)
    _shading(img)
    return img


def dress(name: str, color: Color, label: str) -> np.ndarray:
    img = _canvas(h=1100)
    # Fitted bodice
    _rounded(img, (190, 150), (510, 520), color, radius=70)
    _neckline(img, "v")
    # Flared skirt
    skirt = np.array([[200, 500], [500, 500], [640, 1040], [60, 1040]], dtype=np.int32)
    cv2.fillPoly(img, [skirt], (*color, 255), cv2.LINE_AA)
    # Waist seam
    cv2.line(img, (205, 505), (495, 505), tuple(int(c * 0.7) for c in color) + (255,), 8, cv2.LINE_AA)
    _text(img, label, 400, scale=1.1)
    _shading(img)
    return img


def polo(name: str, color: Color, label: str) -> np.ndarray:
    img = _canvas()
    _add_sleeves(img, color, long=False)
    _rounded(img, (165, 150), (535, 745), color, radius=45)
    _neckline(img, "v")
    # Placket + buttons
    cv2.rectangle(img, (335, 220), (365, 330), tuple(int(c * 0.85) for c in color) + (255,), -1)
    for y in (245, 285, 320):
        cv2.circle(img, (350, y), 7, (230, 230, 230, 255), -1, cv2.LINE_AA)
    # Collar
    collar = tuple(int(c * 0.8) for c in color) + (255,)
    cv2.fillPoly(img, [np.array([[255, 150], [350, 230], [250, 260]], np.int32)], collar, cv2.LINE_AA)
    cv2.fillPoly(img, [np.array([[445, 150], [350, 230], [450, 260]], np.int32)], collar, cv2.LINE_AA)
    _text(img, label, 520, scale=1.0)
    _hem(img, color)
    _shading(img)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = [
        ("classic_tee.png", "Classic Tee — Crimson", "upper",
         tshirt("classic_tee", (45, 60, 200), "AR STUDIO")),
        ("navy_hoodie.png", "Everyday Hoodie — Navy", "upper",
         hoodie("navy_hoodie", (130, 80, 40), "TRY•ON")),
        ("denim_jacket.png", "Trucker Jacket — Indigo", "jacket",
         jacket("denim_jacket", (150, 105, 55), "DENIM")),
        ("summer_dress.png", "Flare Dress — Coral", "dress",
         dress("summer_dress", (95, 120, 235), "")),
        ("forest_polo.png", "Pique Polo — Forest", "upper",
         polo("forest_polo", (70, 120, 60), "POLO")),
    ]
    catalog: List[dict] = []
    for filename, label, category, image in items:
        path = OUT / filename
        cv2.imwrite(str(path), image)
        catalog.append({
            "filename": filename,
            "name": label,
            "category": category,
            "anchor_top": 0.0,
            "anchor_width": 1.0,
            "description": f"Procedurally generated demo garment ({category}).",
        })
        print(f"wrote {path}")

    with (OUT / "catalog.json").open("w", encoding="utf-8") as fh:
        json.dump({"items": catalog}, fh, indent=2)
    print(f"wrote {OUT / 'catalog.json'} ({len(catalog)} items)")


if __name__ == "__main__":
    main()
