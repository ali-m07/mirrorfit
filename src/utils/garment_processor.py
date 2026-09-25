"""Turn ANY garment photo into a try-on-ready transparent cutout.

Given an arbitrary product photo (JPG / PNG / WebP, any background) this
module produces a clean BGRA cutout plus auto-estimated anchors, ready to
drop into the try-on catalog:

    photo -> background removal -> alpha cleanup -> tight crop -> anchor fit

Background-removal strategy (``method="auto"``):

1. Images that already carry a usable alpha channel are kept as-is.
2. When ``rembg`` is installed it is used for matting (best quality; install
   with ``pip install rembg --no-deps`` so it reuses the project's ONNX
   Runtime instead of downgrading it to the CPU build).
3. Otherwise an OpenCV GrabCut heuristic runs (no extra dependencies):
   frame borders are seeded as sure-background, the centre as probable
   foreground, and GrabCut refines the boundary by colour models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from src.utils.anchor_fit import fit_anchors_from_mask
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class GarmentCutout:
    """A processed garment: BGRA pixels plus fitting metadata."""

    bgra: np.ndarray
    anchor_top: float          # detected shoulder-line fraction (fit from mask)
    anchor_width: float        # detected shoulder-span fraction (fit from mask)
    method: str                # passthrough | rembg | grabcut
    category: Optional[str] = None   # detected category (upper/jacket/dress/long)
    label: Optional[str] = None      # raw detector class name ("shirt", "dress", ...)
    detection_score: float = 0.0     # detector confidence (0 = no detection)


def decode_image(data: bytes) -> Optional[np.ndarray]:
    """Decode raw bytes into a BGR/BGRA array, or None when undecodable."""
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def _has_usable_alpha(image: np.ndarray) -> bool:
    """True when the image is RGBA with a meaningful amount of transparency."""
    return (image.ndim == 3 and image.shape[2] == 4
            and float((image[:, :, 3] > 8).mean()) >= 0.02
            and float((image[:, :, 3] < 247).mean()) >= 0.02)


def _rembg_alpha(bgr: np.ndarray) -> Optional[np.ndarray]:
    """U2-Net matting via rembg (optional dependency). None when unavailable."""
    try:
        from rembg import remove  # noqa: PLC0415 — optional, imported lazily
        from PIL import Image
    except ImportError:
        return None
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    out = remove(pil)
    rgba = np.array(out.convert("RGBA"))
    return rgba[:, :, 3]


def _grabcut_alpha(bgr: np.ndarray) -> np.ndarray:
    """GrabCut matting: borders = sure background, centre = probable garment."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    pad_x, pad_y = max(2, int(w * 0.07)), max(2, int(h * 0.07))
    mask[pad_y:max(pad_y + 1, h - pad_y), pad_x:max(pad_x + 1, w - pad_x)] = cv2.GC_PR_FGD
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(rgb, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def _keep_main_components(alpha: np.ndarray, keep_ratio: float = 0.25) -> np.ndarray:
    """Drop disconnected specks; keep the largest blob and sizable satellites."""
    binary = (alpha > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 2:
        return alpha
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    keep = np.zeros(count, dtype=bool)
    keep[1:] = areas >= largest * keep_ratio
    keep[1 + int(areas.argmax())] = True
    return np.where(keep[labels], alpha, 0).astype(np.uint8)


def _clean_alpha(alpha: np.ndarray) -> np.ndarray:
    """Threshold, de-speckle and feather a raw matting mask."""
    binary = (alpha > 127).astype(np.uint8) * 255
    binary = _keep_main_components(binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # Feather the edge so the composite doesn't show a hard cutout rim.
    soft = cv2.GaussianBlur(binary, (5, 5), 0)
    # Keep the original hard core, soften only near the boundary.
    core = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return np.maximum(soft, core)


def _crop_to_content(bgra: np.ndarray, margin: int = 6) -> np.ndarray:
    """Crop to the opaque bounding box with a small transparent margin."""
    ys, xs = np.nonzero(bgra[:, :, 3] > 8)
    if len(xs) == 0:
        return bgra
    h, w = bgra.shape[:2]
    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)
    x1 = min(int(xs.max()) + 1 + margin, w)
    y1 = min(int(ys.max()) + 1 + margin, h)
    return bgra[y0:y1, x0:x1].copy()


def _estimate_anchor_width(bgra: np.ndarray) -> float:
    """Deprecated band-based estimate — kept for direct callers/tests."""
    from src.utils.anchor_fit import fit_anchors_from_mask
    return fit_anchors_from_mask(bgra).anchor_width


def process_garment(image: np.ndarray, method: str = "auto",
                    max_side: int = 1200,
                    detect: bool = True) -> GarmentCutout:
    """Process one decoded image into a catalog-ready BGRA cutout.

    Detection-driven pipeline (no manual numbers):

    1. **Object detection** — YOLOS-FashionPedia finds the garment in the
       photo and classifies it (shirt / jacket / dress / ...). The crop is
       the detected box, not a fixed percentage of the frame.
    2. **Matting** — rembg (when installed) or GrabCut removes the
       background inside the detected region.
    3. **Anchor fitting** — the garment's own silhouette determines the
       shoulder line and neckline (see ``anchor_fit``), so the try-on
       renderer gets its anchors without any human input.

    Unsupported detections (trousers/skirts — the renderer is upper-body)
    raise ``ValueError`` with an explicit reason.
    """
    if image is None or image.ndim < 2:
        raise ValueError("Garment image is empty or undecodable")
    # Already-transparent assets pass straight through (after crop/cleanup).
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    category: Optional[str] = None
    label: Optional[str] = None
    det_score = 0.0
    if detect and image.ndim == 3 and image.shape[2] == 3:
        try:
            from src.vision.garment_detector import GarmentDetector
            # auto_download=False: detection upgrades when the model is
            # fetched (tools/fetch_models.py) but never stalls an upload.
            detector = GarmentDetector(auto_download=False)
            detection = detector.detect(image)
            detector.close()   # release the ORT session promptly
        except Exception as exc:                       # detector is optional
            logger.warning("Garment detection skipped: %s", exc)
            detection = None
        if detection is not None:
            det_score, label, category = detection.score, detection.label, detection.category
            if not detection.supported:
                raise ValueError(
                    f"Detected a {detection.label!r} — lower-body garments are not "
                    "supported by the renderer yet (upper body / dress only)")
            x0, y0, x1, y1 = detection.box
            mx = int((x1 - x0) * 0.06)                 # small margin for matting
            my = int((y1 - y0) * 0.06)
            image = image[max(0, y0 - my):min(image.shape[0], y1 + my),
                          max(0, x0 - mx):min(image.shape[1], x1 + mx)]
            image = image.copy()
    if image.ndim == 3 and image.shape[2] == 4 and method in {"auto", "passthrough"}:
        if _has_usable_alpha(image):
            cut = _crop_to_content(image)
            fit = fit_anchors_from_mask(cut)
            return GarmentCutout(bgra=cut, anchor_top=fit.anchor_top,
                                 anchor_width=fit.anchor_width,
                                 method="passthrough",
                                 category=category, label=label,
                                 detection_score=det_score)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    alpha: Optional[np.ndarray] = None
    used = "grabcut"
    if method in {"auto", "rembg"}:
        alpha = _rembg_alpha(image)
        used = "rembg"
    if alpha is None:
        alpha = _grabcut_alpha(image)
        used = "grabcut"
    alpha = _clean_alpha(alpha)
    if float((alpha > 8).mean()) < 0.01:
        raise ValueError("Garment segmentation produced an empty mask — "
                         "try a photo with a plain background or install rembg")

    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    bgra = _crop_to_content(bgra)
    fit = fit_anchors_from_mask(bgra)
    return GarmentCutout(bgra=bgra, anchor_top=fit.anchor_top,
                         anchor_width=fit.anchor_width,
                         method=used, category=category, label=label,
                         detection_score=det_score)


def process_garment_bytes(data: bytes, method: str = "auto",
                          max_side: int = 1200) -> GarmentCutout:
    """Convenience wrapper for uploaded files (raw bytes in, cutout out)."""
    image = decode_image(data)
    if image is None:
        raise ValueError("Could not decode the uploaded image")
    return process_garment(image, method=method, max_side=max_side)


def update_catalog_json(catalog_path: Path, filename: str,
                        meta: Dict[str, object]) -> None:
    """Insert/replace one garment entry in catalog.json (other entries kept)."""
    import json

    entries: list = []
    if catalog_path.is_file():
        try:
            raw = json.loads(catalog_path.read_text(encoding="utf-8"))
            entries = raw.get("items", []) if isinstance(raw, dict) else raw
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s (%s); starting a fresh catalog",
                           catalog_path, exc)
            entries = []
    entries = [e for e in entries
               if isinstance(e, dict) and e.get("filename") != filename]
    entry = {"filename": filename, **meta}
    entries.append(entry)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps({"items": entries}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
