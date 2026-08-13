"""Shared helpers: clothing catalog, asset loading, filesystem utilities."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Clothing assets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClothingItem:
    """A single garment in the catalog.

    ``anchor_top`` / ``anchor_width`` describe where the garment expects the
    neck point and shoulder span to land, as fractions of the image size.
    This decouples renderer behaviour from how each PNG was drawn, so assets
    from different artists still fit consistently.
    """

    id: str
    name: str
    filename: str
    path: Path
    category: str = "upper"            # upper | dress | jacket | ...
    anchor_top: float = 0.0            # y-fraction where the neckline sits
    anchor_width: float = 1.0          # x-fraction of the shoulder span
    description: str = ""

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "item"


class ClothingCatalog:
    """Discovers and lazily loads transparent PNG garments."""

    SUPPORTED_EXTENSIONS = {".png", ".obj"}

    def __init__(self, directory: Path, catalog_file: str = "catalog.json",
                 extra_dirs: Optional[List[Path]] = None) -> None:
        self.directory = directory
        self.catalog_path = directory / catalog_file
        self.extra_dirs = extra_dirs or []
        self._items: List[ClothingItem] = []
        self._cache: Dict[str, np.ndarray] = {}
        self.reload()

    # -- discovery ---------------------------------------------------------

    def reload(self) -> None:
        """Rescan the clothes directory (and catalog.json metadata)."""
        self._items.clear()
        self._cache.clear()
        metadata = self._load_metadata()
        search_dirs = [self.directory, *self.extra_dirs]
        for base in search_dirs:
            if not base.is_dir():
                logger.warning("Clothes directory missing, creating: %s", base)
                base.mkdir(parents=True, exist_ok=True)
                continue
            for asset in sorted(p for p in base.iterdir()
                               if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS):
                meta = metadata.get(asset.stem, {})
                item = ClothingItem(
                    id=str(meta.get("id") or _slugify(asset.stem)),
                    name=str(meta.get("name") or asset.stem.replace("_", " ").title()),
                    filename=asset.name,
                    path=asset,
                    category=str(meta.get("category", "upper")),
                    anchor_top=float(meta.get("anchor_top", 0.0)),
                    anchor_width=float(meta.get("anchor_width", 1.0)),
                    description=str(meta.get("description", "")),
                )
                if any(existing.id == item.id for existing in self._items):
                    continue
                self._items.append(item)
        logger.info("Clothing catalog loaded: %d item(s)", len(self._items))

    def _load_metadata(self) -> Dict[str, Dict[str, object]]:
        if not self.catalog_path.is_file():
            return {}
        try:
            with self.catalog_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            entries = raw.get("items", raw if isinstance(raw, list) else [])
            result: Dict[str, Dict[str, object]] = {}
            for entry in entries:
                if isinstance(entry, dict) and "filename" in entry:
                    result[Path(str(entry["filename"])).stem] = entry
            return result
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to parse %s: %s", self.catalog_path, exc)
            return {}

    # -- access ------------------------------------------------------------

    @property
    def items(self) -> List[ClothingItem]:
        return list(self._items)

    def __len__(self) -> int:  # noqa: D105
        return len(self._items)

    def get(self, index: int) -> ClothingItem:
        if not self._items:
            raise IndexError("Clothing catalog is empty — add PNGs to assets/clothes/")
        return self._items[index % len(self._items)]

    def find(self, item_id: str) -> Optional[ClothingItem]:
        for item in self._items:
            if item.id == item_id or item.filename == item_id:
                return item
        return None

    def load_image(self, item: ClothingItem) -> np.ndarray:
        """Load the BGRA garment image, caching by item id."""
        if item.id in self._cache:
            return self._cache[item.id]
        image = cv2.imread(str(item.path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read garment image: {item.path}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif image.shape[2] != 4:
            raise ValueError(f"Unsupported garment channels ({image.shape[2]}): {item.path}")
        self._cache[item.id] = image
        return image

    def load_mesh(self, item: ClothingItem):
        """Load a Wavefront OBJ as ``(vertices, triangular_faces)``."""
        if item.path.suffix.lower() != ".obj":
            raise ValueError(f"Not an OBJ garment: {item.path}")
        vertices, faces = [], []
        with item.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if not parts: continue
                if parts[0] == "v" and len(parts) >= 4:
                    vertices.append(tuple(float(v) for v in parts[1:4]))
                elif parts[0] == "f" and len(parts) >= 4:
                    idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    for i in range(1, len(idx) - 1):
                        faces.append((idx[0], idx[i], idx[i + 1]))
        if not vertices or not faces:
            raise ValueError(f"OBJ has no renderable geometry: {item.path}")
        return np.asarray(vertices, dtype=np.float32), faces


# ---------------------------------------------------------------------------
# Alpha compositing
# ---------------------------------------------------------------------------

def alpha_blend(background: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend a BGRA ``overlay`` onto BGR ``background`` in place at (x, y).

    Handles all four edges with clipping — the overlay may be partially or
    fully outside the frame, in which case this is a safe no-op.
    """
    if overlay is None or overlay.size == 0:
        return
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    # Intersection of overlay rect with the frame rect.
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, bw), min(y + oh, bh)
    if x2 <= x1 or y2 <= y1:
        return

    ox1, oy1 = x1 - x, y1 - y
    roi = background[y1:y2, x1:x2]
    patch = overlay[oy1:oy1 + (y2 - y1), ox1:ox1 + (x2 - x1)]

    alpha = (patch[:, :, 3:4].astype(np.float32)) / 255.0
    rgb = patch[:, :, :3].astype(np.float32)
    roi[:] = (rgb * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------

class FPSCounter:
    """Smoothed FPS estimator (exponential moving average of frame times)."""

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._fps: Optional[float] = None
        self._last: Optional[float] = None

    def tick(self) -> float:
        """Record a frame and return the smoothed FPS."""
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 1e-6:
                instant = 1.0 / dt
                self._fps = instant if self._fps is None else (
                    self._alpha * instant + (1 - self._alpha) * self._fps)
        self._last = now
        return self._fps or 0.0

    @property
    def fps(self) -> float:
        return self._fps or 0.0


def timestamped_filename(prefix: str, suffix: str, ext: str, extra: str = "") -> str:
    """e.g. tryon_20260806_153012_navy-hoodie.png — safe for all filesystems."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [prefix, stamp, _slugify(extra)] if extra else [prefix, stamp]
    if suffix:
        parts.append(suffix)
    return "_".join(parts) + f".{ext.lstrip('.')}"


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode a BGR frame as JPEG bytes (for MJPEG / WebSocket streaming)."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def letterbox(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize ``frame`` to ``size`` preserving aspect (pillar/letterbox)."""
    tw, th = size
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x, y = (tw - nw) // 2, (th - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas
