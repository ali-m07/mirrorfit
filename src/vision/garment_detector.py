"""Garment object detection: YOLOS fine-tuned on FashionPedia, via ONNX.

Stage 1 of the ingestion pipeline. Given ANY garment photo this finds the
garment object itself — its bounding box and its category (shirt, jacket,
dress, ...) — so nothing downstream depends on manual numbers:

    photo ──► YOLOS detector ──► (box, category) ──► crop ──► matting ──► fit

The graph is DETR-style: ``logits [1, queries, 46]`` + ``pred_boxes
[1, queries, 4]`` (cxcywh, normalised) — no NMS required, we simply take the
highest-scoring garment-class query. The model (~30 MB) auto-downloads from
HuggingFace on first use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import Request, urlopen

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_URL = ("https://huggingface.co/onnx-community/yolos-fashionpedia-ONNX"
             "/resolve/main/onnx/model.onnx")

# FashionPedia class id -> project garment category.
# Classes not listed here are accessories / garment PARTS (pocket, zipper…)
# and are ignored when picking the dominant garment.
_CLASS_TO_CATEGORY = {
    0: "upper",    # shirt, blouse
    1: "upper",    # top, t-shirt, sweatshirt
    2: "upper",    # sweater
    3: "upper",    # cardigan
    4: "jacket",   # jacket
    5: "upper",    # vest
    6: "lower",    # pants
    7: "lower",    # shorts
    8: "lower",    # skirt
    9: "jacket",   # coat
    10: "dress",   # dress
    11: "long",    # jumpsuit
    12: "long",    # cape
    25: "upper",   # scarf (drapes over the torso)
}

_LABEL_NAMES = {
    0: "shirt", 1: "top", 2: "sweater", 3: "cardigan", 4: "jacket",
    5: "vest", 6: "pants", 7: "shorts", 8: "skirt", 9: "coat",
    10: "dress", 11: "jumpsuit", 12: "cape", 25: "scarf",
}


@dataclass
class GarmentDetection:
    """One detected garment object in a photo."""

    box: Tuple[int, int, int, int]   # x0, y0, x1, y1 in source pixels
    score: float
    class_id: int
    label: str
    category: str                    # project category (upper/jacket/dress/long/lower)

    @property
    def supported(self) -> bool:
        return self.category in {"upper", "jacket", "dress", "long"}


class GarmentDetector:
    """Single-owner YOLOS-FashionPedia ONNX session."""

    def __init__(self, model_path: str = "models/yolos-fashionpedia.onnx",
                 conf_threshold: float = 0.5,
                 auto_download: bool = True) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.auto_download = auto_download
        self._session = None
        self._input_hw: Tuple[int, int] = (512, 864)   # H, W (from config)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_session(self) -> bool:
        """Load the model. Returns False when unavailable (never raises for
        missing optional deps — detection then degrades to mask-only)."""
        if self._session is not None:
            return True
        try:
            import onnxruntime as ort
        except ImportError:
            logger.info("onnxruntime not installed — garment detection disabled")
            return False
        path = (self.model_path if self.model_path.is_absolute()
                else Path(__file__).resolve().parents[2] / self.model_path)
        if not path.is_file():
            if not self.auto_download:
                return False
            try:
                self._download(path)
            except Exception as exc:
                logger.warning("Garment-detector download failed (%s) — "
                               "continuing without detection", exc)
                return False
        self._session = ort.InferenceSession(str(path), sess_options=self._session_options(ort),
                                             providers=["CPUExecutionProvider"])
        shape = self._session.get_inputs()[0].shape
        fixed = [d for d in shape[1:3] if isinstance(d, int) and d > 0]
        if len(fixed) == 2:
            self._input_hw = (fixed[0], fixed[1])
        logger.info("Garment detector ready (yolos-fashionpedia)")
        return True

    @staticmethod
    def _session_options(ort) -> "ort.SessionOptions":
        """Polite threading: ingestion is a one-shot task and must not spin
        CPU cores away from the live render loop while it runs."""
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        return so

    def _download(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        logger.info("Downloading garment detector (~30 MB) to %s ...", dest)
        started = time.time()
        req = Request(MODEL_URL, headers={"User-Agent": "mirrorfit/1.0"})
        with urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        if tmp.stat().st_size < 1_000_000:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("Downloaded detector looks truncated")
        tmp.replace(dest)
        logger.info("Garment detector ready in %.1fs", time.time() - started)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> Optional[GarmentDetection]:
        """Find the dominant garment object in the photo.

        Returns None when nothing garment-like is detected (the caller then
        falls back to whole-image matting).
        """
        if not self.ensure_session():
            return None
        h, w = frame_bgr.shape[:2]
        ih, iw = self._input_hw

        # Preprocess: resize to the fixed input, ImageNet-normalise, NCHW.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - np.array([0.485, 0.456, 0.406], np.float32)) \
             / np.array([0.229, 0.224, 0.225], np.float32)
        blob = blob.transpose(2, 0, 1)[None]

        logits, boxes = self._session.run(None, {self._session.get_inputs()[0].name: blob})
        # Softmax over classes -> per-query (score, class).
        probs = _softmax(logits[0])
        scores = probs.max(axis=1)
        classes = probs.argmax(axis=1)

        best: Optional[Tuple[float, int, int]] = None   # (score, cls, query)
        for q, (score, cls) in enumerate(zip(scores, classes)):
            if cls not in _CLASS_TO_CATEGORY:      # accessories / garment parts
                continue
            if score < self.conf_threshold:
                continue
            if best is None or score > best[0]:
                best = (float(score), int(cls), q)
        if best is None:
            return None

        score, cls, q = best
        cx, cy, bw, bh = (float(v) for v in boxes[0][q])
        # Normalised cxcywh -> source pixels.
        x0 = (cx - bw / 2.0) * w
        y0 = (cy - bh / 2.0) * h
        x1 = (cx + bw / 2.0) * w
        y1 = (cy + bh / 2.0) * h
        box = (int(max(0, x0)), int(max(0, y0)),
               int(min(w, x1)), int(min(h, y1)))
        if box[2] - box[0] < 16 or box[3] - box[1] < 16:
            return None
        return GarmentDetection(box=box, score=score, class_id=cls,
                                label=_LABEL_NAMES.get(cls, f"class{cls}"),
                                category=_CLASS_TO_CATEGORY[cls])

    def close(self) -> None:
        self._session = None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
