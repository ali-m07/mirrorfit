"""GPU-capable pose backend: RTMO one-stage model through ONNX Runtime.

Runs the Apache-2.0 RTMO-s whole-body pose model with ONNX Runtime, choosing
the fastest available execution provider:

- **DirectML** (``DmlExecutionProvider``) — any DirectX-12 GPU: Intel Iris/Arc,
  AMD, NVIDIA. This is the "vga" path on Windows.
- **CPU** (``CPUExecutionProvider``) — fallback when no usable GPU exists.

The caller (``PoseEstimator``) owns the instance and drives it from a single
thread, exactly like the MediaPipe graph. Keypoints are returned already
mapped onto the MediaPipe landmark indices the rest of the pipeline uses, so
the renderer is backend-agnostic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import Request, urlopen

import cv2
import numpy as np

from src.utils.config_loader import OnnxPoseSection
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_URL = "https://huggingface.co/Xenova/RTMO-s/resolve/main/onnx/model.onnx"

# RTMO emits COCO-17 keypoints; the pipeline speaks MediaPipe indices.
# Only the landmarks the try-on pipeline consumes are mapped.
_COCO_TO_MP = {
    0: 0,    # nose
    3: 7,    # left ear
    4: 8,    # right ear
    5: 11,   # left shoulder
    6: 12,   # right shoulder
    7: 13,   # left elbow
    8: 14,   # right elbow
    9: 15,   # left wrist
    10: 16,  # right wrist
    11: 23,  # left hip
    12: 24,  # right hip
}


class OnnxPoseBackend:
    """Single-owner RTMO ONNX session with provider auto-selection."""

    def __init__(self, config: OnnxPoseSection) -> None:
        self.config = config
        self._session = None
        self._input_name = "input"
        self._input_size = int(config.input_size)
        self.backend_name = "onnx"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_session(self) -> None:
        """Create the ORT session (idempotent). Raises when unavailable."""
        if self._session is not None:
            return
        import onnxruntime as ort

        model_path = Path(self.config.model_path)
        if not model_path.is_absolute():
            from src.utils.config_loader import PROJECT_ROOT
            model_path = PROJECT_ROOT / model_path
        if not model_path.is_file():
            if not self.config.auto_download:
                raise RuntimeError(f"ONNX pose model missing: {model_path}")
            self._download(model_path)

        providers = self._select_providers(ort)
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        if "DmlExecutionProvider" in self._session.get_providers() and not self._sane_outputs():
            # Some DirectX-12 drivers (observed on Intel iGPU 31.0.101.x) run
            # this graph through DirectML with silently wrong outputs. Detect
            # that once at startup with a blank frame and fall back to CPU.
            logger.warning("DirectML outputs failed the sanity check on this "
                           "GPU/driver — falling back to CPUExecutionProvider")
            self._session = ort.InferenceSession(str(model_path),
                                                 providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        # Match the actual graph input when it is fixed (NCHW: H/W are the
        # last two dims; this export = 640x640).
        shape = self._session.get_inputs()[0].shape
        if len(shape) == 4:
            fixed = [d for d in shape[2:4] if isinstance(d, int) and d > 0]
            if len(fixed) == 2 and fixed[0] == fixed[1]:
                self._input_size = fixed[0]
        used = self._session.get_providers()
        self.backend_name = "onnx-dml" if "DmlExecutionProvider" in used else "onnx-cpu"
        logger.info("ONNX pose backend ready (model=%s, providers=%s)",
                    model_path.name, used)

    def _sane_outputs(self) -> bool:
        """True when the model finds no person on a featureless frame."""
        size = self._input_size
        blob = np.full((1, 3, size, size), 114, dtype=np.float32)
        try:
            outputs = self._session.run(None, {self._input_name: blob})
        except Exception:
            return False
        dets, _ = self._unpack_outputs(outputs)
        return dets is None or float(dets[:, 4].max()) < 0.5

    def _select_providers(self, ort) -> list:
        """Execution-provider preference list from the configured device."""
        available = ort.get_available_providers()
        device = str(self.config.device).lower()
        if device == "cpu":
            return ["CPUExecutionProvider"]
        # auto / dml — prefer a discrete GPU (CUDA where the build offers it;
        # DirectML covers other DirectX-12 GPUs), CPU as the safety net.
        for gpu in ("CUDAExecutionProvider", "DmlExecutionProvider"):
            if gpu in available:
                return [gpu, "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _download(self, dest: Path) -> None:
        """Fetch the pose model once (idempotent, resumable by re-run)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        logger.info("Downloading pose model (~40 MB) to %s ...", dest)
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
            raise RuntimeError("Downloaded model looks truncated")
        tmp.replace(dest)
        logger.info("Pose model ready in %.1fs", time.time() - started)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(self, frame_bgr: np.ndarray) -> Optional[Dict[int, Tuple[float, float, float]]]:
        """Run the model on one BGR frame.

        Returns raw (unsmoothed) landmarks as ``{mp_index: (x_px, y_px,
        visibility)}`` for the most confident person, or None.
        """
        if self._session is None:
            self.ensure_session()
        h, w = frame_bgr.shape[:2]
        size = self._input_size

        # Preprocess: RGB, stretch-resize to the square input, raw 0..255
        # (this export has normalisation baked into the graph).
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        blob = resized.transpose(2, 0, 1)[None].astype(np.float32, copy=False)

        outputs = self._session.run(None, {self._input_name: blob})
        dets, keypoints = self._unpack_outputs(outputs)

        # Single-user mirror: keep the most confident detection.
        if dets is None or len(dets) == 0:
            return None
        best = int(np.argmax(dets[:, 4]))
        if float(dets[best, 4]) < float(self.config.person_threshold):
            return None
        kpts = keypoints[best]  # (17, 3) in resized-input pixel coords

        sx, sy = w / float(size), h / float(size)
        landmarks: Dict[int, Tuple[float, float, float]] = {}
        for coco_idx, mp_idx in _COCO_TO_MP.items():
            x, y, score = kpts[coco_idx]
            landmarks[mp_idx] = (float(x) * sx, float(y) * sy, float(score))
        return landmarks

    @staticmethod
    def _unpack_outputs(outputs) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Map the session's outputs to (dets[N,5], keypoints[N,17,3])."""
        dets = keypoints = None
        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim == 3 and arr.shape[-1] == 5:
                dets = arr[0]
            elif arr.ndim == 4 and arr.shape[-1] == 3:
                keypoints = arr[0]
            elif arr.ndim == 3 and arr.shape[-1] == 3 and dets is not None and keypoints is None:
                keypoints = arr[0]
        return dets, keypoints

    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._session is not None:
            # ORT sessions have no explicit close; dropping the reference
            # releases the GPU/CPU resources.
            self._session = None
