"""Download the ONNX models used by the optional vision upgrades.

Both models are auto-fetched on first use by their backends, but on slow
connections you may prefer to fetch them once, explicitly::

    python tools/fetch_models.py              # pose + garment detector
    python tools/fetch_models.py --only pose  # just the pose model

- models/rtmo-s.onnx            (~40 MB)  pose backend, Apache-2.0
- models/yolos-fashionpedia.onnx (~123 MB) garment object detection
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

DOWNLOADS = {
    "pose": ("rtmo-s.onnx",
             "https://huggingface.co/Xenova/RTMO-s/resolve/main/onnx/model.onnx"),
    "detector": ("yolos-fashionpedia.onnx",
                 "https://huggingface.co/onnx-community/yolos-fashionpedia-ONNX"
                 "/resolve/main/onnx/model.onnx"),
}


def fetch(key: str) -> bool:
    name, url = DOWNLOADS[key]
    dest = MODELS_DIR / name
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"[skip] {name} already present ({dest.stat().st_size // (1 << 20)} MB)")
        return True
    MODELS_DIR.mkdir(exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {name} <- {url}")
    started = time.time()
    try:
        req = Request(url, headers={"User-Agent": "mirrorfit/1.0"})
        with urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"[fail] {exc}")
        return False
    if tmp.stat().st_size < 1_000_000:
        tmp.unlink(missing_ok=True)
        print("[fail] truncated download")
        return False
    tmp.replace(dest)
    print(f"[done] {name} in {time.time() - started:.0f}s "
          f"({dest.stat().st_size // (1 << 20)} MB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["pose", "detector"], default=None)
    args = parser.parse_args()
    keys = [args.only] if args.only else list(DOWNLOADS)
    ok = all(fetch(k) for k in keys)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
