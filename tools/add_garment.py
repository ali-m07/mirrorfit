"""Add any garment photo to the live try-on catalog.

Works with ordinary product photos (JPG/PNG/WebP, any background): the image
is segmented, cleaned, cropped and its anchors auto-estimated, then it lands
in ``assets/clothes/`` + ``catalog.json`` — instantly available to the running
engine (no restart needed).

Usage::

    python tools/add_garment.py path/to/hoodie.jpg --name "Navy Hoodie"
    python tools/add_garment.py dress.png --category dress --id summer-dress

Optional quality boost: ``pip install rembg --no-deps`` — the U2-Net matting
is used automatically when available (better on busy backgrounds).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from src.utils.config_loader import load_config  # noqa: E402
from src.utils.garment_processor import (  # noqa: E402
    SUPPORTED_IMAGE_SUFFIXES, process_garment, update_catalog_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, help="Path to the garment photo")
    parser.add_argument("--name", default=None, help="Display name (default: filename)")
    parser.add_argument("--id", dest="item_id", default=None,
                        help="Catalog id (default: slugified filename stem)")
    parser.add_argument("--category", default="auto",
                        choices=["auto", "upper", "dress", "long", "jacket"],
                        help="auto = use the detected category (object detection)")
    parser.add_argument("--description", default="")
    parser.add_argument("--method", default="auto", choices=["auto", "grabcut", "rembg"],
                        help="Background-removal method (default: best available)")
    parser.add_argument("--anchor-width", type=float, default=None,
                        help="Override the auto-estimated shoulder-span fraction")
    args = parser.parse_args()

    if args.image.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        parser.error(f"Unsupported image type: {args.image.suffix}")
    if not args.image.is_file():
        parser.error(f"File not found: {args.image}")

    image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Could not decode {args.image}")
        return 2

    print("Processing garment (detect -> matte -> crop -> anchor fit)...")
    cutout = process_garment(image, method=args.method)
    if args.anchor_width is not None:
        cutout.anchor_width = args.anchor_width

    cfg = load_config()
    clothes_dir = cfg.resolve_path(cfg.clothes.directory)
    clothes_dir.mkdir(parents=True, exist_ok=True)
    item_id = args.item_id or args.image.stem.lower().replace(" ", "-")
    target = clothes_dir / f"{item_id}.png"
    cv2.imwrite(str(target), cutout.bgra)

    category = (args.category if args.category != "auto" else None) \
        or cutout.category or "upper"
    meta = {
        "id": item_id,
        "name": args.name or args.image.stem.replace("_", " ").title(),
        "category": category,
        "description": args.description or (
            f"auto-detected: {cutout.label}" if cutout.label else ""),
        "anchor_top": round(cutout.anchor_top, 3),
        "anchor_width": round(cutout.anchor_width, 3),
        "processed_by": cutout.method,
        "detected": cutout.label or "none",
        "detection_score": round(cutout.detection_score, 3),
    }
    catalog_path = clothes_dir / cfg.clothes.catalog_file
    update_catalog_json(catalog_path, target.name, meta)

    print(f"Saved    : {target}")
    print(f"Detected : {cutout.label or 'nothing'} "
          f"(score {cutout.detection_score:.2f}) -> category: {category}")
    print(f"Method   : {cutout.method}   size: {cutout.bgra.shape[1]}x{cutout.bgra.shape[0]}")
    print(f"Anchors  : anchor_top={cutout.anchor_top:.3f}  "
          f"anchor_width={cutout.anchor_width:.3f} (fit from silhouette)")
    print("Done - call POST /clothes/reload (or restart the app) to pick it up live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
