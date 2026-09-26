# tools/e2e/analyze_fit.py — ASCII visualiser + mask stats for fit screenshots.
#
# The harness saves per-run:
#   <run>_full.png        whole browser page
#   <run>_video.png       crop of the mirrored video canvas (person, no garment)
#   <run>_garment.png     crop of the 3D garment canvas (transparent bg)
#
# This script renders each as an ASCII pixel-classification map so a text-only
# agent can genuinely LOOK at the frames, plus prints quantitative extents
# (garment bbox vs skin bbox) used to judge fit criteria.
#
# Classes: b=dark bg, k=skin, h=hair/dark-features, w=bright white,
#          r/g/b/y/m/c=saturated hue, .=mid grey
import sys

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def classify(img):
    """img: HxWx3 BGR -> HxW char codes."""
    b, g, r = img[..., 0].astype(np.int32), img[..., 1].astype(np.int32), img[..., 2].astype(np.int32)
    luma = (0.299 * r + 0.587 * g + 0.114 * b)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn
    out = np.full(img.shape[:2], ".", dtype="<U1")

    is_skin = (r > 110) & (r > g) & (g > b) & ((r - b) > 30) & ((r - b) < 130) & (luma > 55)
    is_hair = (luma < 70) & (~is_skin)
    is_bright = (luma > 200) & (sat < 60)
    is_sat = (sat > 55) & (~is_skin) & (luma > 35)

    out[is_hair] = "h"
    out[is_bright] = "w"
    out[is_skin] = "k"
    # hue letters for saturated pixels (R=0,G=1,B=2 as channel order in numpy after split)
    rr, gg, bb = r, g, b
    hue_max = np.argmax(np.stack([rr, gg, bb], axis=-1), axis=-1)
    letter = np.array(["r", "g", "b"])[hue_max]
    # yellowish: r&g both high vs b
    yellow = (rr > 110) & (gg > 110) & (bb < 100)
    letter = np.where(yellow, "y", letter)
    out[is_sat] = letter[np.where(is_sat)]

    darkbg = (luma < 48) & (~is_skin) & (~is_sat)
    out[darkbg] = " "
    # uppercase hue letters so saturated blue never reads like background
    hue_letter = {"r": "R", "g": "G", "b": "B", "y": "Y", "m": "M", "c": "C"}
    letter = np.array([hue_letter[x] for x in letter.ravel()]).reshape(letter.shape)
    out[is_sat] = letter[np.where(is_sat)]
    return out


def ascii_map(path, cols=100):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"(cannot read {path})")
        return None
    alpha = None
    if img.shape[2] == 4:
        alpha = img[..., 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[:2]
    rows = max(1, round(cols * h / w * 0.5))  # terminal chars are ~2:1
    small = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)
    codes = classify(small)
    if alpha is not None:
        asmall = cv2.resize(alpha, (cols, rows), interpolation=cv2.INTER_AREA)
        codes = np.where(asmall > 40, codes, " ")  # transparent -> blank
    return codes, img.shape[:2]


def mask_stats(img_path):
    """Return stats dicts for skin mask and (if alpha) garment mask."""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    stats = {"h": h, "w": w}
    if img.shape[2] == 4:
        alpha = img[..., 3]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gm = alpha > 40
        ys, xs = np.where(gm)
        if len(ys):
            stats["garment"] = {
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "width_at_bbox_mid": None,
            }
            ymid = (ys.min() + ys.max()) // 2
            row = np.where(gm[ymid])[0]
            if len(row):
                stats["garment"]["width_at_bbox_mid"] = [int(row.min()), int(row.max())]
            # widest row in top 60% (chest/shoulder area)
            top = gm[: int(h * 0.6)]
            widths = top.sum(axis=1)
            yw = int(np.argmax(widths))
            row = np.where(top[yw])[0]
            stats["garment"]["widest_top60"] = {"y": yw, "x0": int(row.min()), "x1": int(row.max()), "w": int(widths[yw])}
    else:
        img_rgb = img
    r, g, b = img_rgb[..., 2].astype(np.int32), img_rgb[..., 1].astype(np.int32), img_rgb[..., 0].astype(np.int32)
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    skin = (r > 110) & (r > g) & (g > b) & ((r - b) > 30) & ((r - b) < 130) & (luma > 55)
    ys, xs = np.where(skin)
    if len(ys):
        stats["skin"] = {"bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], "count": int(skin.sum())}
    return stats


def main():
    paths = sys.argv[1:]
    for p in paths:
        print(f"\n===== {p} =====")
        st = mask_stats(p)
        if st:
            print(f"  size: {st['w']}x{st['h']}")
            if "garment" in st:
                print(f"  garment bbox (x0,y0,x1,y1): {st['garment']['bbox']}  width@mid: {st['garment']['width_at_bbox_mid']}  widest_top60: {st['garment']['widest_top60']}")
            if "skin" in st:
                print(f"  skin bbox: {st['skin']['bbox']}  px: {st['skin']['count']}")
        m = ascii_map(p)
        if m:
            codes, _ = m
            for row in codes:
                print("".join(row))


if __name__ == "__main__":
    main()
