# tools/e2e/make_eval_assets.py — evaluation assets for the fit harness.
#
#  - tools/e2e/assets/solid_green_product.png : bright solid green product image
#    uploaded through the retail input so the baked garment reads clearly
#    against the test person (the default red-on-crimson jacket camouflages).
#  - output/e2e/test_subject_armsdown_dark.y4m : the real-person still with a
#    darkened border/backdrop so person-vs-background masks segment cleanly.
#    MediaPipe detection is re-verified by probe_holistic.py after generation.
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_DIR = os.path.join(ROOT, "tools", "e2e", "assets")
OUT_DIR = os.path.join(ROOT, "output", "e2e")
SRC = os.path.join(ROOT, "output", "screenshots", "tryon_20260925_184641_classic-tee-crimson.png")


def make_green_product():
    """Bright green garment shape on a light neutral backdrop.

    The UV bake removes 'background' by corner-sampling plus border flood
    fill, so the image needs a distinct backdrop — a solid full-frame colour
    would flood-fill to fully transparent and bake nothing.
    """
    w, h = 800, 800
    img = np.full((h, w, 3), 235, dtype=np.uint8)  # light neutral backdrop
    green = (40, 215, 40)  # bright green (BGR)
    cv2.ellipse(img, (w // 2, int(h * 0.42)), (190, 240), 0, 0, 360, green, -1)   # body
    cv2.ellipse(img, (w // 2 - 210, int(h * 0.45)), (55, 150), 12, 0, 360, green, -1)  # left sleeve
    cv2.ellipse(img, (w // 2 + 210, int(h * 0.45)), (55, 150), -12, 0, 360, green, -1)  # right sleeve
    cv2.ellipse(img, (w // 2, int(h * 0.24)), (90, 45), 0, 0, 360, (200, 200, 200), -1)  # collar cutout
    img = cv2.GaussianBlur(img, (5, 5), 0)
    path = os.path.join(ASSET_DIR, "solid_green_product.png")
    cv2.imwrite(path, img)
    return path


def darken_backdrop(img):
    """Darken everything outside an ellipse around the person."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, int(h * 0.62)), (int(w * 0.40), int(h * 0.58)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (61, 61), 0).astype(np.float32) / 255.0
    dark = (img.astype(np.float32) * 0.22).astype(np.uint8)
    out = (dark * (1 - mask[..., None]) + img.astype(np.float32) * mask[..., None]).astype(np.uint8)
    return out


def bgr_to_i420_fullrange(bgr):
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = (b - y) * 0.564 + 128.0
    v = (r - y) * 0.713 + 128.0
    hh, ww = y.shape
    u_small = u.reshape(hh // 2, 2, ww // 2, 2).mean(axis=(1, 3))
    v_small = v.reshape(hh // 2, 2, ww // 2, 2).mean(axis=(1, 3))
    return b"".join([
        np.clip(y, 0, 255).astype(np.uint8).tobytes(),
        np.clip(u_small, 0, 255).astype(np.uint8).tobytes(),
        np.clip(v_small, 0, 255).astype(np.uint8).tobytes(),
    ])


def write_y4m(path, base, frames=450, fps=30):
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W640 H480 F{fps}:1 Ip A1:1 C420jpeg\n".encode("ascii"))
        for n in range(frames):
            f.write(b"FRAME\n")
            f.write(bgr_to_i420_fullrange(base))
    print("wrote", path)


def main():
    os.makedirs(ASSET_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(make_green_product())

    still = cv2.imread(SRC)
    assert still is not None, SRC
    dark = darken_backdrop(still)
    cv2.imwrite(os.path.join(OUT_DIR, "preview_armsdown_dark.png"), dark)
    write_y4m(os.path.join(OUT_DIR, "test_subject_armsdown_dark.y4m"), dark)

    # Preferred subject: public-domain COCO sample (standing person, visible
    # hips, anthropometrically consistent landmarks). Crop around the person,
    # letterbox back to 640x480 (the y4m header size must match the frame
    # bytes or Chrome's chroma planes drift), darken the backdrop.
    standing_path = os.path.join(OUT_DIR, "standing_000000000241.jpg")
    standing = cv2.imread(standing_path)
    if standing is not None:
        # source is portrait 480x640: scale to fit 480 high, letterbox to 640 wide
        scale = 480 / standing.shape[0]
        resized = cv2.resize(standing, (int(standing.shape[1] * scale), 480))
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        bx = (640 - resized.shape[1]) // 2
        canvas[:, bx:bx + resized.shape[1]] = resized
        dark_standing = darken_backdrop(canvas)
        cv2.imwrite(os.path.join(OUT_DIR, "preview_standing_dark.png"), dark_standing)
        write_y4m(os.path.join(OUT_DIR, "test_subject_standing_dark.y4m"), dark_standing)


if __name__ == "__main__":
    main()
