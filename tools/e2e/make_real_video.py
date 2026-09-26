# tools/e2e/make_real_video.py — build .y4m clips from a real webcam still.
#
# MediaPipe Holistic does not detect the fully synthetic subject (probe:
# 0 landmarks), so per the harness plan we fall back to a REAL person frame:
# output/screenshots/tryon_20260925_184641_classic-tee-crimson.png (640x480,
# front-facing, arms down, shoulders spanning ~51% of the frame).
#
# Outputs (output/e2e/):
#   test_subject_armsdown.y4m   real still + sensor-like flicker/noise
#   test_subject_armsspread.y4m same still with skin-toned horizontal arms
#                               drawn from the detected shoulder joints
#   preview_*.png               stills for ASCII inspection
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "output", "screenshots", "tryon_20260925_184641_classic-tee-crimson.png")
OUT_DIR = os.path.join(ROOT, "output", "e2e")

# Detected by probe_holistic.py (normalized video coords).
L_SHOULDER = (0.700, 0.628)
R_SHOULDER = (0.186, 0.685)
NOSE = (0.452, 0.392)


def sample_skin(img, x, y, r=4):
    """Median BGR colour of a small patch (used to tint the drawn arms)."""
    patch = img[max(0, y - r):y + r, max(0, x - r):x + r]
    return tuple(int(v) for v in np.median(patch.reshape(-1, 3), axis=0))


def draw_spread_arms(img):
    """Horizontal skin-tone arms from each shoulder joint to the frame edge.

    Skin is sampled from the face (not the torso, which is clothed); a short
    sleeve stub in the tee colour keeps the shoulder junction plausible.
    """
    h, w = img.shape[:2]
    tee = sample_skin(img, int(NOSE[0] * w), 360, r=30)  # torso area below shirt collar
    face_skin = sample_skin(img, int(NOSE[0] * w) - 25, int(NOSE[1] * h), r=6)
    for (nx, ny), sign in ((L_SHOULDER, 1), (R_SHOULDER, -1)):
        sx = int(nx * w)
        sy = int(ny * h)
        ex = (w - 8) if sign > 0 else 8
        ey = sy + 14  # slight natural droop
        # sleeve stub (short sleeve tee) over the deltoid
        mx = sx + sign * 52
        cv2.line(img, (sx, sy), (mx, ey), tee, 42, cv2.LINE_AA)
        # bare arm out to the hand
        cv2.line(img, (mx, ey), (ex, ey), face_skin, 30, cv2.LINE_AA)
        cv2.line(img, (mx, ey + 6), (ex, ey + 6), tuple(int(v * 0.85) for v in face_skin), 8, cv2.LINE_AA)
        cv2.circle(img, (ex, ey), 17, face_skin, -1, cv2.LINE_AA)


def perturb(img, n):
    """Subtle per-frame brightness flicker + noise so the clip reads as live."""
    gain = 1.0 + 0.015 * np.sin(2 * np.pi * n / 21.0)
    out = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    rng = np.random.default_rng(n)
    noise = rng.integers(-2, 3, img.shape, dtype=np.int16)
    return np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def bgr_to_i420_fullrange(bgr):
    """BT.601 FULL-range BGR -> planar I420 (matches the C420jpeg label)."""
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = (b - y) * 0.564 + 128.0
    v = (r - y) * 0.713 + 128.0
    h, w = y.shape
    u_small = u.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    v_small = v.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
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
            f.write(bgr_to_i420_fullrange(perturb(base, n)))
            if n % 150 == 0:
                print(f"  {os.path.basename(path)}: frame {n}/{frames}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    still = cv2.imread(SRC)
    assert still is not None, SRC

    spread = still.copy()
    draw_spread_arms(spread)

    write_y4m(os.path.join(OUT_DIR, "test_subject_armsdown.y4m"), still)
    write_y4m(os.path.join(OUT_DIR, "test_subject_armsspread.y4m"), spread)

    cv2.imwrite(os.path.join(OUT_DIR, "preview_armsdown.png"), still)
    cv2.imwrite(os.path.join(OUT_DIR, "preview_armsspread.png"), spread)
    print("done")


if __name__ == "__main__":
    main()
