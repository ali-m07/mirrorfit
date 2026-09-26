# tools/e2e/make_test_video.py
"""Generate YUV4MPEG2 (.y4m) test clips for the headless fit harness.

Chrome's --use-file-for-fake-video-capture accepts YUV4MPEG2 streams labelled
C420jpeg (full-range YCbCr), so we render a synthetic front-facing test subject
with OpenCV and pack BT.601 full-range I420 frames manually (cv2's
COLOR_BGR2YUV_I420 emits studio-swing video range, which Chrome would decode
washed-out under the jpeg-range label).

Two variants are produced:
  - test_subject_armsdown.y4m  : arms hanging at the sides
  - test_subject_armsspread.y4m: arms spread horizontally (T-pose-ish)

Usage:
    python tools/e2e/make_test_video.py [--width 640] [--height 480]
                                        [--fps 30] [--frames 450]
                                        [--out-dir output/e2e]
"""
import argparse
import os

import cv2
import numpy as np

# BGR colours
WALL_TOP = (52, 46, 40)      # background gradient (dark room, BGR)
WALL_BOTTOM = (24, 22, 20)
SKIN = (146, 174, 222)       # light skin tone
SKIN_HI = (176, 200, 240)    # lit skin
SKIN_LO = (108, 138, 190)    # shaded skin
SHIRT = (150, 92, 44)        # medium blue tee
SHIRT_LO = (110, 66, 30)     # shirt shading
HAIR = (30, 38, 52)          # dark brown hair
HAIR_HI = (58, 70, 92)
EYE = (40, 30, 25)
MOUTH = (90, 70, 120)


def vgrad(h, w, top, bottom):
    """Vertical BGR gradient image."""
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]          # (h,1,1)
    top_a = np.array(top, dtype=np.float32)[None, None, :]             # (1,1,3)
    bot_a = np.array(bottom, dtype=np.float32)[None, None, :]
    row = top_a * (1 - t) + bot_a * t                                  # (h,1,3)
    return (row * np.ones((1, w, 1), dtype=np.float32)).astype(np.uint8)


def draw_person(img, w, h, arms_spread, t):
    """Draw a front-facing upper-body test subject centred in the frame."""
    cx = w // 2
    sway = int(round(3 * np.sin(2 * np.pi * t * 0.6)))
    bob = int(round(2 * np.sin(2 * np.pi * t * 0.9 + 1.3)))

    head_cy = 118 + bob
    shoulder_y = 216 + bob
    half_shoulder = int(w * 0.225)  # shoulder span ~45% of frame width

    # ---- torso (shirt) with vertical shading + wrinkle hints ---------------
    half_hip = int(w * 0.195)
    torso = np.array([
        [cx - half_shoulder + sway, shoulder_y],
        [cx + half_shoulder + sway, shoulder_y],
        [cx + half_hip + sway, h],
        [cx - half_hip + sway, h],
    ], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [torso], SHIRT)
    cv2.addWeighted(overlay, 1.0, img, 0.0, 0, img)
    # shade the torso sides for roundness
    shade = img.copy()
    cv2.fillPoly(shade, [np.array([
        [cx - half_shoulder + sway, shoulder_y],
        [cx - half_shoulder + 34 + sway, shoulder_y],
        [cx - half_hip + 26 + sway, h],
        [cx - half_hip + sway, h],
    ], dtype=np.int32)], SHIRT_LO)
    cv2.fillPoly(shade, [np.array([
        [cx + half_shoulder - 34 + sway, shoulder_y],
        [cx + half_shoulder + sway, shoulder_y],
        [cx + half_hip + sway, h],
        [cx + half_hip - 26 + sway, h],
    ], dtype=np.int32)], SHIRT_LO)
    cv2.addWeighted(shade, 0.55, img, 0.45, 0, img)
    # collar
    cv2.ellipse(img, (cx + sway, shoulder_y + 2), (36, 16), 0, 0, 180, SKIN_LO, -1, cv2.LINE_AA)

    # ---- neck --------------------------------------------------------------
    cv2.rectangle(img, (cx - 17 + sway, head_cy + 40), (cx + 17 + sway, shoulder_y + 12), SKIN, -1)
    cv2.rectangle(img, (cx + 6 + sway, head_cy + 40), (cx + 17 + sway, shoulder_y + 12), SKIN_LO, -1)

    # ---- arms --------------------------------------------------------------
    if not arms_spread:
        for sign in (-1, 1):
            sx = cx + sign * (half_shoulder - 8) + sway
            cv2.line(img, (sx, shoulder_y + 16), (sx + sign * 18, shoulder_y + 66), SHIRT, 46, cv2.LINE_AA)
            cv2.line(img, (sx + sign * 16, shoulder_y + 58), (sx + sign * 34, h - 18), SKIN, 34, cv2.LINE_AA)
            cv2.line(img, (sx + sign * 28, shoulder_y + 60), (sx + sign * 44, h - 20), SKIN_LO, 10, cv2.LINE_AA)
            cv2.circle(img, (sx + sign * 34, h - 22), 19, SKIN, -1, cv2.LINE_AA)
    else:
        for sign in (-1, 1):
            sx = cx + sign * (half_shoulder - 8) + sway
            ex = cx + sign * (w // 2 - 14) + sway
            midx = cx + sign * int((half_shoulder + (w // 2 - 14)) * 0.62) + sway
            cv2.line(img, (sx, shoulder_y + 12), (midx, shoulder_y + 12), SHIRT, 44, cv2.LINE_AA)
            cv2.line(img, (midx, shoulder_y + 12), (ex, shoulder_y + 12), SKIN, 34, cv2.LINE_AA)
            cv2.circle(img, (ex + sign * 6, shoulder_y + 12), 19, SKIN, -1, cv2.LINE_AA)

    # ---- head --------------------------------------------------------------
    # ears first (behind the face oval)
    for sign in (-1, 1):
        cv2.ellipse(img, (cx + sign * 44 + sway, head_cy + 6), (9, 15), 0, 0, 360, SKIN_LO, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx + sway, head_cy), (44, 54), 0, 0, 360, SKIN, -1, cv2.LINE_AA)
    # lit/shaded halves for volume
    half_face = img.copy()
    cv2.ellipse(half_face, (cx + sway, head_cy), (44, 54), 0, -90, 90, SKIN_LO, -1, cv2.LINE_AA)
    cv2.addWeighted(half_face, 0.35, img, 0.65, 0, img)
    # hair cap + side hair
    cv2.ellipse(img, (cx + sway, head_cy - 10), (46, 50), 0, 180, 360, HAIR, -1, cv2.LINE_AA)
    cv2.rectangle(img, (cx - 46 + sway, head_cy - 10), (cx - 30 + sway, head_cy + 12), HAIR, -1, cv2.LINE_AA)
    cv2.rectangle(img, (cx + 30 + sway, head_cy - 10), (cx + 46 + sway, head_cy + 12), HAIR, -1, cv2.LINE_AA)
    # hair highlight
    cv2.ellipse(img, (cx - 12 + sway, head_cy - 26), (20, 10), -20, 0, 360, HAIR_HI, -1, cv2.LINE_AA)
    # brows / eyes / nose / mouth
    for sign in (-1, 1):
        exx = cx + sign * 18 + sway
        cv2.ellipse(img, (exx, head_cy - 14), (10, 3), 0, 0, 180, EYE, -1, cv2.LINE_AA)
        cv2.ellipse(img, (exx, head_cy - 2), (7, 4), 0, 0, 360, (235, 245, 250), -1, cv2.LINE_AA)
        cv2.circle(img, (exx, head_cy - 2), 3, EYE, -1, cv2.LINE_AA)
    cv2.line(img, (cx + sway, head_cy + 6), (cx + sway, head_cy + 18), SKIN_LO, 3, cv2.LINE_AA)
    cv2.ellipse(img, (cx + sway, head_cy + 30), (16, 6), 0, 0, 180, MOUTH, 2, cv2.LINE_AA)


def make_frame(w, h, arms_spread, t):
    img = vgrad(h, w, WALL_TOP, WALL_BOTTOM)
    draw_person(img, w, h, arms_spread, t)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    rng = np.random.default_rng(int(t * 1000) & 0x7fffffff)
    noise = rng.integers(-2, 3, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


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
    planes = [
        np.clip(y, 0, 255).astype(np.uint8).tobytes(),
        np.clip(u_small, 0, 255).astype(np.uint8).tobytes(),
        np.clip(v_small, 0, 255).astype(np.uint8).tobytes(),
    ]
    return b"".join(planes)


def write_y4m(path, w, h, fps, frames, arms_spread):
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F{fps}:1 Ip A1:1 C420jpeg\n".encode("ascii"))
        for n in range(frames):
            bgr = make_frame(w, h, arms_spread, n / fps)
            f.write(b"FRAME\n")
            f.write(bgr_to_i420_fullrange(bgr))
            if n % 90 == 0:
                print(f"  {os.path.basename(path)}: frame {n}/{frames}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=450)
    ap.add_argument("--out-dir", default=os.path.join("output", "e2e"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    write_y4m(os.path.join(args.out_dir, "test_subject_armsdown.y4m"),
              args.width, args.height, args.fps, args.frames, arms_spread=False)
    write_y4m(os.path.join(args.out_dir, "test_subject_armsspread.y4m"),
              args.width, args.height, args.fps, args.frames, arms_spread=True)

    for name, spread in (("preview_armsdown.png", False), ("preview_armsspread.png", True)):
        cv2.imwrite(os.path.join(args.out_dir, name), make_frame(args.width, args.height, spread, 0.5))
    print("done")


if __name__ == "__main__":
    main()
