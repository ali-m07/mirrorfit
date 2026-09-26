# tools/e2e/run_fit_test.py â€” headless browser fit harness.
#
# Drives the real softWEAR UI (splash -> privacy -> male -> try-on) with a fake
# camera backed by a .y4m clip, waits for pose init, then screenshots:
#   full page, video canvas (person), 3D garment canvas (transparent).
#
# Usage:
#   python tools/e2e/run_fit_test.py --iter 1 --pose armsdown
#   python tools/e2e/run_fit_test.py --iter 1 --pose armsspread
#   python tools/e2e/run_fit_test.py --iter 2 --pose armsdown --retail-image \
#       assets/clothes/pngtree-white-hoodie-mockup-cutout-png-file-png-image_10159088.png
#
# Notes:
# - Uses the system Chrome (Playwright channel="chrome"); the Chromium-for-
#   Testing CDN is unreachable from this machine.
# - The y4m loops, so the subject is permanently "on camera".
import argparse
import json
import os
import sys
import time

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "output", "e2e")
BASE_URL = "http://127.0.0.1:8000/3d/"

POSE_VIDEOS = {
    # primary subject: public-domain COCO sample, standing, visible hips,
    # darkened backdrop; MediaPipe detects all landmarks with v>=0.89
    "armsdown": os.path.join(OUT_DIR, "test_subject_standing_dark.y4m"),
    "armsspread": os.path.join(OUT_DIR, "test_subject_standing_dark.y4m"),
    # previous seated subject (crimson tee), darkened backdrop
    "seated": os.path.join(OUT_DIR, "test_subject_armsdown_dark.y4m"),
}
GREEN_PRODUCT = os.path.join(ROOT, "tools", "e2e", "assets", "solid_green_product.png")


def run(iteration, pose, retail_image=None, settle_seconds=18, inject_tpose=False):
    video = POSE_VIDEOS[pose]
    assert os.path.exists(video), video
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"iter_{iteration}_{pose}" + ("_tpose" if inject_tpose else "") + ("_retail" if retail_image else "")
    console_lines = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=True,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-video-capture={video}",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        ctx = browser.new_context(
            permissions=["camera"],
            viewport={"width": 1280, "height": 800},
            device_scale_factor=1,
        )
        page = ctx.new_page()
        page.on("console", lambda msg: console_lines.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: console_lines.append(f"[pageerror] {err}"))

        page.goto(BASE_URL, wait_until="domcontentloaded")

        # Splash -> Launch Try-On
        try:
            page.wait_for_selector(".primary-btn", timeout=15000)
            page.click(".primary-btn")
        except Exception:
            pass  # splash may be skipped via sessionStorage

        # Privacy modal (fresh profile -> always shows)
        try:
            page.wait_for_selector(".btn-privacy-agree", timeout=8000)
            page.click(".btn-privacy-agree")
        except Exception:
            pass

        # Gender selection
        page.wait_for_selector(".male-option", timeout=15000)
        page.click(".male-option")

        # Wait for the try-on panel and camera init
        page.wait_for_selector("#background-canvas", timeout=30000)

        # Optional retail texture upload via the front-image input
        if retail_image:
            img_path = os.path.join(ROOT, retail_image) if not os.path.isabs(retail_image) else retail_image
            try:
                page.wait_for_selector("input[aria-label='Front garment image']", timeout=20000)
                page.set_input_files("input[aria-label='Front garment image']", img_path)
                console_lines.append("[harness] retail front image set: " + img_path)
            except Exception as e:
                console_lines.append(f"[harness] retail image input NOT found: {e}")

        # Wait for pose init: fitDebug populated + garment mesh present.
        deadline = time.time() + settle_seconds + 20
        debug_ready = False
        while time.time() < deadline:
            dbg = page.evaluate("window.softWearPerformance && window.softWearPerformance.fitDebug || null")
            if dbg:
                debug_ready = True
                break
            time.sleep(1)

        def wait_converged(timeout=60):
            """Wait until the wrapper position stops moving at a usable frame
            rate (headless GL can throttle to ~3 FPS, which freezes the mapper
            mid-slerp and poisons screenshots)."""
            stable = 0
            last_pos = None
            conv_deadline = time.time() + timeout
            while time.time() < conv_deadline:
                cur = page.evaluate(
                    """() => {
                        const f = window.softWearPerformance && window.softWearPerformance.fitDebug;
                        return f ? { p: f.position, fps: window.softWearPerformance.garmentFPS || 0 } : null;
                    }"""
                )
                if cur and cur["fps"] >= 5 and last_pos is not None:
                    moved = sum(abs(cur["p"][k] - last_pos[k]) for k in ("x", "y", "z"))
                    stable = stable + 1 if moved < 0.006 else 0
                    if stable >= 4:
                        return True
                last_pos = cur["p"] if cur else None
                time.sleep(0.7)
            return False

        wait_converged()
        time.sleep(settle_seconds)

        if inject_tpose:
            # Harness-only override: drive the arm bones with a known T-pose
            # (33 MediaPipe world landmarks, hips at origin, arms level with
            # shoulders) so sleeve articulation can be judged even though no
            # available still shows clearly-visible spread forearms.
            tpose = [{"x": 0, "y": 0, "z": 0, "visibility": 0.0}] * 33
            tpose[0] = {"x": 0.0, "y": -0.45, "z": 0.0, "visibility": 0.99}   # nose
            tpose[11] = {"x": -0.20, "y": -0.35, "z": 0.0, "visibility": 0.99}  # L shoulder
            tpose[12] = {"x": 0.20, "y": -0.35, "z": 0.0, "visibility": 0.99}   # R shoulder
            # MediaPipe convention: the person's LEFT side is at +x, so the
            # left arm spreads toward +x and the right arm toward -x.
            tpose[13] = {"x": 0.45, "y": -0.35, "z": 0.0, "visibility": 0.99}   # L elbow
            tpose[14] = {"x": -0.45, "y": -0.35, "z": 0.0, "visibility": 0.99}  # R elbow
            tpose[15] = {"x": 0.62, "y": -0.35, "z": 0.0, "visibility": 0.99}   # L wrist
            tpose[16] = {"x": -0.62, "y": -0.35, "z": 0.0, "visibility": 0.99}  # R wrist
            tpose[23] = {"x": -0.12, "y": 0.0, "z": 0.0, "visibility": 0.99}    # L hip
            tpose[24] = {"x": 0.12, "y": 0.0, "z": 0.0, "visibility": 0.99}     # R hip
            page.evaluate("lm => { window.softWearPerformance.testWorldLandmarks = lm; }", tpose)
            console_lines.append("[harness] T-pose world landmarks injected")
            wait_converged()
            time.sleep(settle_seconds)

        state = page.evaluate(
            """() => ({
                fitDebug: window.softWearPerformance && window.softWearPerformance.fitDebug,
                garmentFPS: window.softWearPerformance && window.softWearPerformance.garmentFPS,
                rects: {
                    background: document.getElementById('background-canvas')?.getBoundingClientRect(),
                    garment: document.querySelector("div[style*='scaleX(-1)']")?.getBoundingClientRect(),
                },
            })"""
        )
        console_lines.append("[harness] page state: " + json.dumps(state))

        # Screenshots first (clean frames): full page, video canvas, garment.
        # Hide every overlay that floats over the video panel (retail upload,
        # garment chooser buttons, info/gesture indicators) so the pixel
        # scorer sees only the video + garment layers.
        page.evaluate(
            """() => {
                const hide = (el) => { if (el) { el.style.display = 'none'; el.dataset.fitHidden = '1'; } };
                hide([...document.querySelectorAll('div')].find(d => (d.textContent || '').trim().startsWith('Retail image')));
                document.querySelectorAll('.on-screen-controls, .garment-info-indicator, .gesture-status-indicator, .model-view-indicator, .gesture-indicator-overlay, .menu-toggle-btn, .control-sidebar')
                    .forEach(hide);
            }"""
        )
        page.screenshot(path=os.path.join(OUT_DIR, f"{tag}_full.png"))
        try:
            el = page.query_selector("#background-canvas")
            if el:
                # hide the 3D layer so the element shot is person-only
                page.eval_on_selector("div[style*='scaleX(-1)']", "el => el.style.visibility = 'hidden'")
                el.screenshot(path=os.path.join(OUT_DIR, f"{tag}_video.png"))
                page.eval_on_selector("div[style*='scaleX(-1)']", "el => el.style.visibility = ''")
        except Exception as e:
            console_lines.append(f"[harness] screenshot video failed: {e}")
        # garment-only: hide the video canvas so the element shot captures just
        # the 3D layer over a transparent background
        try:
            el = page.query_selector("div[style*='scaleX(-1)'] canvas")
            if el:
                page.eval_on_selector("#background-canvas", "el => el.style.visibility = 'hidden'")
                el.screenshot(path=os.path.join(OUT_DIR, f"{tag}_garment.png"))
                page.eval_on_selector("#background-canvas", "el => el.style.visibility = ''")
        except Exception as e:
            console_lines.append(f"[harness] screenshot garment failed: {e}")
        page.evaluate(
            """() => {
                document.querySelectorAll('[data-fit-hidden]').forEach(el => {
                    el.style.display = '';
                    delete el.dataset.fitHidden;
                });
            }"""
        )

        # Open the built-in debug panel (Shift+D) and scrape its text: it
        # reports pose-detection status, selected garment and mesh info.
        try:
            page.keyboard.press("Shift+d")
            page.wait_for_selector(".debug-panel", timeout=5000)
            debug_text = page.eval_on_selector(".debug-panel", "el => el.innerText")
            console_lines.append("[harness] debug-panel:\n" + debug_text)
        except Exception as e:
            console_lines.append(f"[harness] debug panel scrape failed: {e}")

        with open(os.path.join(OUT_DIR, f"{tag}_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1)
        with open(os.path.join(OUT_DIR, "console.log"), "a", encoding="utf-8") as f:
            f.write(f"\n===== {tag} ({time.strftime('%Y-%m-%d %H:%M:%S')}) =====\n")
            f.write("\n".join(console_lines) + "\n")

        print(json.dumps({"tag": tag, "fitDebug": state.get("fitDebug"), "debug_ready": debug_ready}, indent=1))
        browser.close()
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--pose", choices=list(POSE_VIDEOS), default="armsdown")
    ap.add_argument("--retail-image", default=None,
                    help="path to a product image for the UV bake; 'green' uses the solid green eval asset")
    ap.add_argument("--settle", type=float, default=18)
    ap.add_argument("--inject-tpose", action="store_true",
                    help="inject synthetic T-pose world landmarks for the sleeve test")
    args = ap.parse_args()
    retail = GREEN_PRODUCT if args.retail_image == "green" else args.retail_image
    run(args.iter, args.pose, retail, args.settle, args.inject_tpose)


if __name__ == "__main__":
    main()
