# tools/e2e/probe_holistic.py — run MediaPipe Holistic directly against a test
# image/frame source to check detection in isolation from the app.
#
# Usage:
#   python tools/e2e/probe_holistic.py <image1> [image2 ...]
#
# Loads each image onto a canvas, feeds it to Holistic (the same bundle the app
# uses, served from /3d/mediapipe/), and reports how many landmarks came back.
import os
import sys

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    images = [os.path.abspath(p) for p in sys.argv[1:]]
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=[
            "--use-fake-device-for-media-stream",
            "--autoplay-policy=no-user-gesture-required",
        ])
        page = browser.new_context(viewport={"width": 900, "height": 700}).new_page()
        errors = []
        page.on("console", lambda m: errors.append(f"[{m.type}] {m.text}"))
        page.goto("http://127.0.0.1:8000/3d/", wait_until="domcontentloaded")
        page.evaluate(
            """() => {
                document.body.innerHTML = '<canvas id="cam" width="640" height="480"></canvas>';
                window.probeReady = false;
                const s = document.createElement('script');
                s.src = './mediapipe/holistic.js';
                s.onload = async () => {
                    const holistic = new Holistic({ locateFile: (f) => `./mediapipe/${f}` });
                    holistic.setOptions({ modelComplexity: 1, minDetectionConfidence: 0.5,
                                          minTrackingConfidence: 0.5, selfieMode: false });
                    await holistic.initialize();
                    window._holistic = holistic;
                    window.detectOnCanvas = async () => {
                        const canvas = document.getElementById('cam');
                        let last = null;
                        holistic.onResults((res) => { last = res; });
                        for (let i = 0; i < 5; i++) {
                            await holistic.send({ image: canvas });
                            await new Promise(r => setTimeout(r, 100));
                        }
                        if (!last) return { ok: false };
                        const L = last.poseLandmarks;
                        const pick = (i) => L ? { x: +L[i].x.toFixed(3), y: +L[i].y.toFixed(3), z: +L[i].z.toFixed(3), v: +L[i].visibility.toFixed(2) } : null;
                        return {
                            ok: true,
                            pose: L ? L.length : 0,
                            face: last.faceLandmarks ? last.faceLandmarks.length : 0,
                            ls: pick(11), rs: pick(12), le: pick(13), re: pick(14), lh: pick(23), rh: pick(24),
                            nose: pick(0), lw: pick(15), rw: pick(16),
                        };
                    };
                    window.drawFile = async (name, b64) => {
                        const img = new Image();
                        img.src = 'data:image/png;base64,' + b64;
                        await img.decode();
                        document.getElementById('cam').getContext('2d').drawImage(img, 0, 0, 640, 480);
                        return true;
                    };
                    window.probeReady = true;
                };
                document.head.appendChild(s);
            }"""
        )
        page.wait_for_function("window.probeReady === true", timeout=60000)
        import base64
        for img_path in images:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            page.evaluate("b64 => window.drawFile('x', b64)", b64)
            result = page.evaluate("() => window.detectOnCanvas()")
            print(f"{os.path.basename(img_path)}: {result}")
        browser.close()


if __name__ == "__main__":
    main()
