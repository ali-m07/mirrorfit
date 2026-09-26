# tools/e2e/smoke_fakecam.py — verify Chrome launches with the fake y4m camera.
import os
import sys

from playwright.sync_api import sync_playwright

Y4M = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "output/e2e/test_subject_armsdown.y4m")

with sync_playwright() as p:
    browser = p.chromium.launch(
        channel="chrome",
        headless=True,
        args=[
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
            f"--use-file-for-fake-video-capture={Y4M}",
            "--autoplay-policy=no-user-gesture-required",
        ],
    )
    ctx = browser.new_context(permissions=["camera"], viewport={"width": 1280, "height": 800})
    page = ctx.new_page()
    page.goto("http://127.0.0.1:8000/health")
    result = page.evaluate(
        """async () => {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            const track = stream.getVideoTracks()[0];
            const settings = track.getSettings();
            const video = document.createElement('video');
            video.srcObject = stream;
            await video.play();
            await new Promise(r => setTimeout(r, 500));
            const c = document.createElement('canvas');
            c.width = video.videoWidth; c.height = video.videoHeight;
            c.getContext('2d').drawImage(video, 0, 0);
            const px = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            // mean luma to prove real frames arrive (not black)
            let sum = 0;
            for (let i = 0; i < px.length; i += 4) sum += 0.299*px[i] + 0.587*px[i+1] + 0.114*px[i+2];
            return { w: video.videoWidth, h: video.videoHeight,
                     meanLuma: (sum / (px.length / 4)).toFixed(1) };
        }"""
    )
    print("fake camera OK:", result)
    browser.close()
