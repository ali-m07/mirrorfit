# 🪞 MirrorFit ( Real-Time AR Virtual Try-On Platform

> **Let your customers wear your collection before they buy it ( live, in the browser or in-store, at 25-30 FPS.**

MirrorFit is a production-ready, white-labelable **augmented-reality virtual clothing try-on system** built by **Ali Mansouri**. A shopper stands in front of a webcam, and garments from your catalog (t-shirts, hoodies, jackets, dresses) are composited onto their body in real time, tracking shoulders, torso scale, tilt and arm occlusion. Sessions are captured as shareable screenshots and videos.

Built on **OpenCV + MediaPipe + FastAPI**, it runs as a **desktop kiosk app** (retail mirrors, trade shows) and as an **HTTP/WebSocket API** (e-commerce integration, SaaS embedding) from the same codebase.

---

## 💼 Why it matters (for business decision makers)

- **Returns are fashion e-commerce's biggest cost.** 30-40% of online apparel is returned, and "didn't look right on me" is the #1 reason. Virtual try-on directly attacks that number.
- **Conversion lift.** Shoppers who engage with try-on features buy more confidently and add more items per session.
- **Zero-install for customers, one dependency stack for you.** The entire experience runs from a webcam and this Python service ( no app downloads, no special hardware, no per-user licensing.
- **Your brand, not ours.** The platform is designed for white-label deployment: swap the theme colors, drop your garment PNGs into a folder, point your storefront at the API. (See *Licensing*.)
- **Content engine built in.** Every session can produce timestamped screenshots and video clips ( user-generated content for social, with the garment name baked into the filename.

## ✨ Key features

| Area | What you get |
|---|---|
| **Real-time try-on** | Live webcam feed with MediaPipe pose tracking at 25-30+ FPS (720p) |
| **Accurate garment fit** | Automatic scaling from shoulder width & torso length, rotation from shoulder tilt, per-asset anchor tuning |
| **Believable compositing** | Transparent-PNG alpha blending, soft drop shadow, scene-brightness matching, forearm occlusion (arms pass *in front of* the garment) |
| **Temporal smoothing** | EMA landmark filtering + last-good-pose hold: no jitter, no garment popping |
| **🎬 Virtual Studio** | Background replacement via selfie segmentation: gradient, blur, or branded backdrop image |
| **Upper / Full body modes** | One-key switch between torso-framed and full-length framing |
| **Capture** | Timestamped screenshots (with garment name) and MP4 session recordings |
| **API-first** | FastAPI + Uvicorn: REST endpoints, MJPEG stream, and a WebSocket live feed |
| **Two web demos included** | A dependency-free HTML/JS console (served at `/demo`) and a Streamlit app |
| **Ops-ready** | YAML config + env overrides, JSON structured logging, rotating log files, session analytics (switches, screenshots, durations) |

## 🏗 Technical architecture

```
            ┌──────────────────────────────────────────────────────┐
            │                    TryOnEngine (thread)               │
            │                                                       │
 Webcam ──► │  WebcamStream ─► PoseEstimator ─► ClothingRenderer    │ ──► latest_frame()
 (threaded) │  (grab thread)   (MediaPipe Pose    (affine warp,     │     latest_jpeg()
            │                  + EMA smoothing)    shadow, occlusion)│
            │            ┌────► PersonSegmenter ─► StudioCompositor │
            │            │    (Virtual Studio mode)                 │
            └────────────┼──────────────────────────────────────────┘
                         │
        ┌────────────────┼─────────────────┐
        ▼                ▼                 ▼
   main.py          api_server.py     streamlit_app.py
  (desktop        (FastAPI: REST +    (demo frontend,
   kiosk app)      MJPEG + WebSocket)  talks to the API)
```

**Design principles**

- **Single-owner MediaPipe graphs.** `Pose` and `SelfieSegmentation` are not thread-safe, so the engine's worker thread owns them; every other component reads immutable snapshots through locks.
- **Threaded capture.** Camera I/O runs on its own thread with a tiny frame queue, so the pipeline never blocks on the driver.
- **Renderer purity.** `ClothingRenderer` is a stateless function of `(frame, garment, pose)` ( trivially testable, and the seam where 3D garments or a neural warper (e.g. a diffusion-based try-on) can be swapped in later.
- **Catalog-driven assets.** Garments are discovered from a folder + optional `catalog.json` metadata; adding a new SKU is a file copy, not a code change.

### Project layout

```
ar_virtual_tryon/
├── main.py                     # Desktop kiosk application
├── api_server.py               # FastAPI application (REST + MJPEG + WebSocket)
├── streamlit_app.py            # Optional Streamlit demo console
├── config.yaml                 # Every tunable in one place
├── requirements.txt
├── .env.example
├── web/index.html              # Zero-dependency web demo (served at /demo)
├── tools/generate_sample_clothes.py   # Procedural demo garment generator
├── src/
│   ├── camera/webcam.py        # Threaded capture
│   ├── vision/pose_estimator.py       # MediaPipe Pose + EMA smoothing
│   ├── vision/segmenter.py            # Selfie segmentation + studio compositing
│   ├── vision/clothing_renderer.py    # Pose-aligned warp, shadow, occlusion
│   ├── core/tryon_engine.py           # Pipeline orchestration (single owner)
│   ├── core/session_manager.py        # Session lifecycle + analytics
│   ├── api/routes.py  api/schemas.py  # Versioned REST contracts
│   ├── ui/overlay.py                  # Kiosk-grade HUD
│   └── utils/                         # Config, logging, catalog, imaging helpers
├── assets/clothes/             # Transparent garment PNGs + catalog.json
├── output/screenshots/  recordings/
└── tests/                      # Headless test suite (synthetic camera)
```

## 🚀 Getting started

### 1. Install

```bash
git clone <your-repo-url> && cd ar_virtual_tryon
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

> Requires **Python 3.10-3.12** (MediaPipe compatibility) and a webcam.

### 2. Generate the demo wardrobe

```bash
python tools/generate_sample_clothes.py
```

This creates five transparent garment PNGs (tee, hoodie, jacket, dress, polo)
plus `assets/clothes/catalog.json`. Replace them with your own products anytime.

### 3a. Run the desktop app

```bash
python main.py
```

### 3b. Run the API service

```bash
python api_server.py
# → http://localhost:8000
# → interactive docs:  http://localhost:8000/docs
# → zero-install demo: http://localhost:8000/demo   (press "Start Session")
```

Then either open the built-in demo console, or the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

## ⌨️ Keyboard controls (desktop app)

| Key | Action |
|---|---|
| `N` / `B` | Next / previous garment |
| `S` | Screenshot → `output/screenshots/tryon_<timestamp>_<garment>.png` |
| `R` | Start/stop recording → `output/recordings/tryon_<timestamp>_session.mp4` |
| `V` | Toggle **Virtual Studio** (background replacement) |
| `M` | Toggle upper-body / full-body mode |
| `D` | Toggle pose debug skeleton |
| `Q` / `Esc` | Quit |

## 🌐 API documentation

Base URL: `http://localhost:8000` · Interactive OpenAPI docs at **`/docs`**.

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness + engine state + uptime |
| `GET /status` | Live engine snapshot (fps, current garment, mode, recording…) |
| `GET /clothes` | List the garment catalog |
| `POST /tryon/start` | Start a try-on session (optional `{"camera_index": 1}`) |
| `POST /tryon/stop` | Stop the session, returns the session summary |
| `POST /tryon/change-cloth` | `{"cloth_id": "navy-hoodie"}` or `{"direction": "next"}` |
| `POST /tryon/screenshot` | Capture; returns `{"path", "filename"}` |
| `POST /tryon/record` | `{"action": "start" \| "stop" \| "toggle"}` |
| `POST /tryon/mode` | `{"mode": "upper" \| "full"}` |
| `POST /tryon/studio` | `{"enabled": true}` → Virtual Studio on/off |
| `GET /stream.mjpeg` | MJPEG live feed, drop into any `<img>` tag |
| `WS  /ws/stream` | Binary JPEG frames pushed at `api.stream_max_fps` |

**Embed in any webpage in two lines:**

```html
<img src="http://localhost:8000/stream.mjpeg" />
<button onclick="fetch('http://localhost:8000/tryon/change-cloth',
  {method:'POST', headers:{'Content-Type':'application/json'},
   body:'{"direction":"next"}'})">Next garment</button>
```

## 👕 Preparing clothing assets (important for quality)

The renderer assumes garments are **front-facing product renders** with a transparent background:

1. **Format:** PNG with a real alpha channel (RGBA). No JPG.
2. **Orientation:** perfectly front view, garment straight, **neckline at the top** of the image.
3. **Crop:** tight to the garment; the shoulder seam line should span most of the image width. Leave a few pixels of transparent margin.
4. **Recommended resolution:** 700-1200 px wide. Larger wastes memory; smaller looks soft when scaled up.
5. **Alignment:** the *topmost opaque pixel* is treated as the neckline anchor, and full image width as the shoulder span. Fine-tune per garment in `assets/clothes/catalog.json`:
   - `anchor_top` (0-1): shift where the neckline sits relative to the detected neck point.
   - `anchor_width` (0-1): shrink/expand the effective shoulder span (e.g. `0.9` for slim-cut garments).
6. **Category:** set `"category": "dress"` for long garments; the renderer extends them toward the hips automatically.
7. **Best sources:** ghost-mannequin product photos with background removed, 3D renders (CLO3D / Browzwear export), or AI-generated garment cutouts.

Drop the file into `assets/clothes/`, add an entry to `catalog.json` (optional), and it appears in the UI and `/clothes` API immediately; no restart needed if you call `catalog.reload()`.

## ⚙️ Configuration

Everything is in [`config.yaml`](config.yaml): camera index/resolution, MediaPipe confidence thresholds, smoothing strength, garment scale factors, shadow/lighting effects, studio background, output paths, API host/port, log format. Any value can be overridden per-deployment with environment variables:

```bash
ARTRYON__CAMERA__INDEX=1 ARTRYON__API__PORT=9000 python api_server.py
```

## ✅ Tests

The suite runs fully headless (synthetic camera, no webcam or MediaPipe needed):

```bash
pytest
```

Covers alpha compositing edge cases, catalog discovery/metadata, renderer geometry (scaling, tilt, clamping, dress extension), session lifecycle, engine control flow, and the full API contract via FastAPI's `TestClient`.

## 🛣 Roadmap

The architecture is deliberately staged for commercial evolution:

- **Lower-body & accessories**: pants/skirts anchored on hips, hats/glasses anchored on face landmarks (renderer already accepts per-category anchors).
- **Multi-person try-on**: MediaPipe supports N poses; engine sessions become per-person.
- **3D garments**: swap `ClothingRenderer` for a mesh renderer (smpl/3DMM body fit) behind the same engine interface.
- **Generative try-on**: use this pipeline's pose+segmentation conditioning to drive diffusion-based photorealistic warping as a post-pass.
- **Mobile & in-browser**: the API/WebSocket layer is already the backend for a React Native app or a WebRTC/MediaPipe-JS client.
- **SaaS control plane**: per-tenant catalogs (extra clothes dirs are config-driven), API keys, usage analytics from `SessionManager` counters, and branded themes per tenant.

## 📄 Licensing & white-label notes

This codebase is structured so a brand or agency can deploy it under **their own name**: themes, window titles, garment catalogs, studio backdrops and the web demo are all configuration or static assets; no code forks required.

- **MediaPipe**: Apache 2.0 (commercial use OK).
- **OpenCV**: Apache 2.0.
- **FastAPI / Streamlit / NumPy / Pillow**: permissive licenses (MIT/BSD).

Garment imagery and brand assets remain the property of their respective owners; you are responsible for rights to any product photos you load into `assets/clothes/`. For commercial licensing of *this platform* (hosted SaaS, OEM/white-label, or on-prem), contact the maintainers.

---

## 🎯 Perfect For (Use Cases)

This is a **working demo** ready to be tailored to your brand:

- **Fashion E-Commerce**: Let shoppers try before they buy. Integrate with your product catalog, track engagement, reduce returns by 30-40%.
- **Retail Stores & Showrooms**: Interactive mirrors for in-store experience. Multiple customers can try garments simultaneously.
- **Fashion Brands**: White-label solution. Upload your seasonal collections, customize the UI to match your brand identity.
- **Marketing Campaigns**: Generate shareable AR content. Users can record videos wearing your products and share on social media.
- **Trade Shows & Events**: Stand out with interactive demos. No app downloads required.

### 💡 AI-Powered Recommendations (Coming Soon)

We're building intelligent features to maximize conversions:

- **Smart Garment Suggestions**: AI analyzes body shape, style preferences, and trending items to recommend what looks best on each customer
- **Virtual Styling Assistant**: "This jacket pairs perfectly with the shirt you're wearing" - cross-sell automation
- **Fit Prediction**: ML models predict garment fit based on body measurements, reducing size-related returns
- **Personalized Lookbooks**: Generate custom outfit combinations for each user based on their try-on history

**Want to customize this for your brand?** We can adapt the system to:
- Connect to your existing product database and inventory
- Match your brand colors, logo, and UI style
- Add custom analytics and conversion tracking
- Integrate with your e-commerce platform (Shopify, WooCommerce, custom)
- Add your AI recommendation engine

Contact us to discuss white-label deployment and custom features.

---

## 👋 About the Developer

**Ali Mansouri** - Solutions Architect | Cloud-Native · Kubernetes · LLM/RAG · Enterprise Automation

This AR virtual try-on platform was designed and built by Ali Mansouri, combining production-grade computer vision, real-time rendering, and API-first architecture. 

**Background:** Solutions Architect at Snapp (Iran's largest super-app), with 5+ years building cloud-native platforms, enterprise AI systems (RAG, LangChain), and LLM automation. Currently pursuing a PhD in Futures Studies with published research in entrepreneurship and HR tech trends.

**Expertise:** Kubernetes, Helm, CI/CD, Python, enterprise workflow automation (n8n, Jira), machine learning (CNN, MLflow), and production AI architectures.

**Interested in collaborating?** If you're working on fashion tech, AR/VR experiences, e-commerce innovation, or production AI systems, let's connect:

- 📧 Email: ali.mansouri1998@gmail.com
- 💼 LinkedIn: [linkedin.com/in/ali-mansouri-a7984215b](https://www.linkedin.com/in/ali-mansouri-a7984215b/)
- 🔬 ResearchGate: [researchgate.net/profile/Ali-Mansouri-44](https://www.researchgate.net/profile/Ali-Mansouri-44)
- 📚 Google Scholar: [scholar.google.com](https://scholar.google.com)

Open to: international relocation, remote collaboration, and strategic partnerships in fashion tech and AI-powered retail experiences.

---

*Built with OpenCV · MediaPipe · FastAPI · Streamlit*
