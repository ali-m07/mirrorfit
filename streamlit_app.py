"""Streamlit demo frontend (optional alternative to web/index.html).

Run alongside the API server::

    python api_server.py                 # terminal 1  (serves the pipeline)
    streamlit run streamlit_app.py       # terminal 2  (this UI)

The app talks to the FastAPI backend over HTTP, so it demonstrates the exact
integration path a brand's web storefront would use.
"""

from __future__ import annotations

import time

import requests
import streamlit as st

API = st.secrets.get("api_base", "http://127.0.0.1:8000")

st.set_page_config(page_title="AR Virtual Try-On", page_icon="🪞", layout="wide")
st.title("🪞 AR Virtual Try-On — Demo Console")
st.caption(f"Connected to backend at `{API}`")


def call(method: str, path: str, **kwargs):
    try:
        res = requests.request(method, f"{API}{path}", timeout=10, **kwargs)
        res.raise_for_status()
        return res.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


# -- session control ---------------------------------------------------------

col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    if st.button("▶ Start session", use_container_width=True):
        call("POST", "/tryon/start", json={})
with col_b:
    if st.button("■ Stop session", use_container_width=True):
        call("POST", "/tryon/stop", json={})

health = call("GET", "/health")
status = call("GET", "/status") if health and health.get("engine_running") else None

if not status:
    st.info("Start a session to begin the live try-on.")
    st.stop()

# -- live view ----------------------------------------------------------------

left, right = st.columns([2, 1])
with left:
    st.image(f"{API}/stream.mjpeg", caption="Live try-on mirror", use_column_width=True)

with right:
    st.metric("FPS", status.get("fps", 0))
    st.write("**Wearing:**", (status.get("current_cloth") or {}).get("name", "–"))

    clothes = call("GET", "/clothes") or {"items": []}
    names = {i["id"]: i["name"] for i in clothes["items"]}
    if names:
        choice = st.selectbox("Garment", list(names.keys()),
                              format_func=lambda k: names[k])
        if st.button("Wear it", use_container_width=True):
            call("POST", "/tryon/change-cloth", json={"cloth_id": choice})
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Prev"):
            call("POST", "/tryon/change-cloth", json={"direction": "previous"})
    with c2:
        if st.button("Next →"):
            call("POST", "/tryon/change-cloth", json={"direction": "next"})

    st.divider()
    if st.button("📸 Screenshot", use_container_width=True):
        r = call("POST", "/tryon/screenshot", json={})
        if r:
            st.success(f"Saved: {r['filename']}")
    if st.button("⏺ Toggle recording", use_container_width=True):
        r = call("POST", "/tryon/record", json={"action": "toggle"})
        if r:
            st.success(r["message"])

    studio = st.toggle("🎬 Virtual Studio", value=bool(status.get("studio_enabled")))
    call("POST", "/tryon/studio", json={"enabled": studio})
    mode = st.radio("Body mode", ["upper", "full"],
                    index=0 if status.get("mode") == "upper" else 1, horizontal=True)
    call("POST", "/tryon/mode", json={"mode": mode})

st.caption("Tip: the raw MJPEG feed is available at `/stream.mjpeg` for embedding "
           "in any webpage; a WebSocket feed is at `/ws/stream`.")
time.sleep(0.1)
