import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://app:8009")
PHOTOS_DIR = "data/photos"

st.set_page_config(page_title="Smart Photo Knowledgebase", layout="wide")

# --------------- Health checks ---------------

def check_api(endpoint: str) -> bool:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def check_ollama() -> bool:
    try:
        r = requests.head("http://ollama_smart_image_knowledgebase:11434/", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def check_redis() -> bool:
    try:
        import redis
        r = redis.Redis(host="redis", port=6379)
        r.ping()
        return True
    except Exception:
        return False

st.title("📸 Smart Photo Knowledgebase")
st.write("Upload → Ingest → Search")

st.subheader("✅ Services")
c1, c2, c3 = st.columns(3)
with c1:
    if check_api("/health"):
        st.success("FastAPI: OK")
    else:
        st.error("FastAPI: DOWN")
with c2:
    if check_ollama():
        st.success("Ollama: OK")
    else:
        st.error("Ollama: DOWN")

with c3:
    if check_redis():
        st.success("Redis: OK")
    else:
        st.error("Redis: DOWN")
st.markdown("---")

# --------------- Upload ---------------

st.subheader("📤 Upload Images (drag & drop)")

uploaded_files = st.file_uploader(
    "Drop images here",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    for f in uploaded_files:
        path = os.path.join(PHOTOS_DIR, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} images. ✅")
    st.info("Now click ‘Start Ingestion’ to index them.")

st.markdown("---")

# --------------- Ingest ---------------

st.subheader("🧠 Run Image Ingestion")

if st.button("🚀 Start Ingestion"):
    st.info("Ingestion started…")
    try:
        resp = requests.post(f"{API_BASE}/ingest_images", timeout=300)
        if resp.status_code == 200:
            data = resp.json()
            st.success(
                f"Ingestion complete! ✅ Added: {data.get('added')} | Skipped: {data.get('skipped')}"
            )
        else:
            st.error(f"Backend error: {resp.status_code} - {resp.text}")
    except Exception as e:
        st.error(f"Ingestion failed: {e}")

st.markdown("---")

# --------------- Search ---------------

st.subheader("🔍 Search Photos")

query = st.text_input("Query (e.g. 'ritik on a hill', 'sunset selfie', 'two people')")

if st.button("🔎 Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        st.info("Searching…")
        try:
            resp = requests.post(
                f"{API_BASE}/query_text",
                json={"question": query, "k": 20},
                timeout=60,
            )
            if resp.status_code != 200:
                st.error(f"Search error: {resp.status_code} - {resp.text}")
            else:
                results = resp.json().get("results", [])
                if not results:
                    st.warning("No images found.")
                else:
                    st.success(f"Found {len(results)} images. ✅")
                    cols = st.columns(8)
                    for i, img in enumerate(results):
                        rel_path = img.get("path", "")
                        full_path = os.path.join(PHOTOS_DIR, rel_path)
                        if os.path.exists(full_path):
                            with cols[i % 8]:
                                st.image(full_path, use_container_width=True)
                        else:
                            st.error(f"Missing file: {full_path}")
        except Exception as e:
            st.error(f"Search failed: {e}")

st.markdown("---")
st.write("Simple Streamlit UI wired to your Smart Photo backend.")
