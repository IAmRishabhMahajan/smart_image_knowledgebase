import os, json, time, base64, hashlib, requests
from typing import List, Dict, Any

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11435")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "pixtral")

def file_md5(path: str, chunk=1<<20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def ollama_vision(prompt: str, image_paths: List[str]) -> str:
    # Send image(s) + prompt to Pixtral/LLaVA
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [b64_image(p) for p in image_paths],
        "stream": False
    }
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response","")

def load_tracker(tracker_path: str) -> Dict[str, Any]:
    if not os.path.exists(tracker_path): return {}
    with open(tracker_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_tracker(tracker_path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(tracker_path), exist_ok=True)
    tmp = tracker_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, tracker_path)
