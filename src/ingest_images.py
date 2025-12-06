# src/ingest_images.py

import os
import glob
import uuid
from typing import Dict, Any, List
import io
import base64

import numpy as np
from PIL import Image
import torch
import open_clip
import requests

from .redis_index import get_client, ensure_images_schema, upsert_image, IMG_PREFIX

PHOTOS_DIR = os.getenv("PHOTOS_DIR", "/app/data/photos")
TRACKER_PATH = os.getenv("TRACKER_PATH", "/app/data/.ingest_tracker.json")

# Ollama vision config
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama_smart_image_knowledgebase:11435")
VISION_MODEL = os.getenv("VISION_MODEL", "llava:latest")  # or "pixtral:latest"


# ---- Utility: MD5 + tracker ----

def _file_md5(path: str) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_tracker(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    import json

    with open(path, "r") as f:
        return json.load(f)


def _save_tracker(path: str, data: Dict[str, Any]) -> None:
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ---- CLIP encoder (unchanged embeddings) ----

class CLIPEncoder:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def encode_image(self, img: Image.Image) -> np.ndarray:
        t = self.preprocess(img).unsqueeze(0).to(self.device)
        z = self.model.encode_image(t)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.squeeze(0).cpu().float().numpy()

    @torch.no_grad()
    def rank_tags(self, img: Image.Image, labels: List[str]) -> List[str]:
        """
        Old fallback tagger: CLIP over a fixed label set.
        Used only if LLM tagging fails.
        """
        t = self.preprocess(img).unsqueeze(0).to(self.device)
        tokens = self.tokenizer([f"a photo of {l}" for l in labels]).to(self.device)
        image_feat = self.model.encode_image(t)
        text_feat = self.model.encode_text(tokens)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        logits = (image_feat @ text_feat.T).squeeze(0).cpu().numpy()
        idx = np.argsort(-logits)[:5]
        return [labels[i] for i in idx]


DEFAULT_FALLBACK_TAGS = [
    "people",
    "selfie",
    "indoor",
    "outdoor",
    "hill",
    "mountain",
    "beach",
    "sunset",
    "city",
    "office",
]


# ---- NEW: Vision LLM tag generator via Ollama ----

def vision_tags_from_ollama(img: Image.Image) -> List[str]:
    """
    Use a vision-capable model running in Ollama (e.g. llava, pixtral)
    to generate high-quality tags for an image.

    Protocol:
      - Send base64-encoded JPEG to /api/generate
      - Prompt the model to ONLY return a line like:
          TAGS: tag1, tag2, tag3, ...
      - Parse tags out and return as a list of strings.
    """
    # Convert image to JPEG bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        "You are tagging a personal photo library. "
        "Look at the image and produce a single line in the format:\n"
        "TAGS: tag1, tag2, tag3, ...\n"
        "Use short, lowercase nouns or short phrases. No extra text."
    )

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [b64_img],
        "stream": False,
    }

    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response", "") or ""

    # Find the line that starts with TAGS:
    tags_line = ""
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("tags:"):
            tags_line = line
            break

    if not tags_line:
        return []

    # Strip "TAGS:" and split by comma
    tags_str = tags_line.split(":", 1)[1]
    tags = [t.strip().lower() for t in tags_str.split(",") if t.strip()]

    # Deduplicate while preserving order
    deduped = []
    seen = set()
    for t in tags:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return deduped


# ---- MAIN: Incremental ingest using CLIP + Vision LLM tags ----

def incremental_ingest() -> Dict[str, Any]:
    """
    Walk PHOTOS_DIR, compute CLIP embeddings, and generate tags using
    Ollama vision model (Pixtral/LLaVA). If Ollama is unavailable,
    fall back to CLIP-based tag ranking over DEFAULT_FALLBACK_TAGS.
    """
    client = get_client()
    ensure_images_schema(client, dim=512)
    enc = CLIPEncoder()
    tracker = _load_tracker(TRACKER_PATH)

    added, skipped = 0, 0

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    for pat in patterns:
        for path in glob.glob(os.path.join(PHOTOS_DIR, "**", pat), recursive=True):
            md5 = _file_md5(path)
            prev = tracker.get(path)
            if prev and prev.get("hash") == md5:
                skipped += 1
                continue

            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue

            # 1) CLIP embedding (for vector search)
            emb = enc.encode_image(img)

            # 2) Tags: try vision LLM first, then fall back to CLIP tags
            try:
                llm_tags = vision_tags_from_ollama(img)
            except Exception:
                llm_tags = []

            if llm_tags:
                tags = llm_tags
            else:
                # fallback: old CLIP-based tag ranking
                tags = enc.rank_tags(img, DEFAULT_FALLBACK_TAGS)

            image_id = f"img_{uuid.uuid4().hex[:10]}"
            rel_path = os.path.relpath(path, PHOTOS_DIR)

            meta = {
                "image_id": image_id,
                "path": rel_path,
                "tags": tags,
                "person_names": [],
                "timestamp": 0,  # plug EXIF if/when needed
                "hash": md5,
            }

            key = f"{IMG_PREFIX}{image_id}"
            upsert_image(client, key, meta, emb)

            tracker[path] = {"hash": md5}
            added += 1

    _save_tracker(TRACKER_PATH, tracker)
    return {"added": added, "skipped": skipped}
