# src/ingest_images.py

import os
import glob
import uuid
from typing import Dict, Any

import numpy as np
from PIL import Image
import torch
import open_clip

from .redis_index import get_client, ensure_images_schema, upsert_image, IMG_PREFIX

PHOTOS_DIR = os.getenv("PHOTOS_DIR", "/app/data/photos")
TRACKER_PATH = os.getenv("TRACKER_PATH", "/app/data/.ingest_tracker.json")

DEFAULT_TAGS = [
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
    def rank_tags(self, img: Image.Image, labels: list[str]) -> list[str]:
        t = self.preprocess(img).unsqueeze(0).to(self.device)
        tokens = self.tokenizer([f"a photo of {l}" for l in labels]).to(self.device)
        image_feat = self.model.encode_image(t)
        text_feat = self.model.encode_text(tokens)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        logits = (image_feat @ text_feat.T).squeeze(0).cpu().numpy()
        idx = np.argsort(-logits)[:5]
        return [labels[i] for i in idx]


def incremental_ingest() -> Dict[str, Any]:
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

            emb = enc.encode_image(img)
            tags = enc.rank_tags(img, DEFAULT_TAGS)

            image_id = f"img_{uuid.uuid4().hex[:10]}"
            rel_path = os.path.relpath(path, PHOTOS_DIR)

            meta = {
                "image_id": image_id,
                "path": rel_path,
                "tags": tags,
                "person_names": [],
                "timestamp": 0,  # you can plug EXIF datetime later
                "hash": md5,
            }

            key = f"{IMG_PREFIX}{image_id}"
            upsert_image(client, key, meta, emb)

            tracker[path] = {"hash": md5}
            added += 1

    _save_tracker(TRACKER_PATH, tracker)
    return {"added": added, "skipped": skipped}
