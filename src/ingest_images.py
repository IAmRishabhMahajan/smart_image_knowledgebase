import os, glob, json
from typing import Dict, Any, List
from PIL import Image
import numpy as np
import torch, open_clip
from tqdm import tqdm

from .utils import file_md5, load_tracker, save_tracker
from .redis_index import get_client, ensure_images_schema, upsert_image, IMG_PREFIX

PHOTOS_DIR   = os.getenv("PHOTOS_DIR", "/app/data/photos")
TRACKER_PATH = os.getenv("TRACKER_PATH", "/app/vectorstore/tracker.json")
CLIP_MODEL   = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "openai")
IMAGE_EMBED_DIM = int(os.getenv("IMAGE_EMBED_DIM", "512"))

# simple zero-shot label set (expand anytime)
DEFAULT_TAGS = ["person","people","office","meeting","indoor","outdoor",
                "document","whiteboard","computer","dashboard","beach","food","car","animal","diagram","chart"]

class CLIPEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, img: Image.Image) -> np.ndarray:
        t = self.preprocess(img).unsqueeze(0).to(self.device)
        z = self.model.encode_image(t)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.squeeze(0).detach().cpu().float().numpy()

    @torch.no_grad()
    def rank_tags(self, img: Image.Image, labels: List[str]) -> List[str]:
        # zero-shot scoring
        t = self.preprocess(img).unsqueeze(0).to(self.device)
        tok = self.tokenizer([f"a photo of {l}" for l in labels]).to(self.device)
        i_feat = self.model.encode_image(t)
        t_feat = self.model.encode_text(tok)
        i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
        t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        logits = (i_feat @ t_feat.T).squeeze(0).detach().cpu().numpy()
        idx = np.argsort(-logits)[:5]
        return [labels[i] for i in idx]

def iter_images(root: str):
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG","*.JPEG"):
        for p in glob.glob(os.path.join(root, "**", ext), recursive=True):
            yield p

def incremental_ingest() -> Dict[str, Any]:
    client = get_client()
    ensure_images_schema(client, dim=IMAGE_EMBED_DIM)
    enc = CLIPEncoder()
    tracker = load_tracker(TRACKER_PATH)

    added, skipped = 0, 0
    for path in tqdm(list(iter_images(PHOTOS_DIR)), desc="Ingesting images"):
        h = file_md5(path)
        rec = tracker.get(path)
        if rec and rec.get("hash") == h:
            skipped += 1
            continue

        # load image
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        # embeddings + tags
        emb = enc.encode_image(img)
        tags = enc.rank_tags(img, DEFAULT_TAGS)

        # metadata
        from .utils_metadata import extract_exif
        ex = extract_exif(path)
        ts = 0
        if "datetime" in ex:
            # best-effort YYYYMMDD for numeric filter; keep full in Redis if needed
            ts = int(ex["datetime"].replace("-","").replace(":","").replace("T","")[:8])

        meta = {
            "path": os.path.relpath(path, PHOTOS_DIR),
            "tags": tags,
            "lat": ex.get("lat", 0.0),
            "lon": ex.get("lon", 0.0),
            "timestamp": ts,
            "hash": h
        }

        key = f"{IMG_PREFIX}{path}"
        upsert_image(client, key, meta, emb)

        tracker[path] = {"hash": h}
        added += 1

    save_tracker(TRACKER_PATH, tracker)
    return {"added": added, "skipped": skipped}
