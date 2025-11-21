# src/app.py

import os
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from .ingest_images import incremental_ingest
from .face_store import ingest_faces, list_unlabeled, label_face
from .redis_index import get_client, knn_images

import torch
import open_clip

app = FastAPI(title="Smart Image Knowledgebase")

PHOTOS_DIR = os.getenv("PHOTOS_DIR", "/app/data/photos")


# ========== CLIP text encoder for queries ==========

class TextEncoder:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        z = self.model.encode_text(tokens)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.squeeze(0).cpu().float().numpy()


text_encoder = TextEncoder()


# ========== Pydantic models ==========

class QueryTextRequest(BaseModel):
    question: str
     # kept for later, currently ignored


# ========== API endpoints ==========

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest_images")
def ingest_images_endpoint():
    res = incremental_ingest()
    return res


@app.post("/ingest_faces")
def ingest_faces_endpoint():
    res = ingest_faces()
    return res


@app.get("/faces/unlabeled")
def faces_unlabeled(limit: int = 20):
    return {"faces": list_unlabeled(limit=limit)}


@app.post("/label_face")
def label_face_endpoint(face_id: str, person: str, propagate: bool = True):
    res = label_face(face_id, person, propagate=propagate)
    return res


@app.post("/query_text")
def query_text(req: QueryTextRequest):
    """
    Smart unified search:
      - Detect person names inside query
      - Detect tags
      - Run CLIP text embedding
      - Return ONLY image hits (no metadata about detection)
    """

    query = req.question
    client = get_client()

    # ===== 1. Detect known person names =====
    known_people = set()
    for key in client.scan_iter(match="face:*"):
        p = client.hget(key, "person")
        if p:
            p = p.decode().strip().lower()
            if p:
                known_people.add(p)

    q_lower = query.lower()
    persons_in_query = [p for p in known_people if p in q_lower]
    person_filter = persons_in_query[0] if persons_in_query else None


    # ===== 2. Detect scene tags =====
    SCENE_TAGS = [
        "hill","mountain","sunset","beach","forest","night","day",
        "selfie","outdoor","indoor","people","friends","snow","city"
    ]
    matched_tags = [t for t in SCENE_TAGS if t in q_lower]
    tag_filter = matched_tags[0] if matched_tags else None


    # ===== 3. CLIP embedding =====
    vec = text_encoder.encode(query)


    # ===== 4. Perform KNN search =====
    results = knn_images(
        client,
        vec.reshape(1, -1),
        k=8,
        tag_filter=tag_filter,
        person_filter=person_filter,
    )

    # ===== 5. Return **images only** =====
    return {"results": results}
