import os, tempfile
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
import numpy as np

from .ingest_images import incremental_ingest
from .face_store import ingest_faces, list_unlabeled, label_face, search_face_by_upload
from .redis_index import get_client, knn_images, ensure_images_schema
from .utils import ollama_vision

PHOTOS_DIR = os.getenv("PHOTOS_DIR", "/app/data/photos")
IMAGE_EMBED_DIM = int(os.getenv("IMAGE_EMBED_DIM", "512"))

app = FastAPI(title="Smart Image Knowledgebase", version="0.1.0")

class IngestReport(BaseModel):
    added: int
    skipped: int

class QueryText(BaseModel):
    question: str
    k: int = 6
    tag_filter: str | None = None
    describe_with_llm: bool = True

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/ingest_images", response_model=IngestReport)
def ingest_images():
    res = incremental_ingest()
    return IngestReport(**res)

@app.post("/ingest_faces")
def ingest_faces_api():
    res = ingest_faces()
    return res

@app.get("/faces/unlabeled")
def faces_unlabeled(limit: int = 50):
    return {"faces": list_unlabeled(limit=limit)}

@app.post("/label_face")
def label_face_api(face_id: str = Form(...), person: str = Form(...), propagate: bool = Form(True)):
    return label_face(face_id, person, propagate=propagate)

@app.post("/query_text")
def query_text(req: QueryText):
    # Use CLIP text->image retrieval stored in Redis (vector field)
    client = get_client()
    ensure_images_schema(client, dim=IMAGE_EMBED_DIM)  # idempotent
    # we reuse the text as a tag filter first if provided, else pure knn
    # For KNN, we need a vector; simplest approach here: let Pixtral describe top images for question
    # For robust KNN by text, add CLIP text encoder; to keep this endpoint light, we rely on tags + LLM
    # Hybrid: If tag_filter provided, use it; otherwise, just describe best candidates
    # We'll just ask LLM to answer using the top retrieved images by tags (if any), else all images (not ideal at scale).
    hits = knn_images(client, query_vec=(0*np.zeros((1,IMAGE_EMBED_DIM),dtype="float32")+1e-6), k=req.k, tag_filter=req.tag_filter)  # dummy vec; RediSearch requires a vector param
    paths = [os.path.join(PHOTOS_DIR, h["path"]) for h in hits if "path" in h]
    if req.describe_with_llm and paths:
        resp = ollama_vision(f"Answer the question using these images. Question: {req.question}", paths[:4])
        return {"answer": resp, "images": hits[:4]}
    return {"images": hits}

@app.post("/query_image")
def query_image(question: str = Form("Describe this image"), file: UploadFile = File(...)):
    # Save upload temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    # Send uploaded image + question to Pixtral
    resp = ollama_vision(question, [tmp_path])
    return {"answer": resp}
