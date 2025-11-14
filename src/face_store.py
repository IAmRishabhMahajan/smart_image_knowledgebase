import os, uuid, ast
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import cv2

from .redis_index import get_client, ensure_faces_schema, upsert_face, FACE_PREFIX, knn_faces

PHOTOS_DIR       = os.getenv("PHOTOS_DIR", "/app/data/photos")
FACE_EMBED_DIM   = int(os.getenv("FACE_EMBED_DIM","512"))
FACE_SIM_THRESH  = float(os.getenv("FACE_SIM_THRESHOLD","0.60"))

class FaceEngine:
    def __init__(self):
        # CPU-friendly config
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0 if self._has_cuda() else -1, det_size=(640,640))

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def detect_embed(self, img_path: str):
        img = cv2.imread(img_path)
        if img is None:
            return []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.app.get(img)
        out = []
        for f in faces:
            # f.normed_embedding is L2-normalized 512-D
            emb = np.asarray(f.normed_embedding, dtype="float32")
            x1,y1,x2,y2 = map(int, f.bbox)
            out.append({"bbox":[x1,y1,x2,y2], "embedding": emb})
        return out

def ingest_faces() -> Dict[str, Any]:
    from .utils import file_md5

    client = get_client()
    ensure_faces_schema(client, dim=FACE_EMBED_DIM)
    eng = FaceEngine()

    unlabeled_dir = os.path.join(PHOTOS_DIR, "../faces/unlabeled")
    os.makedirs(unlabeled_dir, exist_ok=True)

    added = 0
    for k in client.scan_iter(match="img:*", count=500):
        path_raw = client.hget(k, "path")
        if not path_raw:
            continue

        rel = path_raw.decode() if isinstance(path_raw, bytes) else path_raw
        abs_path = os.path.join(PHOTOS_DIR, rel)

        # run InsightFace detection
        faces = eng.detect_embed(abs_path)
        if not faces:
            continue

        img = cv2.imread(abs_path)
        if img is None:
            continue

        for f in faces:
            face_id = f"f_{uuid.uuid4().hex[:10]}"
            x1, y1, x2, y2 = f["bbox"]

            # crop and save thumbnail
            face_crop = img[y1:y2, x1:x2]
            thumb_path = os.path.join(unlabeled_dir, f"{face_id}.jpg")
            try:
                Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).save(thumb_path)
            except Exception:
                continue

            meta = {
                "image_path": rel,
                "face_id": face_id,
                "person": "",
                "bbox": f["bbox"],
                "thumb_path": os.path.relpath(thumb_path, start=PHOTOS_DIR),
                "timestamp": int(client.hget(k, "timestamp") or 0),
            }

            upsert_face(client, f"{FACE_PREFIX}{face_id}", meta, f["embedding"])
            added += 1

    return {"faces_added": added}


def list_unlabeled(limit=50) -> List[Dict[str, Any]]:
    client = get_client()
    out = []
    for k in client.scan_iter(match="face:*", count=500):
        person = client.hget(k, "person")
        if person and person.decode() != "":
            continue
        face_id = client.hget(k, "face_id")
        image_path = client.hget(k, "image_path")
        bbox = client.hget(k, "bbox")
        thumb_path = client.hget(k, "thumb_path")
        out.append({
            "face_id": face_id.decode(),
            "image_path": image_path.decode(),
            "bbox": bbox.decode(),
            "thumbnail": f"/data/faces/unlabeled/{os.path.basename(thumb_path)}"
        })
        if len(out) >= limit:
            break
    return out


def label_face(face_id: str, person: str, propagate: bool = True) -> Dict[str, Any]:
    client = get_client()
    key = f"{FACE_PREFIX}{face_id}"
    if not client.exists(key):
        return {"updated": 0, "message": "face_id not found"}
    client.hset(key, mapping={"person": person})
    updated = 1

    if propagate:
        emb = client.hget(key,"embedding")
        if emb:
            vec = np.frombuffer(emb, dtype="float32")
            # find similar faces and label them if unlabeled
            neigh = knn_faces(client, vec.reshape(1,-1), k=10)
            for n in neigh:
                nkey = f"{FACE_PREFIX}{n['face_id']}"
                p = client.hget(nkey,"person")
                if not p or p.decode()=="":
                    client.hset(nkey, mapping={"person": person})
                    updated += 1

    return {"updated": updated, "person": person}

def search_face_by_upload(upload_path: str, k: int = 8) -> List[Dict[str, Any]]:
    # run InsightFace on uploaded image; if multiple faces, return nearest for each
    eng = FaceEngine()
    faces = eng.detect_embed(upload_path)
    client = get_client()
    results = []
    for f in faces:
        neigh = knn_faces(client, f["embedding"].reshape(1,-1), k=k)
        results.append({"bbox": f["bbox"], "matches": neigh})
    return results
