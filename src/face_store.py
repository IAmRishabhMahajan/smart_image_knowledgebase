# src/face_store.py

import os
import uuid
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

from .redis_index import (
    get_client,
    ensure_faces_schema,
    upsert_face,
    FACE_PREFIX,
    IMG_PREFIX,
    knn_faces,
)

PHOTOS_DIR = os.getenv("PHOTOS_DIR", "/app/data/photos")


def ingest_faces() -> Dict[str, Any]:
    """
    For each image in images_idx:
      - load image from disk
      - detect faces
      - store face embedding + metadata in faces_idx
      - save thumbnail crops for UI
    """
    client = get_client()
    ensure_faces_schema(client, dim=512)

    app = FaceAnalysis(name="buffalo_l")
    # ctx_id=-1 -> CPU; 0 if you have GPU wired
    app.prepare(ctx_id=-1)

    unlabeled_dir = os.path.join(PHOTOS_DIR, "../faces/unlabeled")
    os.makedirs(unlabeled_dir, exist_ok=True)

    faces_added = 0

    for key in client.scan_iter(match=f"{IMG_PREFIX}*"):
        image_id_raw = client.hget(key, "image_id")
        path_raw = client.hget(key, "path")
        ts_raw = client.hget(key, "timestamp")

        if not image_id_raw or not path_raw:
            continue

        image_id = image_id_raw.decode()
        rel_path = path_raw.decode()
        ts = int(ts_raw) if ts_raw else 0

        abs_path = os.path.join(PHOTOS_DIR, rel_path)
        img = cv2.imread(abs_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = app.get(rgb)

        for det in detections:
            face_id = f"f_{uuid.uuid4().hex[:10]}"
            x1, y1, x2, y2 = map(int, det.bbox)

            # crop + save thumbnail
            crop = img[y1:y2, x1:x2]
            thumb_path = os.path.join(unlabeled_dir, f"{face_id}.jpg")
            try:
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(thumb_path)
            except Exception:
                continue

            emb = np.array(det.normed_embedding, dtype="float32")

            meta = {
                "face_id": face_id,
                "image_id": image_id,
                "bbox": [x1, y1, x2, y2],
                "person": "",
                "thumb_path": thumb_path,
                "timestamp": ts,
            }

            face_key = f"{FACE_PREFIX}{face_id}"
            upsert_face(client, face_key, meta, emb)
            faces_added += 1

    return {"faces_added": faces_added}


def _update_image_person_names(
    client, image_id: str, person: str
) -> None:
    """
    Append person name into images_idx.person_names without duplicates.
    """
    img_key = f"{IMG_PREFIX}{image_id}"
    existing = client.hget(img_key, "person_names")

    names: set[str] = set()
    if existing:
        raw = existing.decode()
        for n in raw.split(","):
            n = n.strip()
            if n:
                names.add(n)

    names.add(person)
    client.hset(img_key, "person_names", ",".join(sorted(names)))


def label_face(face_id: str, person: str, propagate: bool = True) -> Dict[str, Any]:
    """
    Label a face (face_id -> person name).
    Also propagate to visually similar faces, and update images.person_names.
    """
    client = get_client()
    key = f"{FACE_PREFIX}{face_id}"

    if not client.exists(key):
        return {"updated": 0, "message": "face_id not found"}

    # 1) Label the primary face
    client.hset(key, mapping={"person": person})
    updated = 1

    image_id_raw = client.hget(key, "image_id")
    if image_id_raw:
        image_id = image_id_raw.decode()
        _update_image_person_names(client, image_id, person)

    # 2) Propagate to similar faces
    if propagate:
        emb_raw = client.hget(key, "embedding")
        if emb_raw:
            vec = np.frombuffer(emb_raw, dtype="float32")
            neigh = knn_faces(client, vec.reshape(1, -1), k=10)

            for n in neigh:
                nkey = f"{FACE_PREFIX}{n['face_id']}"
                p = client.hget(nkey, "person")

                # only label unlabeled faces
                if not p or (isinstance(p, bytes) and p.decode() == ""):
                    client.hset(nkey, mapping={"person": person})
                    updated += 1

                    n_img_id_raw = client.hget(nkey, "image_id")
                    if n_img_id_raw:
                        img_id2 = n_img_id_raw.decode()
                        _update_image_person_names(client, img_id2, person)

    return {"updated": updated, "person": person}


def list_unlabeled(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Return a sample of unlabeled faces for UI to label.
    """
    client = get_client()
    out: List[Dict[str, Any]] = []

    for key in client.scan_iter(match=f"{FACE_PREFIX}*"):
        person_raw = client.hget(key, "person")
        if person_raw:
            p = person_raw.decode()
            if p.strip():
                continue  # already labeled

        face_id_raw = client.hget(key, "face_id")
        image_id_raw = client.hget(key, "image_id")
        bbox_raw = client.hget(key, "bbox")
        thumb_raw = client.hget(key, "thumb_path")

        out.append(
            {
                "face_id": face_id_raw.decode() if face_id_raw else "",
                "image_id": image_id_raw.decode() if image_id_raw else "",
                "bbox": bbox_raw.decode() if bbox_raw else "",
                "thumb_path": thumb_raw.decode() if thumb_raw else "",
            }
        )

        if len(out) >= limit:
            break

    return out
