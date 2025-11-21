# src/redis_index.py

import os
from typing import Any, Dict, List

import numpy as np
import redis
from redis.commands.search.field import (
    TagField,
    TextField,
    NumericField,
    VectorField,
)
from redis.commands.search.index_definition import IndexDefinition, IndexType

# --- Redis config ---

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

IMAGES_INDEX = "images_idx"
FACES_INDEX = "faces_idx"

IMG_PREFIX = "img:"
FACE_PREFIX = "face:"


def get_client() -> redis.Redis:
    # binary mode; we decode manually where needed
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)


# ===================== IMAGES INDEX =====================

def ensure_images_schema(client: redis.Redis, dim: int) -> None:
    """
    Ensure images_idx exists with:
      - image_id, path, tags, person_names, timestamp, hash, embedding
    """
    try:
        client.ft(IMAGES_INDEX).info()
        return
    except Exception:
        pass

    schema = [
        TextField("image_id"),
        TextField("path"),
        TagField("tags"),
        TagField("person_names"),
        NumericField("timestamp"),
        TextField("hash"),
        VectorField(
            "embedding",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    ]

    client.ft(IMAGES_INDEX).create_index(
        fields=schema,
        definition=IndexDefinition(prefix=[IMG_PREFIX], index_type=IndexType.HASH),
    )


def upsert_image(client: redis.Redis, key: str, meta: Dict[str, Any], emb: np.ndarray) -> None:
    mapping = {
        "image_id": meta.get("image_id", ""),
        "path": meta.get("path", ""),
        "tags": ",".join(meta.get("tags", [])),
        "person_names": ",".join(meta.get("person_names", [])),
        "timestamp": int(meta.get("timestamp", 0)),
        "hash": meta.get("hash", ""),
        "embedding": emb.astype("float32").tobytes(),
    }
    client.hset(key, mapping=mapping)


# ===================== FACES INDEX =====================

def ensure_faces_schema(client: redis.Redis, dim: int) -> None:
    """
    Ensure faces_idx exists with:
      - face_id, image_id, bbox, person, thumb_path, timestamp, embedding
    """
    try:
        client.ft(FACES_INDEX).info()
        return
    except Exception:
        pass

    schema = [
        TextField("face_id"),
        TextField("image_id"),
        TextField("bbox"),
        TextField("person"),
        TextField("thumb_path"),
        NumericField("timestamp"),
        VectorField(
            "embedding",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    ]

    client.ft(FACES_INDEX).create_index(
        fields=schema,
        definition=IndexDefinition(prefix=[FACE_PREFIX], index_type=IndexType.HASH),
    )


def upsert_face(client: redis.Redis, key: str, meta: Dict[str, Any], emb: np.ndarray) -> None:
    mapping = {
        "face_id": meta.get("face_id", ""),
        "image_id": meta.get("image_id", ""),
        "bbox": str(meta.get("bbox", "")),
        "person": meta.get("person", ""),
        "thumb_path": meta.get("thumb_path", ""),
        "timestamp": int(meta.get("timestamp", 0)),
        "embedding": emb.astype("float32").tobytes(),
    }
    client.hset(key, mapping=mapping)


# ===================== KNN HELPERS =====================

def knn_images(
    client: redis.Redis,
    query_vec: np.ndarray,
    k: int = 8,
    tag_filter: str | None = None,
    person_filter: str | None = None,
) -> List[Dict[str, Any]]:
    """
    KNN over images_idx using CLIP embeddings.

    - query_vec: (dim,) or (1, dim) float32
    - tag_filter: optional @tags: filter
    - person_filter: optional @person_names: filter
    """
    base_knn = f"[KNN {k} @embedding $vec AS score]"

    filters = []

    if tag_filter:
        filters.append(f"@tags:{{{tag_filter}}}")

    if person_filter:
        filters.append(f"@person_names:{{{person_filter}}}")

    if filters:
        filter_str = "(" + " ".join(filters) + ")=>"
        q = f"{filter_str}[KNN {k} @embedding $vec AS score]"
    else:
        q = f"*=>[KNN {k} @embedding $vec AS score]"


    try:
        res = client.ft(IMAGES_INDEX).search(
            q,
            query_params={"vec": query_vec.astype("float32").tobytes()},
            return_fields=[
                "image_id",
                "path",
                "tags",
                "person_names",
                "timestamp",
                "score",
                "hash",
            ],
            # dialect=2,
        )
    except TypeError:
        res = client.ft(IMAGES_INDEX).search(
            q,
            query_params={"vec": query_vec.astype("float32").tobytes()},
            # dialect=2,
        )

    out: List[Dict[str, Any]] = []
    for d in getattr(res, "docs", []):
        tags_raw = getattr(d, "tags", "") or ""
        persons_raw = getattr(d, "person_names", "") or ""
        out.append(
            {
                "image_id": getattr(d, "image_id", ""),
                "path": getattr(d, "path", ""),
                "tags": tags_raw.split(",") if tags_raw else [],
                "person_names": persons_raw.split(",") if persons_raw else [],
                "timestamp": int(getattr(d, "timestamp", 0)),
                "score": float(getattr(d, "score", 0.0)),
                "hash": getattr(d, "hash", ""),
            }
        )
    return out


def knn_faces(
    client: redis.Redis,
    query_vec: np.ndarray,
    k: int = 8,
    person: str | None = None,
) -> List[Dict[str, Any]]:
    """
    KNN over faces_idx using face embeddings.
    """
    base_knn = f"[KNN {k} @embedding $vec AS score]"

    if person:
        q = f"@person:{{{person}}}=>{base_knn}"
    else:
        q = f"*=>{base_knn}"

    try:
        res = client.ft(FACES_INDEX).search(
            q,
            query_params={"vec": query_vec.astype("float32").tobytes()},
            return_fields=[
                "face_id",
                "image_id",
                "bbox",
                "person",
                "thumb_path",
                "score",
                "timestamp",
            ],
            # dialect=2,
        )
    except TypeError:
        res = client.ft(FACES_INDEX).search(
            q,
            query_params={"vec": query_vec.astype("float32").tobytes()},
            # dialect=2,
        )

    out: List[Dict[str, Any]] = []
    for d in getattr(res, "docs", []):
        out.append(
            {
                "face_id": getattr(d, "face_id", ""),
                "image_id": getattr(d, "image_id", ""),
                "bbox": getattr(d, "bbox", ""),
                "person": getattr(d, "person", ""),
                "thumb_path": getattr(d, "thumb_path", ""),
                "score": float(getattr(d, "score", 0.0)),
                "timestamp": int(getattr(d, "timestamp", 0)),
            }
        )
    return out
