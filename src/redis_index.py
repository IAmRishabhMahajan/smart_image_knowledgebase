import os, numpy as np, redis
from typing import Dict, Any, List
from redis.commands.search.field import TagField, TextField, NumericField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search import Search



REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

IMAGES_INDEX = "images_idx"
FACES_INDEX  = "faces_idx"
IMG_PREFIX   = "img:"
FACE_PREFIX  = "face:"

def get_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

def ensure_images_schema(client, dim: int):
    try:
        Search(client, IMAGES_INDEX).info()
        return
    except Exception:
        pass
    schema = (
        TextField("path"),
        TagField("tags"),
        NumericField("lat"),
        NumericField("lon"),
        NumericField("timestamp"),
        VectorField("embedding", "FLAT", {
            "TYPE":"FLOAT32", "DIM":dim, "DISTANCE_METRIC":"COSINE"
        }),
        TextField("hash")  # for incremental ingestion
    )
    client.ft(IMAGES_INDEX).create_index(
        fields=schema,
        definition=IndexDefinition(prefix=[IMG_PREFIX], index_type=IndexType.HASH)
    )

def ensure_faces_schema(client, dim: int):
    try:
        Search(client, FACES_INDEX).info()
        return
    except Exception:
        pass
    schema = (
        TextField("image_path"),
        TextField("face_id"),
        TagField("person"),
        TextField("bbox"),
        VectorField("embedding", "FLAT", {
            "TYPE":"FLOAT32","DIM":dim,"DISTANCE_METRIC":"COSINE"
        }),
        NumericField("timestamp")
    )
    client.ft(FACES_INDEX).create_index(
        fields=schema,
        definition=IndexDefinition(prefix=[FACE_PREFIX], index_type=IndexType.HASH)
    )

def upsert_image(client, key: str, meta: Dict[str, Any], emb: np.ndarray):
    mapping = {
        "path": meta.get("path",""),
        "tags": ",".join(meta.get("tags",[])),
        "lat": float(meta.get("lat", 0.0)),
        "lon": float(meta.get("lon", 0.0)),
        "timestamp": int(meta.get("timestamp", 0)),
        "embedding": emb.astype("float32").tobytes(),
        "hash": meta.get("hash","")
    }
    client.hset(key, mapping=mapping)

def upsert_face(client, key: str, meta: Dict[str, Any], emb: np.ndarray):
    mapping = {
        "image_path": meta.get("image_path", ""),
        "face_id": meta.get("face_id", ""),
        "person": meta.get("person", ""),
        "bbox": str(meta.get("bbox", "")),
        "timestamp": int(meta.get("timestamp", 0)),
        "embedding": emb.astype("float32").tobytes(),
        "thumb_path": meta.get("thumb_path", ""),  
    }
    client.hset(key, mapping=mapping)


def knn_images(client, query_vec: np.ndarray, k: int = 8, tag_filter: str = None):
    # Always include *=> to make the query syntactically valid for RediSearch
    base = f"*=>[KNN {k} @embedding $vec AS score]"
    q = base if not tag_filter else f"@tags:{{{tag_filter}}}=>[KNN {k} @embedding $vec AS score]"

    try:
        # Try the modern client first
        res = client.ft(IMAGES_INDEX).search(
            q,
            query_params={"vec": query_vec.astype("float32").tobytes()},
            # dialect=2,
            return_fields=["path", "tags", "timestamp", "score", "hash"]
        )
    except TypeError:
        # Fallback for older client (no return_fields kwarg)
        res = client.ft(IMAGES_INDEX).search(
            q,
            query_params={"vec": query_vec.astype("float32").tobytes()}
            # ,dialect=2
        )

    out = []
    for d in getattr(res, "docs", []):
        out.append({
            "path": getattr(d, "path", ""),
            "tags": getattr(d, "tags", "").split(",") if getattr(d, "tags", "") else [],
            "timestamp": int(getattr(d, "timestamp", 0)),
            "score": float(getattr(d, "score", 0.0)),
            "hash": getattr(d, "hash", "")
        })
    return out


def knn_faces(client, query_vec: np.ndarray, k: int = 8, person: str = None):
    # valid RediSearch syntax for pure KNN search
    base = f"*=>[KNN {k} @embedding $vec AS score]"
    q = base if not person else f"@person:{{{person}}}=>[KNN {k} @embedding $vec AS score]"

    try:
        res = client.ft(FACES_INDEX).search(
            q,
            query_params={"vec": query_vec.astype("float32").tobytes()}
            #,dialect=2,
            ,return_fields=[
                "face_id", "image_path", "bbox", "person",
                "score", "timestamp", "thumb_path"
            ]
        )
    except TypeError:
        # fallback for clients that don’t accept return_fields keyword
        res = client.ft(FACES_INDEX).search(
            q,
            query_params={"vec": query_vec.astype("float32").tobytes()}
            #,dialect=2
        )

    out = []
    for d in getattr(res, "docs", []):
        out.append({
            "face_id": getattr(d, "face_id", ""),
            "image_path": getattr(d, "image_path", ""),
            "bbox": getattr(d, "bbox", ""),
            "person": getattr(d, "person", ""),
            "thumb_path": getattr(d, "thumb_path", ""),
            "score": float(getattr(d, "score", 0.0)),
            "timestamp": int(getattr(d, "timestamp", 0))
        })
    return out
