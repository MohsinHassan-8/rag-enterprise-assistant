"""ChromaDB wrapper: persistent store, validated batched upsert, semantic search."""
import chromadb
import numpy as np

DEFAULT_COLLECTION = "wiki_chunks"
DEFAULT_BATCH_SIZE = 2000  # well under Chroma's per-call batch limit


def get_collection(path, name: str = DEFAULT_COLLECTION, reset: bool = False):
    """Open (or create) a persistent collection that ranks by cosine distance."""
    client = chromadb.PersistentClient(
        path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
    )
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass  # collection did not exist yet
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_chunks(
    collection,
    ids,
    embeddings,
    documents,
    metadatas=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Validate, then upsert in batches. Re-running with the same ids is safe."""
    n = len(ids)
    if not (len(embeddings) == len(documents) == n):
        raise ValueError("ids, embeddings and documents must have the same length")
    if metadatas is not None and len(metadatas) != n:
        raise ValueError("metadatas must have the same length as ids")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n == 0:
        return {"upserted": 0, "skipped_empty": 0}
    if len(set(ids)) != n:
        raise ValueError("ids must be unique within one upsert call")

    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError("embeddings must be 2-D (n_chunks x dim)")
    if not np.isfinite(emb).all():
        raise ValueError("embeddings contain NaN or inf")

    keep = [i for i, d in enumerate(documents) if d and d.strip()]
    for start in range(0, len(keep), batch_size):
        idx = keep[start : start + batch_size]
        collection.upsert(
            ids=[ids[i] for i in idx],
            embeddings=emb[idx].tolist(),
            documents=[documents[i] for i in idx],
            metadatas=[metadatas[i] for i in idx] if metadatas is not None else None,
        )
    return {"upserted": len(keep), "skipped_empty": n - len(keep)}


def search(collection, query_embedding, n_results: int = 5, where: dict | None = None) -> list[dict]:
    """Return the nearest chunks as dicts, best first. Score is 1 - cosine distance."""
    if n_results <= 0:
        raise ValueError("n_results must be positive")
    total = collection.count()
    if total == 0:
        return []

    vec = np.asarray(query_embedding, dtype=np.float32).ravel()
    if vec.size == 0 or not np.isfinite(vec).all():
        raise ValueError("query_embedding must be a finite, non-empty vector")

    res = collection.query(
        query_embeddings=[vec.tolist()],
        n_results=min(n_results, total),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "rank": rank,
            "id": id_,
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "score": 1.0 - dist,
        }
        for rank, (id_, doc, meta, dist) in enumerate(
            zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            ),
            start=1,
        )
    ]

PLACE_FIELDS = ["place_1", "place_2", "place_3", "place_4", "place_5"]


def build_where(theme=None, topic_id=None, place=None):
    """Build a Chroma `where` filter from optional theme, topic id and place.

    A place matches if it is in any of the place_1..place_5 metadata slots.
    Returns None when no filter is given.
    """
    clauses = []
    if theme is not None:
        if not isinstance(theme, str) or not theme.strip():
            raise ValueError("theme must be a non-empty string")
        clauses.append({"theme": theme})
    if topic_id is not None:
        if isinstance(topic_id, bool) or not isinstance(topic_id, int):
            raise ValueError("topic_id must be an int")
        clauses.append({"topic_id": topic_id})
    if place is not None:
        if not isinstance(place, str) or not place.strip():
            raise ValueError("place must be a non-empty string")
        clauses.append({"$or": [{f: place} for f in PLACE_FIELDS]})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def update_metadata(collection, ids, metadatas, batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    """Replace metadata for existing ids in batches. Vectors and text are untouched."""
    n = len(ids)
    if len(metadatas) != n:
        raise ValueError("metadatas must have the same length as ids")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(set(ids)) != n:
        raise ValueError("ids must be unique")
    for m in metadatas:
        for k, v in m.items():
            if v is None or not isinstance(v, (str, int, float, bool)):
                raise ValueError("metadata value for %r must be str, int, float or bool" % k)
    for start in range(0, n, batch_size):
        collection.update(ids=ids[start : start + batch_size],
                          metadatas=metadatas[start : start + batch_size])
    return {"updated": n}
