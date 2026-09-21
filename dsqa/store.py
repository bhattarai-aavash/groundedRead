from __future__ import annotations

from typing import Any

from dsqa.config import CONFIG
from dsqa.ingest import Chunk
from dsqa.llm import embed
from dsqa.retrieve import search


def _client() -> Any:
    import chromadb

    return chromadb.PersistentClient(path=CONFIG.chroma_dir)


def _collection() -> Any:
    # No embedding_function: we always pass vectors computed by llm.embed().
    return _client().get_or_create_collection(
        name=CONFIG.collection,
        metadata={"hnsw:space": "cosine"},
    )


def _row(doc_id: str, text: str, meta: dict[str, Any] | None, distance: float | None = None) -> dict[str, Any]:
    meta = meta or {}
    return {
        "id": doc_id,
        "text": text,
        "source": str(meta.get("source", "")),
        "page": int(meta.get("page", 0) or 0),
        "distance": distance,
        "context": str(meta.get("context", "")),
    }


def upsert(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    col = _collection()
    texts = [c.embed_text for c in chunks]
    col.upsert(
        ids=[c.id for c in chunks],
        embeddings=embed(texts),
        documents=[c.text for c in chunks],
        metadatas=[
            {"source": c.source, "page": int(c.page), "context": c.context} for c in chunks
        ],
    )
    return len(chunks)


def _dense(text: str, n: int) -> list[dict[str, Any]]:
    col = _collection()
    n_total = col.count()
    if n_total == 0 or not text.strip():
        return []
    n = min(n, n_total)
    res = col.query(
        query_embeddings=embed([text]),
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    ids = res["ids"][0]
    docs = res["documents"][0] if res["documents"] else [""] * len(ids)
    metas = res["metadatas"][0] if res["metadatas"] else [{}] * len(ids)
    dists = res["distances"][0] if res["distances"] else [0.0] * len(ids)
    return [_row(i, d or "", m, float(dist) if dist is not None else None) for i, d, m, dist in zip(ids, docs, metas, dists)]


def all_docs() -> list[dict[str, Any]]:
    col = _collection()
    if col.count() == 0:
        return []
    res = col.get(include=["documents", "metadatas"])
    ids = res.get("ids") or []
    docs = res.get("documents") or [""] * len(ids)
    metas = res.get("metadatas") or [{}] * len(ids)
    return [_row(i, d or "", m) for i, d, m in zip(ids, docs, metas)]


def query(
    text: str,
    k: int | None = None,
    use_bm25: bool | None = None,
    use_rerank: bool | None = None,
    use_expand: bool | None = None,
) -> list[dict[str, Any]]:
    return search(
        text,
        dense_fn=_dense,
        all_docs_fn=all_docs,
        top_k=k,
        use_bm25=use_bm25,
        use_rerank=use_rerank,
        use_expand=use_expand,
    )


def count() -> int:
    return int(_collection().count())


def reset() -> None:
    client = _client()
    try:
        client.delete_collection(CONFIG.collection)
    except Exception:
        pass
    client.get_or_create_collection(
        name=CONFIG.collection,
        metadata={"hnsw:space": "cosine"},
    )
