from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable, Iterable

from dsqa.config import CONFIG
from dsqa.expand import expand_query

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over in-memory tokenized docs. Small enough to skip a dependency."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.n, 1)
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(set(doc))
        self.idf = {
            term: math.log((self.n - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()
        }
        self.doc_len = [len(d) for d in docs]
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: Iterable[str]) -> list[float]:
        out: list[float] = []
        for i, tf in enumerate(self.tf):
            dl = self.doc_len[i] or 1
            score = 0.0
            for term in query:
                freq = tf.get(term)
                if not freq:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score += idf * (freq * (self.k1 + 1)) / denom
            out.append(score)
        return out


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion. rank_lists are ordered ids, best first."""
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for i, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def _doc_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or f"{row.get('source')}:{row.get('page')}:{hash(row.get('text'))}")


def fuse_and_trim(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    by_id = {_doc_id(d): d for d in dense + sparse}
    fused = rrf(
        [[_doc_id(d) for d in dense], [_doc_id(d) for d in sparse]],
    )
    return [by_id[i] for i in fused if i in by_id][:top_k]


_reranker = None
_reranker_name: str | None = None


def rerank(
    query: str,
    docs: list[dict[str, Any]],
    top_k: int,
    predict: Callable[[list[tuple[str, str]]], list[float]] | None = None,
) -> list[dict[str, Any]]:
    if not docs:
        return []
    if predict is None:
        predict = _cross_encoder_predict
    pairs = [(query, str(d.get("text") or "")) for d in docs]
    scores = predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
    out: list[dict[str, Any]] = []
    for doc, score in ranked[:top_k]:
        row = dict(doc)
        row["rerank"] = float(score)
        out.append(row)
    return out


def _cross_encoder_predict(pairs: list[tuple[str, str]]) -> list[float]:
    global _reranker, _reranker_name
    from sentence_transformers import CrossEncoder

    if _reranker is None or _reranker_name != CONFIG.rerank_model:
        _reranker = CrossEncoder(CONFIG.rerank_model)
        _reranker_name = CONFIG.rerank_model
    scores = _reranker.predict(pairs)
    return [float(s) for s in scores]


def search(
    query: str,
    *,
    dense_fn: Callable[[str, int], list[dict[str, Any]]],
    all_docs_fn: Callable[[], list[dict[str, Any]]],
    top_k: int | None = None,
    use_bm25: bool | None = None,
    use_rerank: bool | None = None,
    use_expand: bool | None = None,
    rerank_pool: int | None = None,
    rerank_predict: Callable[[list[tuple[str, str]]], list[float]] | None = None,
) -> list[dict[str, Any]]:
    """Dense search, optional BM25+RRF, optional cross-encoder rerank. Flags default to CONFIG."""
    query = expand_query(query, enabled=use_expand)
    k = top_k if top_k is not None else CONFIG.top_k
    hybrid = CONFIG.use_bm25 if use_bm25 is None else use_bm25
    do_rerank = CONFIG.use_rerank if use_rerank is None else use_rerank
    pool = rerank_pool if rerank_pool is not None else CONFIG.rerank_pool
    fetch = max(k, pool if (hybrid or do_rerank) else k)

    dense = dense_fn(query, fetch)
    merged = dense
    if hybrid:
        corpus = all_docs_fn()
        if corpus:
            bm25 = BM25([tokenize(str(d.get("text") or "")) for d in corpus])
            scores = bm25.scores(tokenize(query))
            order = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)[:fetch]
            sparse = [corpus[i] for i in order]
            merged = fuse_and_trim(dense, sparse, fetch)
    if do_rerank:
        merged = rerank(query, merged[:fetch], k, predict=rerank_predict)
    return merged[:k]
