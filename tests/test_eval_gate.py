from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TypedDict

import pytest

from dsqa.config import CONFIG
from dsqa.ingest import Chunk, _chunk_id
from dsqa.retrieve import tokenize
from dsqa.store import query, reset, upsert
from eval.run_eval import judge_score, retrieval_metrics, run


class Row(TypedDict):
    id: str
    question: str
    gold_sources: list[str]
    gold_pages: list[int]
    gold_answer: str
    must_include: list[str]


# Ten synthetic papers. Vectors are bag-of-token hashes so CI never downloads MiniLM.
CORPUS = [
    ("raft.pdf", 5, "Raft elects a leader with a randomized election timeout. A candidate that wins a majority of votes in a term becomes leader."),
    ("spanner.pdf", 5, "TrueTime returns an interval guaranteed to contain absolute time. Spanner commit-wait uses TrueTime for external consistency."),
    ("dynamo.pdf", 5, "Dynamo partitions data with consistent hashing. Keys and nodes sit on a ring; virtual nodes balance load."),
    ("gfs.pdf", 2, "The Google File System splits files into 64 MB chunks. A single master stores the namespace; chunkservers hold replicas."),
    ("paxos.pdf", 7, "A Paxos proposer sends a prepare request numbered n. After a majority of acceptors reply it may send an accept request."),
    ("crdt.pdf", 1, "A conflict-free replicated data type merges concurrent updates without consensus because operations commute."),
    ("hlc.pdf", 1, "A hybrid logical clock combines NTP physical time with a logical counter so happened-before is preserved."),
    ("cap.pdf", 1, "The CAP theorem says a partitioned asynchronous network cannot provide both linearizable consistency and availability."),
    ("kafka.pdf", 2, "Kafka topics are partitioned logs. Consumers read by offset rather than the broker tracking per-message acks."),
    ("chubby.pdf", 1, "Chubby is a lock service. Replicas elect a master and keep metadata consistent with Paxos."),
]

QUESTIONS: list[Row] = [
    {"id": "raft", "question": "How does Raft elect a leader?", "gold_sources": ["raft.pdf"], "gold_pages": [5], "gold_answer": "majority term", "must_include": ["majority"]},
    {"id": "truetime", "question": "What is TrueTime in Spanner?", "gold_sources": ["spanner.pdf"], "gold_pages": [5], "gold_answer": "TrueTime interval", "must_include": ["TrueTime"]},
    {"id": "dynamo", "question": "How does Dynamo partition data?", "gold_sources": ["dynamo.pdf"], "gold_pages": [5], "gold_answer": "consistent hashing", "must_include": ["consistent hashing"]},
    {"id": "gfs", "question": "How does GFS store files?", "gold_sources": ["gfs.pdf"], "gold_pages": [2], "gold_answer": "chunks master", "must_include": ["master"]},
    {"id": "paxos", "question": "How does a Paxos proposer prepare?", "gold_sources": ["paxos.pdf"], "gold_pages": [7], "gold_answer": "prepare accept", "must_include": ["prepare"]},
    {"id": "crdt", "question": "What is a conflict-free replicated data type?", "gold_sources": ["crdt.pdf"], "gold_pages": [1], "gold_answer": "conflict-free", "must_include": ["conflict-free"]},
    {"id": "hlc", "question": "What is a hybrid logical clock?", "gold_sources": ["hlc.pdf"], "gold_pages": [1], "gold_answer": "NTP logical", "must_include": ["NTP"]},
    {"id": "cap", "question": "What trade-off does the CAP theorem state?", "gold_sources": ["cap.pdf"], "gold_pages": [1], "gold_answer": "partition availability", "must_include": ["partition"]},
    {"id": "kafka", "question": "How do Kafka consumers read messages?", "gold_sources": ["kafka.pdf"], "gold_pages": [2], "gold_answer": "offset partition", "must_include": ["offset"]},
    {"id": "chubby", "question": "How does Chubby keep replicas consistent?", "gold_sources": ["chubby.pdf"], "gold_pages": [1], "gold_answer": "Paxos lock", "must_include": ["Paxos"]},
]

HIT_AT_K_MIN = 0.8


def fake_embed(texts: list[str]) -> list[list[float]]:
    dim = 64
    out: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        for tok in tokenize(text):
            vec[hash(tok) % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        out.append([x / norm for x in vec])
    return out


class GateLLM:
    def chat(self, system: str, user: str) -> str:
        if "grounding judge" in system.lower() or '"grounded"' in system:
            return '{"grounded": true, "reason": "supported", "missing": ""}'
        if "dsqa supervisor" in system.lower():
            if "Held excerpts (0)" in user:
                return '{"next": "retriever", "reason": "need excerpts"}'
            return '{"next": "writer", "reason": "enough"}'
        if "retriever specialist" in system.lower():
            q = next((row["question"] for row in QUESTIONS if row["question"] in user), "raft")
            return json.dumps({"query": q, "reason": "search"})
        if "grade a student" in system.lower() or '"correct"' in system:
            return '{"correct": true}'
        if "INSUFFICIENT_CONTEXT" in user and "chocolate" in user.lower():
            return "INSUFFICIENT_CONTEXT"
        # Copy a distinctive token from the first excerpt so keyword_acc can pass.
        for row in QUESTIONS:
            if row["question"] in user:
                token = row["must_include"][0]
                source = row["gold_sources"][0]
                page = row["gold_pages"][0]
                return f"{token} is described in the excerpt. [{source} p.{page}]"
        return "INSUFFICIENT_CONTEXT"


@pytest.fixture
def indexed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dsqa.store.embed", fake_embed)
    monkeypatch.setattr("dsqa.llm.embed", fake_embed)
    CONFIG.use_bm25 = True
    CONFIG.use_rerank = False
    CONFIG.grounding_check = True
    reset()
    chunks = [
        Chunk(
            id=_chunk_id(source, page, text),
            text=text,
            source=source,
            page=page,
            embed_text=text,
        )
        for source, page, text in CORPUS
    ]
    assert upsert(chunks) == 10


def test_ci_retrieval_gate(indexed: None, tmp_path: Path) -> None:
    """Fail the build if hybrid retrieval regresses on the 10-question synthetic set."""
    dataset = tmp_path / "ci.jsonl"
    dataset.write_text(
        "\n".join(__import__("json").dumps(row) for row in QUESTIONS) + "\n",
        encoding="utf-8",
    )
    hits = []
    for row in QUESTIONS:
        retrieved = query(row["question"], use_bm25=True, use_rerank=False)
        gold = {(row["gold_sources"][0], int(row["gold_pages"][0]))}
        metrics = retrieval_metrics(retrieved, gold)
        hits.append(metrics["hit@k"])
    hit_at_k = sum(hits) / len(hits)
    assert hit_at_k >= HIT_AT_K_MIN, f"hit@k={hit_at_k:.3f} dropped below {HIT_AT_K_MIN}"


def test_ci_fake_llm_eval_gate(indexed: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dsqa.graph import _compiled

    monkeypatch.setattr("dsqa.graph.get_llm", lambda: GateLLM())
    monkeypatch.setattr("eval.run_eval.get_llm", lambda: GateLLM())
    CONFIG.grounding_check = False
    _compiled.clear()
    dataset = tmp_path / "ci.jsonl"
    dataset.write_text(
        "\n".join(__import__("json").dumps(row) for row in QUESTIONS) + "\n",
        encoding="utf-8",
    )
    result = run(
        dataset,
        tag="ci-gate",
        no_judge=True,
        retrieval_only=False,
        use_bm25=True,
        use_rerank=False,
    )
    summary = result["summary"]
    assert summary["hit@k"] >= HIT_AT_K_MIN, summary
    assert summary["keyword_acc"] >= 0.8, summary


def test_judge_fail_closed_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[int] = []

    def boom(*_args: object, **_kwargs: object) -> float:
        called.append(1)
        return 1.0

    monkeypatch.setattr("eval.run_eval.judge_correct", boom)
    assert (
        judge_score("q", "gold", "majority vote elects a leader", ["term", "majority"], False)
        == 0.0
    )
    assert called == []
    assert judge_score("q", "gold", "term and majority", ["term", "majority"], False) == 1.0
    assert called == [1]
    assert judge_score("q", "gold", "majority only", ["term"], True) is None
