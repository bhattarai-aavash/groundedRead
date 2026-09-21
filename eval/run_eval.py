from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dsqa.config import CONFIG  # noqa: E402
from dsqa.graph import ask, parse_json  # noqa: E402
from dsqa.grounding import missing_required_terms  # noqa: E402
from dsqa.llm import get_llm  # noqa: E402
from dsqa.prompts import JUDGE_SYSTEM, judge_user  # noqa: E402
from dsqa.store import query  # noqa: E402


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pairs(sources: list[str], pages: list[int]) -> set[tuple[str, int]]:
    if not sources:
        return set()
    if len(pages) == len(sources):
        return {(s, int(p)) for s, p in zip(sources, pages)}
    return {(s, int(p)) for s in sources for p in pages}


def retrieval_metrics(
    retrieved: list[dict[str, Any]],
    gold: set[tuple[str, int]],
) -> dict[str, float]:
    if not gold:
        return {"hit@k": float("nan"), "recall@k": float("nan"), "mrr": float("nan")}
    ranked = [(str(d.get("source", "")), int(d.get("page", 0))) for d in retrieved]
    hits = [i + 1 for i, pair in enumerate(ranked) if pair in gold]
    hit = 1.0 if hits else 0.0
    found = {pair for pair in ranked if pair in gold}
    recall = len(found) / len(gold)
    mrr = 1.0 / hits[0] if hits else 0.0
    return {"hit@k": hit, "recall@k": recall, "mrr": mrr}


def keyword_acc(answer: str, must_include: list[str]) -> float:
    return 1.0 if not missing_required_terms(answer, must_include) else 0.0


def judge_score(
    question: str,
    gold_answer: str,
    answer: str,
    must_include: list[str],
    no_judge: bool,
) -> float | None:
    """Fail closed on missing must-include terms before calling the LLM judge."""
    if no_judge:
        return None
    if missing_required_terms(answer, must_include):
        return 0.0
    return judge_correct(question, gold_answer, answer)


def judge_correct(question: str, gold_answer: str, answer: str) -> float:
    llm = get_llm()
    raw = llm.chat(JUDGE_SYSTEM, judge_user(question, gold_answer, answer))
    try:
        data = parse_json(raw)
        return 1.0 if data.get("correct") else 0.0
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0


def _mean(xs: list[float]) -> float:
    vals = [x for x in xs if x == x]
    return sum(vals) / len(vals) if vals else 0.0


def _span_totals(spans: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "latency_ms": sum(float(s.get("ms") or 0) for s in spans),
        "tokens_in": sum(float(s.get("tokens_in") or 0) for s in spans),
        "tokens_out": sum(float(s.get("tokens_out") or 0) for s in spans),
        "cost_usd": sum(float(s.get("cost_usd") or 0) for s in spans),
    }


def run(
    dataset_path: Path,
    tag: str,
    no_judge: bool,
    retrieval_only: bool = False,
    use_bm25: bool | None = None,
    use_rerank: bool | None = None,
    use_expand: bool | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = load_dataset(dataset_path)
    if limit is not None:
        rows = rows[:limit]
    started = time.perf_counter()
    per_row: list[dict[str, Any]] = []
    for row in rows:
        question = str(row["question"])
        gold_sources = list(row.get("gold_sources") or [])
        gold_pages = [int(p) for p in (row.get("gold_pages") or [])]
        gold = _pairs(gold_sources, gold_pages)
        negative = len(gold_sources) == 0

        retrieved = query(question, use_bm25=use_bm25, use_rerank=use_rerank, use_expand=use_expand)
        ret = retrieval_metrics(retrieved, gold)

        answer = ""
        attempts = 0
        kw = 0.0
        abstained = False
        judge = None
        trace: list[str] = []
        spans: list[dict[str, Any]] = []
        must_include = list(row.get("must_include") or [])
        if not retrieval_only:
            state = ask(question, required_terms=must_include)
            answer = str(state.get("answer") or "")
            attempts = int(state.get("attempts") or 0)
            kw = keyword_acc(answer, must_include)
            abstained = "INSUFFICIENT_CONTEXT" in answer
            judge = judge_score(
                question, str(row.get("gold_answer") or ""), answer, must_include, no_judge
            )
            trace = list(state.get("trace") or [])
            spans = [dict(s) for s in (state.get("spans") or [])]

        per_row.append(
            {
                "id": row.get("id"),
                "question": question,
                "negative": negative,
                "answer": answer,
                "attempts": attempts,
                "keyword_acc": kw,
                "judge_acc": judge,
                "abstention": abstained,
                "retrieval": ret,
                "retrieved": [
                    {"source": d.get("source"), "page": d.get("page")} for d in retrieved
                ],
                "trace": trace,
                "spans": spans,
                "cost": _span_totals(spans),
            }
        )

    elapsed = time.perf_counter() - started
    pos = [r for r in per_row if not r["negative"]]
    neg = [r for r in per_row if r["negative"]]
    judges = [float(r["judge_acc"]) for r in per_row if r["judge_acc"] is not None]
    costs = [_span_totals(r.get("spans") or []) for r in per_row]
    bm25 = CONFIG.use_bm25 if use_bm25 is None else use_bm25
    rerank = CONFIG.use_rerank if use_rerank is None else use_rerank
    expand = CONFIG.use_expand if use_expand is None else use_expand
    summary = {
        "provider": CONFIG.llm_provider,
        "model": CONFIG.llm_model,
        "embed_model": CONFIG.embed_model,
        "top_k": CONFIG.top_k,
        "grounding_check": CONFIG.grounding_check,
        "max_retries": CONFIG.max_retries,
        "extractor": CONFIG.extractor,
        "contextual": CONFIG.contextual,
        "use_bm25": bm25,
        "use_rerank": rerank,
        "use_expand": expand,
        "n": len(per_row),
        "hit@k": _mean([float(r["retrieval"]["hit@k"]) for r in pos]),
        "recall@k": _mean([float(r["retrieval"]["recall@k"]) for r in pos]),
        "mrr": _mean([float(r["retrieval"]["mrr"]) for r in pos]),
        "keyword_acc": None if retrieval_only else _mean([float(r["keyword_acc"]) for r in per_row]),
        "judge_acc": None if retrieval_only or no_judge else _mean(judges),
        "abstention_acc": None
        if retrieval_only
        else (_mean([1.0 if r["abstention"] else 0.0 for r in neg]) if neg else None),
        "mean_attempts": None if retrieval_only else _mean([float(r["attempts"]) for r in per_row]),
        "retry_rate": None
        if retrieval_only
        else _mean([1.0 if r["attempts"] > 1 else 0.0 for r in per_row]),
        "mean_latency_ms": _mean([c["latency_ms"] for c in costs]) if not retrieval_only else None,
        "mean_tokens_in": _mean([c["tokens_in"] for c in costs]) if not retrieval_only else None,
        "mean_tokens_out": _mean([c["tokens_out"] for c in costs]) if not retrieval_only else None,
        "mean_cost_usd": _mean([c["cost_usd"] for c in costs]) if not retrieval_only else None,
        "wall_time_s": round(elapsed, 3),
        "tag": tag,
        "retrieval_only": retrieval_only,
    }
    return {"summary": summary, "rows": per_row}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline dsqa eval suite")
    parser.add_argument("--dataset", default=str(Path(__file__).with_name("dataset.jsonl")))
    parser.add_argument("--tag", default="run")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="score hit@k/recall@k/mrr without calling the LLM",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bm25", dest="bm25", action="store_true", default=None)
    parser.add_argument("--no-bm25", dest="bm25", action="store_false")
    parser.add_argument("--rerank", dest="rerank", action="store_true", default=None)
    parser.add_argument("--no-rerank", dest="rerank", action="store_false")
    parser.add_argument("--expand", dest="expand", action="store_true", default=None)
    parser.add_argument("--no-expand", dest="expand", action="store_false")
    args = parser.parse_args(argv)

    result = run(
        Path(args.dataset),
        args.tag,
        no_judge=args.no_judge,
        retrieval_only=args.retrieval_only,
        use_bm25=args.bm25,
        use_rerank=args.rerank,
        use_expand=args.expand,
        limit=args.limit,
    )
    print(json.dumps(result["summary"], indent=2))

    out_dir = Path(__file__).with_name("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{stamp}-{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
