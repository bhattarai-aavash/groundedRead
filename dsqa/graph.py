from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from dsqa.config import CONFIG
from dsqa.grounding import missing_required_terms
from dsqa.guardrails import ABSTAIN, output_violations, sanitize_question
from dsqa.llm import estimate_cost_usd, estimate_tokens, get_llm
from dsqa.prompts import (
    CRITIC_SYSTEM,
    RETRIEVER_SYSTEM,
    SUPERVISOR_SYSTEM,
    WRITER_SYSTEM,
    check_user,
    generate_user,
    team_user,
)
from dsqa.store import query

Handoff = Literal["retriever", "writer", "end"]


class Span(TypedDict, total=False):
    node: str
    ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    detail: str


class State(TypedDict):
    question: str
    search_query: str
    docs: list[dict[str, Any]]
    answer: str
    grounded: bool
    reason: str
    missing: str
    attempts: int
    steps: int
    handoff: str
    last_speaker: str
    queries: list[str]
    required_terms: list[str]
    raw_question: str
    trace: list[str]
    spans: list[Span]


_compiled: dict[tuple[Any, ...], Any] = {}


def parse_json(text: str) -> dict[str, Any]:
    """Tolerant JSON object parser for local models that wrap JSON in prose."""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(s[start : end + 1])


def _append_trace(state: State, message: str) -> list[str]:
    return list(state.get("trace") or []) + [message]


def _append_span(
    state: State,
    node: str,
    started: float,
    *,
    tokens_in: int = 0,
    tokens_out: int = 0,
    detail: str = "",
) -> list[Span]:
    ms = (time.perf_counter() - started) * 1000
    span: Span = {
        "node": node,
        "ms": round(ms, 1),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(estimate_cost_usd(tokens_in, tokens_out), 6),
        "detail": detail,
    }
    return list(state.get("spans") or []) + [span]


def _merge_docs(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cap = max(CONFIG.top_k * 2, 8)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for doc in list(old) + list(new):
        key = str(doc.get("id") or f"{doc.get('source')}:{doc.get('page')}:{doc.get('text')}")
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
        if len(out) >= cap:
            break
    return out


def _coerce_handoff(raw: str, state: State, steps: int) -> Handoff:
    nxt = raw.strip().lower()
    if nxt in {"retrieve", "search"}:
        nxt = "retriever"
    if nxt in {"generate", "answer"}:
        nxt = "writer"
    if nxt in {"abstain", "stop", "done"}:
        nxt = "end"
    if nxt not in {"retriever", "writer", "end"}:
        nxt = "retriever" if not (state.get("docs") or []) else "writer"
    docs = list(state.get("docs") or [])
    answer = str(state.get("answer") or "")
    if nxt == "writer" and not docs and steps < CONFIG.max_steps:
        return "retriever"
    if nxt == "end" and not answer and docs:
        return "writer"
    return nxt  # type: ignore[return-value]


def supervisor(state: State) -> dict[str, Any]:
    started = time.perf_counter()
    docs = list(state.get("docs") or [])
    steps = int(state.get("steps") or 0)
    tin = tout = 0
    if steps >= CONFIG.max_steps:
        nxt: Handoff = "writer" if docs else "end"
        reason = "max_steps exhausted; forcing close"
    else:
        steps += 1
        llm = get_llm()
        user = team_user(
            state["question"],
            docs,
            state.get("missing") or "",
            steps,
            CONFIG.max_steps,
            list(state.get("queries") or []),
            state.get("last_speaker") or "",
        )
        raw = llm.chat(SUPERVISOR_SYSTEM, user)
        tin, tout = estimate_tokens(SUPERVISOR_SYSTEM + user), estimate_tokens(raw)
        try:
            data = parse_json(raw)
            nxt = _coerce_handoff(str(data.get("next") or data.get("handoff") or ""), state, steps)
            reason = str(data.get("reason") or "")
        except (json.JSONDecodeError, ValueError, TypeError):
            nxt = "retriever" if not docs else "writer"
            reason = "unparseable supervisor JSON; falling back"
    detail = f"next={nxt}"
    return {
        "handoff": nxt,
        "steps": steps,
        "last_speaker": "supervisor",
        "reason": reason,
        "trace": _append_trace(state, f"supervisor {detail} reason={reason!r}"),
        "spans": _append_span(
            state, "supervisor", started, tokens_in=tin, tokens_out=tout, detail=detail
        ),
    }


def retriever(state: State) -> dict[str, Any]:
    started = time.perf_counter()
    llm = get_llm()
    user = team_user(
        state["question"],
        list(state.get("docs") or []),
        state.get("missing") or "",
        int(state.get("steps") or 0),
        CONFIG.max_steps,
        list(state.get("queries") or []),
        "supervisor",
    )
    raw = llm.chat(RETRIEVER_SYSTEM, user)
    tin, tout = estimate_tokens(RETRIEVER_SYSTEM + user), estimate_tokens(raw)
    try:
        data = parse_json(raw)
        query_text = str(data.get("query") or "").strip() or state["question"]
        reason = str(data.get("reason") or "")
    except (json.JSONDecodeError, ValueError, TypeError):
        query_text = state["question"]
        reason = "unparseable retriever JSON; using the question"
    queries = list(state.get("queries") or []) + [query_text]
    detail = f"q={query_text!r}"
    return {
        "search_query": query_text,
        "queries": queries,
        "last_speaker": "retriever",
        "reason": reason,
        "trace": _append_trace(state, f"retriever {detail} reason={reason!r}"),
        "spans": _append_span(
            state, "retriever", started, tokens_in=tin, tokens_out=tout, detail=detail
        ),
    }


def search(state: State) -> dict[str, Any]:
    started = time.perf_counter()
    q = state.get("search_query") or state["question"]
    found = query(q)
    docs = _merge_docs(list(state.get("docs") or []), found)
    sources = ", ".join(f"{d.get('source')} p.{d.get('page')}" for d in found[:3])
    detail = f"q={q!r} n={len(found)} held={len(docs)} {sources}"
    return {
        "docs": docs,
        "last_speaker": "search",
        "trace": _append_trace(state, f"search {detail}"),
        "spans": _append_span(state, "search", started, detail=detail),
    }


def _writer_user(state: State) -> str:
    return generate_user(
        state["question"],
        state.get("docs") or [],
        missing=str(state.get("missing") or ""),
        required_terms=list(state.get("required_terms") or []),
        attempt=int(state.get("attempts") or 0) + 1,
    )


def writer(state: State) -> dict[str, Any]:
    started = time.perf_counter()
    llm = get_llm()
    user = _writer_user(state)
    answer = llm.chat(WRITER_SYSTEM, user).strip()
    attempts = int(state.get("attempts") or 0) + 1
    tin, tout = estimate_tokens(WRITER_SYSTEM + user), estimate_tokens(answer)
    return {
        "answer": answer,
        "attempts": attempts,
        "last_speaker": "writer",
        "trace": _append_trace(state, f"writer attempt={attempts}"),
        "spans": _append_span(
            state, "writer", started, tokens_in=tin, tokens_out=tout, detail=f"attempt={attempts}"
        ),
    }


def finish(state: State) -> dict[str, Any]:
    started = time.perf_counter()
    if str(state.get("answer") or "").strip():
        return {
            "last_speaker": "supervisor",
            "trace": _append_trace(state, "finish keep existing answer"),
            "spans": _append_span(state, "finish", started, detail="keep"),
        }
    reason = state.get("reason") or "supervisor ended without an answer"
    return {
        "answer": ABSTAIN,
        "grounded": False,
        "reason": reason,
        "attempts": int(state.get("attempts") or 0) + 1,
        "last_speaker": "supervisor",
        "trace": _append_trace(state, f"finish abstain {reason!r}"),
        "spans": _append_span(state, "finish", started, detail=reason),
    }


def critic(state: State) -> dict[str, Any]:
    started = time.perf_counter()
    answer = state.get("answer") or ""
    if ABSTAIN in answer:
        return {
            "grounded": False,
            "reason": "answer abstained with INSUFFICIENT_CONTEXT",
            "missing": state.get("question", ""),
            "last_speaker": "critic",
            "trace": _append_trace(state, "critic short-circuit abstention grounded=False"),
            "spans": _append_span(state, "critic", started, detail="short-circuit abstention"),
        }
    missing_kw = missing_required_terms(answer, state.get("required_terms"))
    if missing_kw:
        missing = "; ".join(missing_kw)
        reason = f"missing required terms: {missing}"
        return {
            "grounded": False,
            "reason": reason,
            "missing": missing,
            "last_speaker": "critic",
            "trace": _append_trace(state, f"critic fail-closed {reason}"),
            "spans": _append_span(state, "critic", started, detail=reason),
        }
    if CONFIG.guardrails:
        violations = output_violations(answer, list(state.get("docs") or []))
        if violations:
            reason = "; ".join(violations)
            return {
                "grounded": False,
                "reason": reason,
                "missing": reason,
                "last_speaker": "critic",
                "trace": _append_trace(state, f"critic fail-closed {reason}"),
                "spans": _append_span(state, "critic", started, detail=reason),
            }
    llm = get_llm()
    user = check_user(state["question"], answer, state.get("docs") or [])
    raw = llm.chat(CRITIC_SYSTEM, user)
    tin, tout = estimate_tokens(CRITIC_SYSTEM + user), estimate_tokens(raw)
    try:
        data = parse_json(raw)
        grounded = bool(data.get("grounded"))
        reason = str(data.get("reason", ""))
        missing = data.get("missing", "")
        if isinstance(missing, list):
            missing = "; ".join(str(x) for x in missing)
        missing = str(missing)
    except (json.JSONDecodeError, ValueError, TypeError):
        grounded = True
        reason = "unparseable critic JSON; treating as grounded"
        missing = ""
    return {
        "grounded": grounded,
        "reason": reason,
        "missing": missing,
        "last_speaker": "critic",
        "trace": _append_trace(state, f"critic grounded={grounded} reason={reason!r}"),
        "spans": _append_span(
            state, "critic", started, tokens_in=tin, tokens_out=tout, detail=f"grounded={grounded}"
        ),
    }


def _route_supervisor(state: State) -> Handoff:
    nxt = str(state.get("handoff") or "writer")
    if nxt in {"retriever", "writer", "end"}:
        return nxt  # type: ignore[return-value]
    return "writer"


def _route_after_critic(state: State) -> Literal["end", "supervisor", "guard"]:
    if state.get("grounded"):
        return "end"
    if int(state.get("attempts") or 0) > CONFIG.max_retries:
        return "guard" if CONFIG.guardrails else "end"
    if int(state.get("steps") or 0) >= CONFIG.max_steps:
        return "guard" if CONFIG.guardrails else "end"
    return "supervisor"


def guard(state: State) -> dict[str, Any]:
    """Last-line output guard: never return an ungrounded draft (incl. jailbreaks)."""
    started = time.perf_counter()
    if not CONFIG.guardrails or state.get("grounded"):
        return {
            "last_speaker": "guard",
            "trace": _append_trace(state, "guard pass"),
            "spans": _append_span(state, "guard", started, detail="pass"),
        }
    answer = str(state.get("answer") or "")
    if ABSTAIN in answer:
        return {
            "last_speaker": "guard",
            "trace": _append_trace(state, "guard already abstained"),
            "spans": _append_span(state, "guard", started, detail="already-abstain"),
        }
    reason = state.get("reason") or "ungrounded after retries"
    return {
        "answer": ABSTAIN,
        "grounded": False,
        "reason": reason,
        "last_speaker": "guard",
        "trace": _append_trace(state, f"guard abstain {reason!r}"),
        "spans": _append_span(state, "guard", started, detail=reason),
    }


def _build() -> StateGraph:
    g: StateGraph = StateGraph(State)
    g.add_node("supervisor", supervisor)
    g.add_node("retriever", retriever)
    g.add_node("search", search)
    g.add_node("writer", writer)
    g.add_node("finish", finish)
    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {"retriever": "retriever", "writer": "writer", "end": "finish"},
    )
    g.add_edge("retriever", "search")
    g.add_edge("search", "supervisor")
    g.add_edge("finish", END)
    if CONFIG.grounding_check:
        g.add_node("critic", critic)
        g.add_node("guard", guard)
        g.add_edge("writer", "critic")
        g.add_conditional_edges(
            "critic",
            _route_after_critic,
            {"end": END, "supervisor": "supervisor", "guard": "guard"},
        )
        g.add_edge("guard", END)
    else:
        g.add_edge("writer", END)
    return g


def get_graph() -> Any:
    key = (
        CONFIG.grounding_check,
        CONFIG.max_retries,
        CONFIG.max_steps,
        CONFIG.use_bm25,
        CONFIG.use_rerank,
        CONFIG.use_expand,
        CONFIG.top_k,
        CONFIG.guardrails,
    )
    if key not in _compiled:
        _compiled[key] = _build().compile()
    return _compiled[key]


def _initial(question: str, required_terms: list[str] | None = None) -> State:
    raw = question.strip()
    cleaned = (
        sanitize_question(raw, CONFIG.max_question_chars) if CONFIG.guardrails else raw
    )
    return {
        "question": cleaned or raw,
        "raw_question": raw,
        "search_query": "",
        "docs": [],
        "answer": "",
        "grounded": False,
        "reason": "",
        "missing": "",
        "attempts": 0,
        "steps": 0,
        "handoff": "",
        "last_speaker": "",
        "queries": [],
        "required_terms": list(required_terms or []),
        "trace": [],
        "spans": [],
    }


def ask(question: str, required_terms: list[str] | None = None) -> State:
    result = get_graph().invoke(_initial(question, required_terms))
    return result  # type: ignore[no-any-return]


# Aliases so existing tests that call check() still work.
check = critic


def _stream_writer(state: State) -> Iterator[dict[str, Any]]:
    llm = get_llm()
    user = _writer_user(state)
    started = time.perf_counter()
    stream = getattr(llm, "chat_stream", None)
    parts: list[str] = []
    if stream is None:
        answer = llm.chat(WRITER_SYSTEM, user)
        parts.append(answer)
        yield {"event": "token", "data": {"text": answer}}
    else:
        for piece in stream(WRITER_SYSTEM, user):
            parts.append(piece)
            yield {"event": "token", "data": {"text": piece}}
    answer = "".join(parts).strip()
    tin, tout = estimate_tokens(WRITER_SYSTEM + user), estimate_tokens(answer)
    attempts = int(state.get("attempts") or 0) + 1
    state["answer"] = answer
    state["attempts"] = attempts
    state["last_speaker"] = "writer"
    state["trace"] = _append_trace(state, f"writer attempt={attempts}")
    state["spans"] = _append_span(
        state, "writer", started, tokens_in=tin, tokens_out=tout, detail=f"attempt={attempts}"
    )


def ask_stream(question: str, required_terms: list[str] | None = None) -> Iterator[dict[str, Any]]:
    """Supervisor loop; stream tokens only while the writer drafts."""
    state: State = _initial(question, required_terms)
    budget = CONFIG.max_steps + CONFIG.max_retries + 2
    for _ in range(budget):
        yield {
            "event": "status",
            "data": {"stage": "supervisor", "step": int(state.get("steps") or 0)},
        }
        state.update(supervisor(state))  # type: ignore[typeddict-item]
        nxt = str(state.get("handoff") or "writer")
        if nxt == "retriever":
            yield {"event": "status", "data": {"stage": "retriever"}}
            state.update(retriever(state))  # type: ignore[typeddict-item]
            yield {"event": "status", "data": {"stage": "retrieve"}}
            state.update(search(state))  # type: ignore[typeddict-item]
            continue
        if nxt == "end":
            state.update(finish(state))  # type: ignore[typeddict-item]
            break
        yield {
            "event": "status",
            "data": {"stage": "writer", "n_docs": len(state.get("docs") or [])},
        }
        replace = int(state.get("attempts") or 0) > 0
        first = True
        for event in _stream_writer(state):
            if replace and first and event.get("event") == "token":
                data = dict(event.get("data") or {})
                data["replace"] = True
                yield {"event": "token", "data": data}
                first = False
                continue
            first = False
            yield event
        if CONFIG.grounding_check:
            yield {"event": "status", "data": {"stage": "critic"}}
            state.update(critic(state))  # type: ignore[typeddict-item]
            route = _route_after_critic(state)
            if route == "supervisor":
                yield {"event": "status", "data": {"stage": "retry"}}
                continue
            if route == "guard":
                yield {"event": "status", "data": {"stage": "guard"}}
                state.update(guard(state))  # type: ignore[typeddict-item]
        break
    yield {"event": "done", "data": state}
