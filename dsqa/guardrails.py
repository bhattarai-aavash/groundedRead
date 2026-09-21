from __future__ import annotations

import re
from typing import Any

ABSTAIN = "INSUFFICIENT_CONTEXT"

_CITE = re.compile(
    r"\[(?:source=)?([^\[\]\s]+\.pdf)\s+p\.?\s*(\d+)\]",
    re.I,
)
_JAILBREAK_PREFIX = (
    r"(?:"
    r"ignore (?:previous|all|the|retrieved)?\s*(?:instructions?|excerpts?|papers?)"
    r"|do not cite(?: papers?)?"
    r"|output only \w+"
    r"|you are now [^.?!]+"
    r"|forget (?:your|all) (?:instructions?|rules?)"
    r")"
    r"(?:\s+and)?"
    r"\s*[,.;:]?\s*"
)
_PREFIX = re.compile(rf"^{_JAILBREAK_PREFIX}", re.I)
_LEADING_AND = re.compile(r"^(?:and|then|please)\s+", re.I)
_INJECTION_ANSWER = re.compile(
    r"^\s*pwned\s*[.!]?\s*$"
    r"|as a helpful chef"
    r"|i(?:'m| am) (?:now )?a (?:helpful )?chef"
    r"|ignore (?:the )?(?:excerpts|papers|instructions)",
    re.I,
)


def sanitize_question(question: str, max_chars: int = 2000) -> str:
    """Strip jailbreak prefixes; keep the underlying research question."""
    q = " ".join(question.split()).strip()
    if max_chars > 0:
        q = q[:max_chars].strip()
    prev = None
    while prev != q:
        prev = q
        q = _PREFIX.sub("", q).strip()
        q = _LEADING_AND.sub("", q).strip()
    return q or question.strip()[: max(max_chars, 1)]


def citation_hits(answer: str) -> list[tuple[str, int]]:
    return [(m.group(1), int(m.group(2))) for m in _CITE.finditer(answer)]


def held_sources(docs: list[dict[str, Any]]) -> set[str]:
    return {str(d.get("source") or "").lower() for d in docs if d.get("source")}


def citation_violations(answer: str, docs: list[dict[str, Any]]) -> list[str]:
    """Non-abstaining answers must cite a held excerpt."""
    if ABSTAIN in answer:
        return []
    hits = citation_hits(answer)
    if not hits:
        return ["no citations"]
    allowed = held_sources(docs)
    if not allowed:
        return ["citations without retrieved excerpts"]
    bad = [f"{src} p.{page}" for src, page in hits if src.lower() not in allowed]
    return [f"citation not in retrieved set: {b}" for b in bad]


def injection_compliance(answer: str) -> bool:
    """True when the draft followed a jailbreak instead of answering."""
    if ABSTAIN in answer:
        return False
    return bool(_INJECTION_ANSWER.search(answer.strip()))


def output_violations(
    answer: str,
    docs: list[dict[str, Any]],
) -> list[str]:
    """Deterministic output guards. Empty list means the draft may go to the critic LLM."""
    text = (answer or "").strip()
    if not text:
        return ["empty answer"]
    if injection_compliance(text):
        return ["injection compliance"]
    return citation_violations(text, docs)
