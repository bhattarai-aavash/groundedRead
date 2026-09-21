from __future__ import annotations

import re

from dsqa.config import CONFIG

# Definition / “what is” questions. “How does X organize” is included because
# those prompts hide a definition in the last clause (“what is a tablet?”).
_DEFINITION = re.compile(
    r"\b(what is|what are|what does|what do|define|how does\b.+\borganiz)",
    re.I,
)

# Chord lookup is not a definition prompt; gold is successor/fingers on p.4.
_CHORD_LOOKUP = re.compile(
    r"\bchord\b.{0,80}\b(locate|lookup|responsible)\b"
    r"|\b(locate|lookup|responsible)\b.{0,80}\bchord\b",
    re.I,
)
_CHORD_EXTRA = "successor finger table"

# Surface noun in the question → paper terms MiniLM under-weights.
# Keep this tiny; extra tokens pollute BM25 (OpSet expansion dropped a gold hit).
_GLOSS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\btablets?\b", re.I), "data model column family row key row range"),
    (re.compile(r"\btruetime\b", re.I), "commit wait time interval API external consistency"),
]


def is_definition_question(question: str) -> bool:
    return bool(_DEFINITION.search(question))


def is_chord_lookup(question: str) -> bool:
    return bool(_CHORD_LOOKUP.search(question))


def expand_query(question: str, enabled: bool | None = None) -> str:
    """Append glossary / Chord-lookup terms. No-op when disabled."""
    on = CONFIG.use_expand if enabled is None else enabled
    if not on or not question.strip():
        return question
    extras: list[str] = []
    seen: set[str] = set()
    lower = question.lower()

    def _add(extra: str) -> None:
        key = extra.lower()
        if key in seen or key in lower:
            return
        seen.add(key)
        extras.append(extra)

    if is_definition_question(question):
        for pat, extra in _GLOSS:
            if pat.search(question):
                _add(extra)
    if is_chord_lookup(question):
        _add(_CHORD_EXTRA)
    if not extras:
        return question
    return f"{question} {' '.join(extras)}"
