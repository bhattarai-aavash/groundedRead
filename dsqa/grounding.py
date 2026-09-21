from __future__ import annotations

from collections.abc import Iterable


def missing_required_terms(answer: str, terms: Iterable[str] | None) -> list[str]:
    """Return must-include substrings absent from the answer (case-insensitive)."""
    if not terms:
        return []
    lower = answer.lower()
    return [t for t in terms if t and t.lower() not in lower]
