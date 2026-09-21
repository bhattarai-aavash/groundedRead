from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Protocol

from dsqa.cache import get_json, put_json
from dsqa.config import CONFIG

_embedder = None


class LLM(Protocol):
    def chat(self, system: str, user: str) -> str: ...


def estimate_tokens(text: str) -> int:
    # Cheap stand-in for a tokenizer; good enough for cost/latency traces.
    return max(1, (len(text) + 3) // 4)


def estimate_cost_usd(tokens_in: int, tokens_out: int) -> float:
    provider = CONFIG.llm_provider
    if provider == "ollama":
        return 0.0
    if provider == "openai":
        return tokens_in * 0.15 / 1_000_000 + tokens_out * 0.60 / 1_000_000
    if provider == "anthropic":
        return tokens_in * 3.00 / 1_000_000 + tokens_out * 15.00 / 1_000_000
    return 0.0


def _llm_key(system: str, user: str) -> str:
    return json.dumps(
        {
            "provider": CONFIG.llm_provider,
            "model": CONFIG.llm_model,
            "temperature": CONFIG.temperature,
            "system": system,
            "user": user,
        },
        sort_keys=True,
    )


class CachedLLM:
    def __init__(self, inner: LLM) -> None:
        self.inner = inner

    def chat(self, system: str, user: str) -> str:
        if CONFIG.cache_llm:
            hit = get_json("llm", _llm_key(system, user))
            if isinstance(hit, str):
                return hit
        out = self.inner.chat(system, user)
        if CONFIG.cache_llm:
            put_json("llm", _llm_key(system, user), out)
        return out

    def chat_stream(self, system: str, user: str) -> Iterator[str]:
        stream = getattr(self.inner, "chat_stream", None)
        if stream is None:
            yield self.chat(system, user)
            return
        if CONFIG.cache_llm:
            hit = get_json("llm", _llm_key(system, user))
            if isinstance(hit, str):
                yield hit
                return
        parts: list[str] = []
        for piece in stream(system, user):
            parts.append(piece)
            yield piece
        if CONFIG.cache_llm:
            put_json("llm", _llm_key(system, user), "".join(parts))


class OllamaLLM:
    def chat(self, system: str, user: str) -> str:
        return "".join(self.chat_stream(system, user))

    def chat_stream(self, system: str, user: str) -> Iterator[str]:
        payload = {
            "model": CONFIG.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "options": {
                "temperature": CONFIG.temperature,
                "num_predict": CONFIG.max_tokens,
            },
        }
        req = urllib.request.Request(
            f"{CONFIG.ollama_host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    piece = str((data.get("message") or {}).get("content") or "")
                    if piece:
                        yield piece
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama request failed at {CONFIG.ollama_host}: {exc}"
            ) from exc


class AnthropicLLM:
    def chat(self, system: str, user: str) -> str:
        import anthropic  # lazy: local-only runs must not need this package

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=CONFIG.llm_model,
            max_tokens=CONFIG.max_tokens,
            temperature=CONFIG.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in msg.content if getattr(b, "type", "") == "text"]
        return "".join(parts) if parts else str(msg.content[0].text)


class OpenAILLM:
    def chat(self, system: str, user: str) -> str:
        from openai import OpenAI  # lazy: local-only runs must not need this package

        client = OpenAI()
        resp = client.chat.completions.create(
            model=CONFIG.llm_model,
            temperature=CONFIG.temperature,
            max_tokens=CONFIG.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def get_llm() -> LLM:
    provider = CONFIG.llm_provider
    if provider == "ollama":
        inner: LLM = OllamaLLM()
    elif provider == "anthropic":
        inner = AnthropicLLM()
    elif provider == "openai":
        inner = OpenAILLM()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER={provider!r}; use ollama|anthropic|openai")
    return CachedLLM(inner)


def embed(texts: list[str]) -> list[list[float]]:
    """Local sentence-transformers embeddings, independent of LLM_PROVIDER."""
    global _embedder
    if not texts:
        return []
    out: list[list[float] | None] = [None] * len(texts)
    missing: list[int] = []
    for i, text in enumerate(texts):
        if CONFIG.cache_embed:
            hit = get_json("embed", f"{CONFIG.embed_model}\0{text}")
            if isinstance(hit, list):
                out[i] = [float(x) for x in hit]
                continue
        missing.append(i)
    if missing:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer(CONFIG.embed_model)
        vectors = _embedder.encode([texts[i] for i in missing], normalize_embeddings=True)
        for j, i in enumerate(missing):
            row = vectors[j].tolist()
            out[i] = row
            if CONFIG.cache_embed:
                put_json("embed", f"{CONFIG.embed_model}\0{texts[i]}", row)
    return [row or [] for row in out]
