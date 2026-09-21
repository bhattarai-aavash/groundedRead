from __future__ import annotations

import os
from dataclasses import dataclass

# Simpler than python-dotenv: read the process env so local-only runs need no extra dep.
_DEFAULT_MODELS: dict[str, str] = {
    "ollama": "llama3.1",
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
}


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass
class Config:
    llm_provider: str = "ollama"
    llm_model: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    ollama_host: str = "http://127.0.0.1:11434"
    embed_model: str = "all-MiniLM-L6-v2"
    chroma_dir: str = ".chroma"
    collection: str = "dsqa"
    chunk_chars: int = 1200
    chunk_overlap: int = 200
    top_k: int = 6
    grounding_check: bool = True
    max_retries: int = 1
    max_steps: int = 4
    extractor: str = "pymupdf"
    use_bm25: bool = True
    # MS MARCO MiniLM lost on this corpus (RESULTS.md). Stay off until a
    # scientific-QA cross-encoder beats dense+BM25+expand on eval/dataset.jsonl.
    use_rerank: bool = False
    use_expand: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_pool: int = 30
    contextual: bool = True
    cache_embed: bool = True
    cache_llm: bool = False
    cache_dir: str = ".cache"
    guardrails: bool = True
    max_question_chars: int = 2000

    def __post_init__(self) -> None:
        if not self.llm_model:
            self.llm_model = _DEFAULT_MODELS.get(self.llm_provider, "llama3.1")
        self.extractor = self.extractor.strip().lower()
        if self.extractor not in {"pymupdf", "plumber"}:
            self.extractor = "pymupdf"


def load_config() -> Config:
    return Config(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
        llm_model=os.getenv("LLM_MODEL", "").strip(),
        temperature=float(os.getenv("TEMPERATURE", "0")),
        max_tokens=_int(os.getenv("MAX_TOKENS"), 1024),
        ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
        embed_model=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"),
        chroma_dir=os.getenv("CHROMA_DIR", ".chroma"),
        collection=os.getenv("CHROMA_COLLECTION", "dsqa"),
        chunk_chars=_int(os.getenv("CHUNK_CHARS"), 1200),
        chunk_overlap=_int(os.getenv("CHUNK_OVERLAP"), 200),
        top_k=_int(os.getenv("TOP_K"), 6),
        grounding_check=_bool(os.getenv("GROUNDING_CHECK"), True),
        max_retries=_int(os.getenv("MAX_RETRIES"), 1),
        max_steps=_int(os.getenv("MAX_STEPS"), 4),
        extractor=os.getenv("EXTRACTOR", "pymupdf"),
        use_bm25=_bool(os.getenv("USE_BM25"), True),
        use_rerank=_bool(os.getenv("USE_RERANK"), False),
        use_expand=_bool(os.getenv("USE_EXPAND"), True),
        rerank_model=os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        rerank_pool=_int(os.getenv("RERANK_POOL"), 30),
        contextual=_bool(os.getenv("CONTEXTUAL"), True),
        cache_embed=_bool(os.getenv("CACHE_EMBED"), True),
        cache_llm=_bool(os.getenv("CACHE_LLM"), False),
        cache_dir=os.getenv("CACHE_DIR", ".cache"),
        guardrails=_bool(os.getenv("GUARDRAILS"), True),
        max_question_chars=_int(os.getenv("MAX_QUESTION_CHARS"), 2000),
    )


CONFIG = load_config()
