from __future__ import annotations

import pytest

from dsqa.config import CONFIG
from dsqa.graph import _compiled


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Keep tests off the real index and graph cache."""
    CONFIG.chroma_dir = str(tmp_path / "chroma")
    CONFIG.collection = "dsqa-test"
    CONFIG.cache_dir = str(tmp_path / "cache")
    CONFIG.cache_embed = False
    CONFIG.cache_llm = False
    CONFIG.use_rerank = False
    CONFIG.use_bm25 = True
    CONFIG.use_expand = True
    CONFIG.grounding_check = True
    CONFIG.max_retries = 1
    CONFIG.max_steps = 4
    CONFIG.contextual = True
    CONFIG.extractor = "pymupdf"
    CONFIG.top_k = 4
    CONFIG.rerank_pool = 8
    CONFIG.llm_provider = "ollama"
    CONFIG.guardrails = True
    _compiled.clear()
    yield
    _compiled.clear()
