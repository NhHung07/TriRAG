"""Kiểm thử cache SQLite và bộ đếm token của CachedLLM."""

from __future__ import annotations

from pathlib import Path

from trirag.llm.base import CachedLLM, SQLiteCache, UsageTracker
from trirag.llm.fake import FakeBackend


def test_sqlite_cache_deduplicates_calls(tmp_path: Path):
    db_path = tmp_path / "llm_cache.sqlite"
    backend = FakeBackend(responder=lambda prompt, sys: f"ans:{prompt}", model="fake-v1")
    tracker = UsageTracker()
    llm = CachedLLM(backend, SQLiteCache(db_path), rpm=None, tracker=tracker, default_tag="test")

    out1 = llm.generate("What is RAG?", system="sys", temperature=0.0, max_tokens=32)
    out2 = llm.generate("What is RAG?", system="sys", temperature=0.0, max_tokens=32)
    assert out1 == out2 == "ans:What is RAG?"
    assert backend.calls == 1

    snap = tracker.snapshot("test")
    assert snap["calls"] == 1
    assert snap["cache_hits"] == 1

    # Khởi tạo lại trên cùng file SQLite -> vẫn trúng cache, không gọi backend thêm lần nào
    llm_reloaded = CachedLLM(backend, SQLiteCache(db_path), rpm=None, tracker=tracker, default_tag="test")
    out3 = llm_reloaded.generate("What is RAG?", system="sys", temperature=0.0, max_tokens=32)
    assert out3 == out1
    assert backend.calls == 1
