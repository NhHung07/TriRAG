"""Tạo LLM từ cấu hình."""

from __future__ import annotations

from pathlib import Path

from .base import CachedLLM, LLMBackend, SQLiteCache, UsageTracker

_CACHES: dict[str, SQLiteCache] = {}
TRACKER = UsageTracker()  # thống kê toàn cục cho một tiến trình


def _cache(path: Path) -> SQLiteCache:
    key = str(path)
    if key not in _CACHES:
        _CACHES[key] = SQLiteCache(path)
    return _CACHES[key]


def make_backend(spec: dict) -> LLMBackend:
    provider = spec.get("provider", "gemini")
    if provider == "gemini":
        from .gemini import GeminiBackend

        return GeminiBackend(spec["model"], thinking_budget=spec.get("thinking_budget", 0))
    if provider in ("openai", "openai_compat"):
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(
            spec["model"],
            base_url=spec.get("base_url"),
            api_key_env=spec.get("api_key_env", "JUDGE_API_KEY"),
            base_url_env=spec.get("base_url_env", "JUDGE_BASE_URL"),
        )
    if provider == "fake":
        from .fake import FakeBackend

        return FakeBackend(model=spec.get("model", "fake"))
    raise ValueError(f"provider không hỗ trợ: {provider}")


def make_llm(spec: dict, cache_dir: str | Path, default_tag: str) -> CachedLLM:
    backend = make_backend(spec)
    cache = _cache(Path(cache_dir) / "llm_cache.sqlite") if spec.get("cache", True) else None
    return CachedLLM(
        backend,
        cache,
        rpm=spec.get("rpm"),
        max_retries=spec.get("max_retries", 8),
        tracker=TRACKER,
        default_tag=default_tag,
    )
