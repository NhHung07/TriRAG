"""Lớp LLM dùng chung: cache SQLite, giới hạn tốc độ, retry, thống kê token.

Mọi lời gọi LLM trong dự án (sinh câu trả lời, tóm tắt RAPTOR, trích xuất LightRAG,
tách câu hỏi SG-RAG, chấm điểm) đều đi qua :class:`CachedLLM` để:

* chạy lại thí nghiệm không tốn quota và cho kết quả giống hệt (temperature=0 + cache);
* tôn trọng giới hạn RPM của API (token bucket);
* đếm số lời gọi / token để báo cáo chi phí lập chỉ mục.
"""

from __future__ import annotations

import logging
import random
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..utils import sha256

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class RetryableError(Exception):
    """Lỗi tạm thời (429, 5xx, timeout) — nên thử lại."""


class LLMBackend(ABC):
    """Một nhà cung cấp LLM cụ thể (Gemini, OpenAI-compatible, giả lập...)."""

    model: str

    @abstractmethod
    def complete(
        self, prompt: str, *, system: str | None, temperature: float, max_tokens: int
    ) -> LLMResponse: ...


class SQLiteCache:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, model TEXT, "
            "response TEXT, input_tokens INT, output_tokens INT, created REAL)"
        )
        self._conn.commit()

    def get(self, key: str) -> LLMResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response, input_tokens, output_tokens FROM llm_cache WHERE key=?", (key,)
            ).fetchone()
        return LLMResponse(row[0], row[1], row[2]) if row else None

    def put(self, key: str, model: str, resp: LLMResponse) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?,?,?,?,?,?)",
                (key, model, resp.text, resp.input_tokens, resp.output_tokens, time.time()),
            )
            self._conn.commit()


class RateLimiter:
    """Giới hạn số request/phút (khoảng cách tối thiểu giữa 2 request)."""

    def __init__(self, rpm: float | None):
        self.interval = 60.0 / rpm if rpm else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)


@dataclass
class UsageStats:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    api_seconds: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class UsageTracker:
    """Thống kê theo nhãn (tag), VD 'index:lightrag', 'answer', 'judge'."""

    by_tag: dict[str, UsageStats] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, tag: str, resp: LLMResponse, cached: bool, seconds: float) -> None:
        with self._lock:
            s = self.by_tag.setdefault(tag, UsageStats())
            if cached:
                s.cache_hits += 1
            else:
                s.calls += 1
                s.input_tokens += resp.input_tokens
                s.output_tokens += resp.output_tokens
                s.api_seconds += seconds

    def snapshot(self, tag: str) -> dict:
        with self._lock:
            return self.by_tag.get(tag, UsageStats()).as_dict()


class CachedLLM:
    """Bọc một :class:`LLMBackend` với cache + rate limit + retry + thống kê."""

    def __init__(
        self,
        backend: LLMBackend,
        cache: SQLiteCache | None,
        rpm: float | None = None,
        max_retries: int = 8,
        tracker: UsageTracker | None = None,
        default_tag: str = "default",
    ):
        self.backend = backend
        self.cache = cache
        self.limiter = RateLimiter(rpm)
        self.max_retries = max_retries
        self.tracker = tracker or UsageTracker()
        self.default_tag = default_tag

    @property
    def model(self) -> str:
        return self.backend.model

    def cache_key(self, prompt: str, system: str | None, temperature: float, max_tokens: int) -> str:
        return sha256(f"{self.backend.model}\x1f{system or ''}\x1f{prompt}\x1f{temperature}\x1f{max_tokens}")

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tag: str | None = None,
        use_cache: bool = True,
    ) -> str:
        return self.generate_full(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens, tag=tag, use_cache=use_cache
        )[0].text

    def generate_full(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tag: str | None = None,
        use_cache: bool = True,
    ) -> tuple[LLMResponse, bool]:
        """Trả về (response, cached)."""
        tag = tag or self.default_tag
        key = self.cache_key(prompt, system, temperature, max_tokens)
        if use_cache and self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.tracker.record(tag, hit, cached=True, seconds=0.0)
                return hit, True

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            t0 = time.perf_counter()
            try:
                resp = self.backend.complete(
                    prompt, system=system, temperature=temperature, max_tokens=max_tokens
                )
                break
            except RetryableError as e:
                if attempt == self.max_retries:
                    raise
                sleep = min(delay * (2**attempt), 120) * (1 + random.random() * 0.25)
                log.warning("LLM lỗi tạm thời (%s) — thử lại sau %.1fs", e, sleep)
                time.sleep(sleep)
        seconds = time.perf_counter() - t0
        if use_cache and self.cache is not None:
            self.cache.put(key, self.backend.model, resp)
        self.tracker.record(tag, resp, cached=False, seconds=seconds)
        return resp, False
