"""Embedding chạy local (không tốn quota API), dùng CHUNG cho mọi phương pháp.

* :class:`SentenceTransformerEmbedder` — model HuggingFace (mặc định ``BAAI/bge-m3``, đa ngữ,
  hỗ trợ tiếng Việt), có cache vector trên đĩa (SQLite) để không phải mã hoá lại.
* :class:`HashEmbedder` — embedding băm túi-từ, chỉ dùng cho test.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .utils import sha256

log = logging.getLogger(__name__)


class Embedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        """Trả về ma trận float32 (n, dim), đã chuẩn hoá L2."""


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class _VectorCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS vec (key TEXT PRIMARY KEY, v BLOB)")
        self._conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                part = keys[i : i + 500]
                q = f"SELECT key, v FROM vec WHERE key IN ({','.join('?' * len(part))})"
                for k, v in self._conn.execute(q, part):
                    out[k] = np.frombuffer(v, dtype=np.float32)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO vec VALUES (?, ?)",
                [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()],
            )
            self._conn.commit()


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        device: str = "auto",
        batch_size: int = 16,
        query_prefix: str = "",
        doc_prefix: str = "",
        max_seq_length: int | None = 512,
        cache_dir: str | Path | None = None,
    ):
        self.name = model
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.max_seq_length = max_seq_length
        self._model = None
        self._lock = threading.Lock()
        self._dim: int | None = None
        self._cache = (
            _VectorCache(Path(cache_dir) / f"emb_{re.sub(r'[^A-Za-z0-9]+', '_', model)}.sqlite")
            if cache_dir
            else None
        )

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = self.device
            if device == "auto":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Nạp embedding model %s trên %s", self.name, device)
            kwargs = {}
            if device == "cuda":
                kwargs["model_kwargs"] = {"torch_dtype": "float16"}
            self._model = SentenceTransformer(self.name, device=device, **kwargs)
            if self.max_seq_length:
                self._model.max_seq_length = self.max_seq_length
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dim(self) -> int:  # type: ignore[override]
        if self._dim is None:
            self._load()
        return self._dim  # type: ignore[return-value]

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self.query_prefix if is_query else self.doc_prefix
        inputs = [prefix + t for t in texts]
        keys = [sha256(f"{self.name}\x1f{s}") for s in inputs]
        cached = self._cache.get_many(keys) if self._cache else {}
        missing = [i for i, k in enumerate(keys) if k not in cached]
        if missing:
            with self._lock:  # model không thread-safe trên GPU
                model = self._load()
                vecs = model.encode(
                    [inputs[i] for i in missing],
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=len(missing) > 256,
                    convert_to_numpy=True,
                )
            new = {keys[i]: v for i, v in zip(missing, vecs)}
            if self._cache:
                self._cache.put_many(new)
            cached.update(new)
        return _normalize(np.stack([cached[k] for k in keys]))


class HashEmbedder(Embedder):
    """Embedding băm túi-từ (deterministic, không cần model) — CHỈ dùng cho test."""

    def __init__(self, dim: int = 256):
        self.name = f"hash-{dim}"
        self.dim = dim

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                out[i, int(sha256(tok)[:8], 16) % self.dim] += 1.0
        return _normalize(out)


def make_embedder(spec: dict, cache_dir: str | Path | None = None) -> Embedder:
    if spec.get("model", "").startswith("hash"):
        return HashEmbedder(int(spec.get("dim", 256)))
    return SentenceTransformerEmbedder(
        model=spec.get("model", "BAAI/bge-m3"),
        device=spec.get("device", "auto"),
        batch_size=spec.get("batch_size", 16),
        query_prefix=spec.get("query_prefix", ""),
        doc_prefix=spec.get("doc_prefix", ""),
        max_seq_length=spec.get("max_seq_length", 512),
        cache_dir=cache_dir,
    )
