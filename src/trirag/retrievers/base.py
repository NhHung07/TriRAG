"""Interface truy hồi chung + bộ chứa thành phần dùng chung (LLM, embedding).

Nguyên tắc công bằng: mọi phương pháp chỉ trả về NGỮ CẢNH (danh sách :class:`RetrievedItem`
có id tài liệu nguồn). Việc sinh câu trả lời do ``trirag.generation.Generator`` đảm nhận,
dùng chung LLM + prompt + ngân sách token cho tất cả phương pháp.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from ..data.schema import Document
from ..utils import read_json, write_json

log = logging.getLogger(__name__)


@dataclass
class RetrievedItem:
    text: str
    source_doc_ids: list[str]
    score: float = 0.0
    kind: str = "chunk"  # chunk | summary | entity | relation | path
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    items: list[RetrievedItem]
    latency_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def ranked_doc_ids(self) -> list[str]:
        """Danh sách id tài liệu nguồn theo thứ tự xuất hiện (khử trùng)."""
        seen: dict[str, None] = {}
        for it in self.items:
            for d in it.source_doc_ids:
                seen.setdefault(d, None)
        return list(seen)


class Components:
    """Khởi tạo lười (lazy) LLM / embedding để BM25 chạy được mà không cần API key hay torch."""

    def __init__(self, cfg: dict, *, answer_llm=None, index_llm=None, judge_llm=None, embedder=None):
        self.cfg = cfg
        self.cache_dir = Path(cfg["paths"]["cache_dir"])
        if answer_llm is not None:
            self.__dict__["answer_llm"] = answer_llm
        if index_llm is not None:
            self.__dict__["index_llm"] = index_llm
        if judge_llm is not None:
            self.__dict__["judge_llm"] = judge_llm
        if embedder is not None:
            self.__dict__["embedder"] = embedder

    @cached_property
    def answer_llm(self):
        from ..llm import make_llm

        return make_llm(self.cfg["llm"], self.cache_dir, "answer")

    @cached_property
    def index_llm(self):
        from ..llm import make_llm

        return make_llm(self.cfg["index_llm"], self.cache_dir, "index")

    @cached_property
    def judge_llm(self):
        from ..llm import make_llm

        if not self.cfg.get("judge"):
            raise RuntimeError("Chưa cấu hình `judge` trong config.")
        return make_llm(self.cfg["judge"], self.cache_dir, "judge")

    @cached_property
    def embedder(self):
        from ..embeddings import make_embedder

        return make_embedder(self.cfg["embedding"], self.cache_dir)


class Retriever(ABC):
    """Lớp cơ sở. Lớp con cài ``_build``, ``_load``, ``_retrieve``."""

    type_name: str = "base"

    def __init__(self, name: str, params: dict, components: Components):
        self.name = name
        self.params = params
        self.c = components
        self.index_dir: Path | None = None

    # ---- vòng đời chỉ mục ----
    @staticmethod
    def is_built(index_dir: str | Path) -> bool:
        return (Path(index_dir) / "DONE").exists()

    @property
    def llm_tag(self) -> str:
        return f"index:{self.name}"

    def build_index(self, docs: list[Document], index_dir: str | Path) -> dict:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = index_dir
        from ..llm import TRACKER

        before = TRACKER.snapshot(self.llm_tag)
        t0 = time.perf_counter()
        extra = self._build(docs, index_dir) or {}
        seconds = time.perf_counter() - t0
        after = TRACKER.snapshot(self.llm_tag)
        llm = {k: after[k] - before.get(k, 0) for k in after}
        stats = {
            "method": self.name,
            "type": self.type_name,
            "n_docs": len(docs),
            "index_seconds": round(seconds, 2),
            "llm": llm,
            "index_size_mb": round(_dir_size(index_dir) / 1e6, 2),
            "params": self.params,
            **extra,
        }
        write_json(index_dir / "index_stats.json", stats)
        (index_dir / "DONE").write_text("ok")
        return stats

    def load(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self._load(self.index_dir)

    def index_stats(self) -> dict:
        p = self.index_dir / "index_stats.json" if self.index_dir else None
        return read_json(p) if p and p.exists() else {}

    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        t0 = time.perf_counter()
        res = self._retrieve(query, top_k)
        res.latency_s = time.perf_counter() - t0
        return res

    # ---- cài đặt trong lớp con ----
    @abstractmethod
    def _build(self, docs: list[Document], index_dir: Path) -> dict | None: ...

    @abstractmethod
    def _load(self, index_dir: Path) -> None: ...

    @abstractmethod
    def _retrieve(self, query: str, top_k: int) -> RetrievalResult: ...


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
