"""CẢI TIẾN CỦA NHÓM: Hybrid BM25 + Dense (Reciprocal Rank Fusion) + reranker cross-encoder.

1. Lấy ``candidates`` ứng viên từ mỗi nhánh BM25 và Dense.
2. Hợp nhất bằng RRF: score(d) = Σ_r w_r / (k + rank_r(d)).
3. (Tuỳ chọn) xếp hạng lại top ứng viên bằng cross-encoder (mặc định ``BAAI/bge-reranker-v2-m3``,
   đa ngữ, hỗ trợ tiếng Việt).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from ..data.schema import Document
from .base import RetrievalResult, Retriever
from .bm25 import BM25Retriever, _item
from .dense import DenseRetriever

log = logging.getLogger(__name__)

_RERANKERS: dict[str, object] = {}
_RR_LOCK = threading.Lock()


def get_reranker(model: str, device: str = "auto"):
    with _RR_LOCK:
        if model not in _RERANKERS:
            from sentence_transformers import CrossEncoder

            if device == "auto":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Nạp reranker %s trên %s", model, device)
            _RERANKERS[model] = CrossEncoder(model, device=device, max_length=512)
        return _RERANKERS[model]


def rrf_fuse(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever(Retriever):
    type_name = "hybrid"

    def __init__(self, name, params, components):
        super().__init__(name, params, components)
        self.bm25 = BM25Retriever(f"{name}/bm25", params.get("bm25", {}), components)
        self.dense = DenseRetriever(f"{name}/dense", {}, components)

    def _build(self, docs: list[Document], index_dir: Path):
        self.bm25.build_index(docs, index_dir / "bm25")
        self.dense.build_index(docs, index_dir / "dense")
        self.docs = docs
        return {"embedding_model": self.c.embedder.name}

    def _load(self, index_dir: Path):
        self.bm25.load(index_dir / "bm25")
        self.dense.load(index_dir / "dense")
        self.docs = self.dense.docs

    def _retrieve(self, query: str, top_k: int) -> RetrievalResult:
        n = self.params.get("candidates", 30)
        r_bm25 = list(np.argsort(-self.bm25.scores(query))[:n])
        r_dense = list(np.argsort(-self.dense.scores(query))[:n])
        fused = rrf_fuse(
            [r_bm25, r_dense], k=self.params.get("rrf_k", 60), weights=self.params.get("weights", [1.0, 1.0])
        )
        if self.params.get("rerank", False):
            pool = [i for i, _ in fused[: self.params.get("rerank_pool", 20)]]
            reranker = get_reranker(self.params.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
                                    self.params.get("device", "auto"))
            ce = reranker.predict([(query, self.docs[i].full_text) for i in pool], show_progress_bar=False)
            ranked = sorted(zip(pool, ce), key=lambda x: -x[1])
        else:
            ranked = fused
        return RetrievalResult([_item(self.docs[i], float(s)) for i, s in ranked[:top_k]])
