"""Baseline Dense RAG: embedding local + tìm kiếm cosine (brute-force numpy).

Với kho vài nghìn – vài chục nghìn chunk, brute-force nhanh hơn và đơn giản hơn FAISS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.schema import Document
from ..utils import append_jsonl, read_jsonl
from .base import RetrievalResult, Retriever
from .bm25 import _item


class DenseRetriever(Retriever):
    type_name = "dense"

    def _build(self, docs: list[Document], index_dir: Path):
        self.docs = docs
        self.emb = self.c.embedder.encode([d.full_text for d in docs])
        np.save(index_dir / "embeddings.npy", self.emb)
        (index_dir / "docs.jsonl").unlink(missing_ok=True)
        append_jsonl(index_dir / "docs.jsonl", (d.to_dict() for d in docs))
        return {"embedding_model": self.c.embedder.name, "dim": int(self.emb.shape[1])}

    def _load(self, index_dir: Path):
        self.docs = [Document.from_dict(d) for d in read_jsonl(index_dir / "docs.jsonl")]
        self.emb = np.load(index_dir / "embeddings.npy")

    def scores(self, query: str) -> np.ndarray:
        q = self.c.embedder.encode([query], is_query=True)[0]
        return self.emb @ q

    def _retrieve(self, query: str, top_k: int) -> RetrievalResult:
        s = self.scores(query)
        order = np.argsort(-s)[:top_k]
        return RetrievalResult([_item(self.docs[i], float(s[i])) for i in order])
