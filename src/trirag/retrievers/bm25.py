"""Baseline BM25 (Okapi) trên đơn vị tài liệu."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from ..data.schema import Document
from .base import RetrievalResult, RetrievedItem, Retriever

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever(Retriever):
    type_name = "bm25"

    def _build(self, docs: list[Document], index_dir: Path):
        from rank_bm25 import BM25Okapi

        self.docs = docs
        self.bm25 = BM25Okapi(
            [tokenize(d.full_text) for d in docs], k1=self.params.get("k1", 1.5), b=self.params.get("b", 0.75)
        )
        with (index_dir / "bm25.pkl").open("wb") as f:
            pickle.dump({"docs": [d.to_dict() for d in docs], "bm25": self.bm25}, f)

    def _load(self, index_dir: Path):
        with (index_dir / "bm25.pkl").open("rb") as f:
            obj = pickle.load(f)
        self.docs = [Document.from_dict(d) for d in obj["docs"]]
        self.bm25 = obj["bm25"]

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

    def _retrieve(self, query: str, top_k: int) -> RetrievalResult:
        s = self.scores(query)
        order = np.argsort(-s)[:top_k]
        return RetrievalResult([_item(self.docs[i], float(s[i])) for i in order])


def _item(doc: Document, score: float) -> RetrievedItem:
    src = doc.meta.get("source_doc_id", doc.doc_id)
    return RetrievedItem(text=doc.text, source_doc_ids=[src], score=score, kind="chunk", title=doc.title, meta=doc.meta)
