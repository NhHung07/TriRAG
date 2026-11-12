"""Kiểu dữ liệu chung cho mọi bộ dữ liệu."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    """Một đơn vị truy hồi (đoạn văn / chunk) có id nguồn để truy vết trích dẫn."""

    doc_id: str
    title: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(d["doc_id"], d.get("title", ""), d["text"], d.get("meta", {}))


@dataclass
class QAExample:
    qid: str
    question: str
    answers: list[str]  # có thể rỗng (VD câu hỏi UltraDomain chấm bằng judge)
    gold_doc_ids: list[str] = field(default_factory=list)
    corpus_id: str = "all"  # câu hỏi được trả lời trên kho tài liệu nào
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QAExample":
        return cls(
            d["qid"],
            d["question"],
            list(d.get("answers", [])),
            list(d.get("gold_doc_ids", [])),
            d.get("corpus_id", "all"),
            d.get("meta", {}),
        )


@dataclass
class Dataset:
    name: str
    examples: list[QAExample]
    corpora: dict[str, list[Document]]  # corpus_id -> tài liệu

    def stats(self) -> dict:
        return {
            "name": self.name,
            "n_examples": len(self.examples),
            "n_corpora": len(self.corpora),
            "n_docs": sum(len(v) for v in self.corpora.values()),
        }
