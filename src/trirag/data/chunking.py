"""Cắt văn bản thành chunk theo số token, giữ nguyên id tài liệu nguồn."""

from __future__ import annotations

import re

from ..utils import count_tokens
from .schema import Document

_SENT_SPLIT = re.compile(r"(?<=[.!?。;:])\s+|\n+")


def split_text(text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """Cắt theo câu, gom câu đến khi chạm ``max_tokens`` (câu quá dài bị cắt cứng)."""
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for s in sentences:
        n = count_tokens(s)
        if n > max_tokens:  # câu quá dài → cắt theo từ
            words = s.split()
            step = max(1, int(len(words) * max_tokens / n))
            pieces = [" ".join(words[i : i + step]) for i in range(0, len(words), step)]
        else:
            pieces = [s]
        for p in pieces:
            pn = count_tokens(p)
            if cur and cur_tok + pn > max_tokens:
                chunks.append(" ".join(cur))
                if overlap_tokens > 0:
                    keep: list[str] = []
                    kt = 0
                    for prev in reversed(cur):
                        kt += count_tokens(prev)
                        if kt > overlap_tokens:
                            break
                        keep.insert(0, prev)
                    cur, cur_tok = keep, sum(count_tokens(k) for k in keep)
                else:
                    cur, cur_tok = [], 0
            cur.append(p)
            cur_tok += pn
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def split_document(doc: Document, max_tokens: int, overlap_tokens: int = 0) -> list[Document]:
    """Cắt một Document thành nhiều chunk; mỗi chunk giữ ``source_doc_id`` = doc gốc."""
    if count_tokens(doc.text) <= max_tokens:
        return [Document(doc.doc_id, doc.title, doc.text, {**doc.meta, "source_doc_id": doc.doc_id})]
    parts = split_text(doc.text, max_tokens, overlap_tokens)
    return [
        Document(f"{doc.doc_id}::{i}", doc.title, p, {**doc.meta, "source_doc_id": doc.doc_id})
        for i, p in enumerate(parts)
    ]
