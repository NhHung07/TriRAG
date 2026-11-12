"""Tài liệu tiếng Việt cho demo (VD đề cương môn học dạng PDF 50–70 trang).

Trích văn bản theo trang (PyMuPDF), chuẩn hoá Unicode NFC, cắt theo đề mục
(Chương / Bài / Mục / 1.2.3 ...) rồi theo số token. Mỗi chunk giữ tên file + số trang để trích dẫn.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from ..utils import count_tokens
from .chunking import split_text
from .schema import Dataset, Document, QAExample

HEADING_RE = re.compile(
    r"^\s*((CHƯƠNG|Chương|PHẦN|Phần|BÀI|Bài|MỤC|Mục|TUẦN|Tuần)\s+[\dIVXLC]+\b"
    r"|\d{1,2}(\.\d{1,2}){0,3}\.?\s+[A-ZÀ-Ỹ])"
)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def _read_pages(path: Path) -> list[tuple[int, str]]:
    if path.suffix.lower() == ".pdf":
        import fitz  # PyMuPDF

        with fitz.open(path) as pdf:
            return [(i + 1, page.get_text("text")) for i, page in enumerate(pdf)]
    return [(1, path.read_text(encoding="utf-8"))]


def is_heading(line: str) -> bool:
    return len(line) <= 120 and bool(HEADING_RE.match(line))


def load_vi_documents(input_dir: str | Path, max_tokens: int = 400, overlap_tokens: int = 50) -> list[Document]:
    input_dir = Path(input_dir)
    files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"})
    docs: list[Document] = []
    for f in files:
        sections: list[tuple[str, list[tuple[int, str]]]] = [("", [])]
        for page_no, page_text in _read_pages(f):
            for line in page_text.splitlines():
                line = _norm(line)
                if not line or re.fullmatch(r"\d{1,3}", line):  # bỏ dòng trống / số trang
                    continue
                if is_heading(line):
                    sections.append((line, []))
                sections[-1][1].append((page_no, line))
        idx = 0
        for heading, lines in sections:
            if not lines:
                continue
            text = "\n".join(l for _, l in lines)
            pages = sorted({p for p, _ in lines})
            parts = [text] if count_tokens(text) <= max_tokens else split_text(text, max_tokens, overlap_tokens)
            for part in parts:
                docs.append(
                    Document(
                        doc_id=f"{f.stem}#{idx:04d}",
                        title=f"{f.stem} — {heading}" if heading else f.stem,
                        text=part,
                        meta={"file": f.name, "pages": pages, "section": heading},
                    )
                )
                idx += 1
    return docs


def load_vi_questions(path: str | Path) -> list[QAExample]:
    """File JSONL do nhóm tự soạn: {"question": ..., "answers": [...], "gold_doc_ids": [...] (tuỳ chọn)}."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            d = json.loads(line)
            out.append(QAExample(d.get("qid", f"vi-{i:03d}"), d["question"], d.get("answers", []), d.get("gold_doc_ids", [])))
    return out


def load_vi(input_dir, questions_path=None, max_tokens=400, overlap_tokens=50) -> Dataset:
    docs = load_vi_documents(input_dir, max_tokens, overlap_tokens)
    qs = load_vi_questions(questions_path) if questions_path else []
    return Dataset("vi_docs", qs, {"all": docs})
