"""UltraDomain — tái lập thu nhỏ LightRAG (đánh giá win-rate bằng LLM judge).

Dữ liệu: https://huggingface.co/datasets/TommyChien/UltraDomain (file ``<domain>.jsonl``,
mỗi dòng có trường ``context`` là toàn văn một cuốn sách). Như trong ``reproduce/Step_0.py``
của LightRAG, ta lấy các context không trùng lặp; ở đây cắt tổng độ dài còn ``max_tokens``.

Câu hỏi không có đáp án chuẩn → sinh tự động bằng LLM (tương tự ``reproduce/Step_2.py``):
LLM đọc mô tả kho dữ liệu, đề xuất N người dùng × M tác vụ × K câu hỏi mức khái quát.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..utils import count_tokens, truncate_tokens
from .chunking import split_text
from .schema import Dataset, Document, QAExample


def load_ultradomain_corpus(path: str | Path, max_tokens: int = 150_000, chunk_tokens: int = 1200) -> list[Document]:
    contexts: list[str] = []
    seen = set()
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            ctx = json.loads(line).get("context", "")
            if ctx and ctx not in seen:
                seen.add(ctx)
                contexts.append(ctx)
    docs: list[Document] = []
    used = 0
    for ci, ctx in enumerate(contexts):
        remaining = max_tokens - used
        if remaining <= 0:
            break
        n = count_tokens(ctx)
        if n > remaining:
            ctx = truncate_tokens(ctx, remaining)
            n = remaining
        used += n
        for k, chunk in enumerate(split_text(ctx, chunk_tokens)):
            docs.append(Document(f"ud{ci}#c{k}", f"context {ci}", chunk))
    return docs


QUESTION_PROMPT = """Given the following description of a dataset:

{description}

Identify {n_users} potential users who would engage with this dataset. For each user, list {n_tasks} tasks \
they would perform with this dataset. Then, for each (user, task) combination, write {n_questions} questions \
that require a high-level understanding of the ENTIRE dataset (not a single fact lookup).

Return ONLY valid JSON with this structure:
{{"users": [{{"user": "...", "tasks": [{{"task": "...", "questions": ["...", "..."]}}]}}]}}"""


def corpus_description(docs: list[Document], tokens: int = 2000) -> str:
    text = "\n".join(d.text for d in docs)
    head = truncate_tokens(text, tokens // 2)
    if len(head) >= len(text):
        return text
    return head + "\n...\n" + text[-len(head) :]


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("LLM không trả về JSON")
    return json.loads(m.group(0))


def generate_questions(llm, docs: list[Document], n_users=2, n_tasks=4, n_questions=5) -> list[QAExample]:
    prompt = QUESTION_PROMPT.format(
        description=corpus_description(docs), n_users=n_users, n_tasks=n_tasks, n_questions=n_questions
    )
    data = _parse_json(llm.generate(prompt, max_tokens=4096, tag="dataset:ud_questions"))
    out: list[QAExample] = []
    for ui, u in enumerate(data.get("users", [])):
        for ti, t in enumerate(u.get("tasks", [])):
            for qi, q in enumerate(t.get("questions", [])):
                out.append(
                    QAExample(f"ud-u{ui}t{ti}q{qi}", q, [], [], "all", {"user": u.get("user"), "task": t.get("task")})
                )
    return out


def load_ultradomain(path, max_tokens=150_000, chunk_tokens=1200, llm=None, **qkw) -> Dataset:
    docs = load_ultradomain_corpus(path, max_tokens, chunk_tokens)
    examples = generate_questions(llm, docs, **qkw) if llm is not None else []
    return Dataset("ultradomain", examples, {"all": docs})
