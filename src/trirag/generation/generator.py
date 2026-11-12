"""Bộ sinh câu trả lời dùng chung: đóng gói ngữ cảnh theo ngân sách token rồi gọi LLM."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..retrievers.base import RetrievedItem
from ..utils import count_tokens, truncate_tokens
from .prompts import PROMPTS

KIND_LABEL = {"chunk": "", "summary": "summary", "entity": "entity", "relation": "relation", "path": "graph path"}


@dataclass
class PackedContext:
    text: str
    items: list[RetrievedItem]  # các item thực sự nằm trong ngữ cảnh (theo thứ tự)
    n_tokens: int
    doc_ids: list[str] = field(default_factory=list)


def format_item(i: int, it: RetrievedItem) -> str:
    label = KIND_LABEL.get(it.kind, it.kind)
    head = f"[{i}]"
    if it.title:
        head += f" {it.title}"
    if label:
        head += f" ({label})"
    return f"{head}\n{it.text.strip()}"


def pack_context(items: list[RetrievedItem], budget_tokens: int) -> PackedContext:
    """Thêm item theo thứ tự đến khi hết ngân sách; item cuối bị cắt bớt nếu còn ≥ 50 token."""
    parts: list[str] = []
    used: list[RetrievedItem] = []
    total = 0
    for it in items:
        block = format_item(len(used) + 1, it)
        n = count_tokens(block) + 2
        if total + n > budget_tokens:
            remaining = budget_tokens - total
            if remaining >= 50:
                parts.append(truncate_tokens(block, remaining - 2))
                used.append(it)
                total = budget_tokens
            break
        parts.append(block)
        used.append(it)
        total += n
    doc_ids = list(dict.fromkeys(d for it in used for d in it.source_doc_ids))
    return PackedContext("\n\n".join(parts), used, total, doc_ids)


@dataclass
class Answer:
    text: str
    latency_s: float
    cached: bool
    input_tokens: int
    output_tokens: int


class Generator:
    def __init__(self, llm, prompt: str = "qa_short", max_tokens: int = 64, temperature: float = 0.0):
        if prompt not in PROMPTS:
            raise ValueError(f"Prompt không tồn tại: {prompt}. Có: {list(PROMPTS)}")
        self.llm = llm
        self.system, self.user = PROMPTS[prompt]
        self.max_tokens = max_tokens
        self.temperature = temperature

    def answer(self, question: str, context: PackedContext) -> Answer:
        prompt = self.user.format(context=context.text or "(empty)", question=question)
        t0 = time.perf_counter()
        resp, cached = self.llm.generate_full(
            prompt, system=self.system, temperature=self.temperature, max_tokens=self.max_tokens, tag="answer"
        )
        return Answer(resp.text.strip(), time.perf_counter() - t0, cached, resp.input_tokens, resp.output_tokens)
