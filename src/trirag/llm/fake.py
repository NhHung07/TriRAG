"""LLM giả lập — cho unit test / smoke test không tốn quota (hỗ trợ cả RAPTOR & LightRAG)."""

from __future__ import annotations

import json
import re
from typing import Callable

from .base import LLMBackend, LLMResponse


def _default_responder(prompt: str, system: str | None) -> str:
    # 1. LightRAG trích từ khoá truy vấn
    if "high_level_keywords" in prompt and "low_level_keywords" in prompt:
        q_match = re.search(r"Query:\s*(.+)", prompt)
        q = q_match.group(1) if q_match else prompt[-200:]
        words = [w for w in re.findall(r"[A-Za-zÀ-ỹ0-9]{3,}", q) if w.lower() not in {"what", "which", "where", "who", "when", "the", "and"}][:4]
        if not words:
            words = ["entity"]
        return json.dumps({"high_level_keywords": words[:2], "low_level_keywords": words})

    # 2. LightRAG trích thực thể & quan hệ từ văn bản
    if "---Input Text---" in prompt or (system and "Knowledge Graph Specialist" in system):
        inp = prompt.rsplit("---Input Text---", 1)[-1]
        ents = list(dict.fromkeys(re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}(?:\s+[A-Z][a-zA-Z0-9]{2,})?", inp)))[:6]
        if len(ents) < 2:
            ents = ["Paris", "France"]
        lines = [f"entity<|#|>e{e}<|#|>Concept<|#|>Entity {e} in {inp[:80]}." for e in ents]
        for i in range(len(ents) - 1):
            lines.append(f"relation<|#|>e{ents[i]}<|#|>e{ents[i+1]}<|#|>related, connected<|#|>Relation between {ents[i]} and {ents[i+1]}.")
        lines.append("<|COMPLETE|>")
        return "\n".join(lines)

    # 3. Sinh câu hỏi UltraDomain
    if '"users"' in prompt and '"tasks"' in prompt:
        return json.dumps({"users": [{"user": "researcher", "tasks": [{"task": "review", "questions": ["What are the main themes?"]}]}]})

    # 4. LLM-as-a-judge
    if '"Comprehensiveness"' in prompt and '"Overall"' in prompt:
        return json.dumps({
            c: {"winner": "Answer 1", "reason": "fake"} for c in ("Comprehensiveness", "Diversity", "Empowerment", "Overall")
        })

    # 5. Tóm tắt RAPTOR / trả lời ngắn
    return "Summary of key facts and entities."


class FakeBackend(LLMBackend):
    def __init__(self, responder: Callable[[str, str | None], str] | None = None, model: str = "fake"):
        self.model = model
        self.responder = responder or _default_responder
        self.calls = 0

    def complete(self, prompt, *, system, temperature, max_tokens) -> LLMResponse:
        self.calls += 1
        text = self.responder(prompt, system)
        return LLMResponse(text=text, input_tokens=max(1, len(prompt) // 4), output_tokens=max(1, len(text) // 4))
