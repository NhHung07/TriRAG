"""LLM-as-a-judge: so sánh cặp câu trả lời (theo giao thức đánh giá của LightRAG).

Tiêu chí: Comprehensiveness, Diversity, Empowerment, Overall. Mỗi cặp được chấm 2 lần với thứ tự
A/B đảo ngược để giảm thiên lệch vị trí; win-rate = số phiếu thắng / tổng phiếu.
Nên dùng model judge KHÁC model sinh câu trả lời (cấu hình ``judge`` trong config).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

CRITERIA = ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]

JUDGE_SYSTEM = (
    "You are an expert evaluator. You compare two answers to the same question and judge them "
    "strictly on the given criteria. Be impartial: ignore answer order and length unless it reflects quality."
)

JUDGE_USER = """Question:
{question}

Answer 1:
{a1}

Answer 2:
{a2}

Evaluate both answers on these criteria:
- Comprehensiveness: How much detail does the answer provide to cover all aspects of the question?
- Diversity: How varied and rich is the answer in offering different perspectives and insights?
- Empowerment: How well does the answer help the reader understand the topic and make informed judgements?
- Overall: Which answer is better overall, considering the three criteria above?

Return ONLY valid JSON:
{{"Comprehensiveness": {{"winner": "Answer 1" or "Answer 2", "reason": "..."}},
 "Diversity": {{"winner": "...", "reason": "..."}},
 "Empowerment": {{"winner": "...", "reason": "..."}},
 "Overall": {{"winner": "...", "reason": "..."}}}}"""


def _parse(text: str) -> dict[str, int]:
    """Trả về {criterion: 1 hoặc 2} (0 nếu không đọc được)."""
    m = re.search(r"\{.*\}", text, re.S)
    out = {c: 0 for c in CRITERIA}
    if not m:
        return out
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    for c in CRITERIA:
        w = str((data.get(c) or {}).get("winner", ""))
        out[c] = 1 if "1" in w else 2 if "2" in w else 0
    return out


def judge_pair(llm, question: str, ans_a: str, ans_b: str) -> dict:
    """Chấm 2 lượt (A,B) và (B,A). Trả về số phiếu cho A / B theo từng tiêu chí."""
    votes = {c: {"A": 0, "B": 0} for c in CRITERIA}
    raw = []
    for order in ("AB", "BA"):
        a1, a2 = (ans_a, ans_b) if order == "AB" else (ans_b, ans_a)
        text = llm.generate(
            JUDGE_USER.format(question=question, a1=a1, a2=a2), system=JUDGE_SYSTEM, max_tokens=1024, tag="judge"
        )
        parsed = _parse(text)
        raw.append(parsed)
        for c, w in parsed.items():
            if w == 0:
                continue
            winner = order[w - 1]  # vị trí 1 → order[0]
            votes[c][winner] += 1
    return {"votes": votes, "raw": raw}


def win_rates(results: list[dict]) -> dict[str, dict[str, float]]:
    tot: dict[str, dict[str, int]] = defaultdict(lambda: {"A": 0, "B": 0})
    for r in results:
        for c, v in r["votes"].items():
            tot[c]["A"] += v["A"]
            tot[c]["B"] += v["B"]
    out = {}
    for c in CRITERIA:
        n = tot[c]["A"] + tot[c]["B"]
        out[c] = {"A": tot[c]["A"] / n if n else float("nan"), "B": tot[c]["B"] / n if n else float("nan"), "n_votes": n}
    return out
