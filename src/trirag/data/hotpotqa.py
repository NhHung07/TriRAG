"""HotpotQA (distractor, dev) — benchmark so sánh chung.

Lấy mẫu ``n_questions`` câu hỏi cố định theo seed. Kho tài liệu = hợp (khử trùng theo tiêu đề)
tất cả đoạn văn context của các câu được chọn (~10 đoạn/câu), dùng chung cho mọi câu hỏi
→ bài toán truy hồi trên kho chung chứ không phải 10 đoạn riêng của từng câu.

Tải dữ liệu: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .schema import Dataset, Document, QAExample

URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


def _sample(raw: list[dict], n: int, seed: int, stratify: str) -> list[dict]:
    rng = random.Random(seed)
    if n >= len(raw):
        return list(raw)
    if stratify == "balanced":  # chia đều bridge / comparison
        by_type: dict[str, list[dict]] = defaultdict(list)
        for ex in raw:
            by_type[ex["type"]].append(ex)
        types = sorted(by_type)
        picked: list[dict] = []
        for i, t in enumerate(types):
            k = n // len(types) + (1 if i < n % len(types) else 0)
            picked.extend(rng.sample(by_type[t], min(k, len(by_type[t]))))
        rng.shuffle(picked)
        return picked
    return rng.sample(raw, n)


def load_hotpotqa(path: str | Path, n_questions: int = 300, seed: int = 42, stratify: str = "balanced") -> Dataset:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = _sample(raw, n_questions, seed, stratify)

    docs: dict[str, Document] = {}
    examples: list[QAExample] = []
    for ex in selected:
        for title, sents in ex["context"]:
            if title not in docs:
                docs[title] = Document(doc_id=title, title=title, text="".join(sents).strip())
        gold = list(dict.fromkeys(t for t, _ in ex["supporting_facts"]))
        examples.append(
            QAExample(
                qid=ex["_id"],
                question=ex["question"],
                answers=[ex["answer"]],
                gold_doc_ids=gold,
                corpus_id="all",
                meta={"type": ex["type"], "level": ex["level"]},
            )
        )
    return Dataset("hotpotqa", examples, {"all": list(docs.values())})
