"""QASPER — tái lập thu nhỏ RAPTOR (mỗi bài báo là một kho tài liệu riêng).

Tải dữ liệu: https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
(dùng file ``qasper-dev-v0.3.json``).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .schema import Dataset, Document, QAExample

URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"


def _answer_strings(qa: dict) -> list[str]:
    out = []
    for a in qa.get("answers", []):
        ans = a["answer"]
        if ans.get("unanswerable"):
            out.append("unanswerable")
        elif ans.get("extractive_spans"):
            out.append(", ".join(ans["extractive_spans"]))
        elif ans.get("yes_no") is not None:
            out.append("yes" if ans["yes_no"] else "no")
        elif ans.get("free_form_answer"):
            out.append(ans["free_form_answer"])
    return list(dict.fromkeys(out))


def load_qasper(
    path: str | Path,
    n_papers: int = 15,
    seed: int = 42,
    max_questions: int | None = 100,
    skip_unanswerable: bool = True,
) -> Dataset:
    raw: dict = json.loads(Path(path).read_text(encoding="utf-8"))
    rng = random.Random(seed)

    def usable(qa):
        ans = _answer_strings(qa)
        return ans and not (skip_unanswerable and all(a == "unanswerable" for a in ans))

    candidates = sorted(pid for pid, p in raw.items() if any(usable(q) for q in p.get("qas", [])))
    paper_ids = rng.sample(candidates, min(n_papers, len(candidates)))

    corpora: dict[str, list[Document]] = {}
    examples: list[QAExample] = []
    for pid in paper_ids:
        paper = raw[pid]
        docs = [Document(f"{pid}#abs", f"{paper['title']} — Abstract", paper.get("abstract", "").strip())]
        for si, sec in enumerate(paper.get("full_text", [])):
            for pi, para in enumerate(sec.get("paragraphs", [])):
                if para and para.strip():
                    docs.append(
                        Document(f"{pid}#s{si}p{pi}", f"{paper['title']} — {sec.get('section_name') or ''}", para.strip())
                    )
        corpora[pid] = docs
        text_to_id = {d.text: d.doc_id for d in docs}
        for qa in paper.get("qas", []):
            if not usable(qa):
                continue
            ans = _answer_strings(qa)
            if skip_unanswerable:
                ans = [a for a in ans if a != "unanswerable"]
            evidence = {
                text_to_id[e.strip()]
                for a in qa["answers"]
                for e in a["answer"].get("evidence", [])
                if e.strip() in text_to_id
            }
            examples.append(
                QAExample(qa["question_id"], qa["question"], ans, sorted(evidence), corpus_id=pid, meta={"paper": pid})
            )
    rng.shuffle(examples)
    if max_questions:
        examples = examples[:max_questions]
        keep = {e.corpus_id for e in examples}
        corpora = {k: v for k, v in corpora.items() if k in keep}
    return Dataset("qasper", examples, corpora)
