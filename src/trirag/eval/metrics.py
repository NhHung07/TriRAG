"""Độ đo đánh giá.

* Câu trả lời: Exact Match, token-F1 — chuẩn hoá theo script chính thức của HotpotQA/SQuAD
  (lấy max trên các đáp án tham chiếu).
* Truy hồi (theo id tài liệu nguồn): Recall@k, All-Recall@k (đủ mọi tài liệu vàng — quan trọng
  với câu hỏi multi-hop), MRR, nDCG@k (quan hệ nhị phân).
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Token-F1 theo script chính thức HotpotQA (xử lý đặc biệt yes/no/noanswer)."""
    p, g = normalize_answer(prediction), normalize_answer(gold)
    special = {"yes", "no", "noanswer"}
    if (p in special or g in special) and p != g:
        return 0.0
    p_toks, g_toks = p.split(), g.split()
    common = Counter(p_toks) & Counter(g_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p_toks), same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def max_over_golds(fn, prediction: str, golds: list[str]) -> float:
    return max((fn(prediction, g) for g in golds), default=0.0)


def answer_metrics(prediction: str, golds: list[str]) -> dict[str, float]:
    return {"em": max_over_golds(exact_match, prediction, golds), "f1": max_over_golds(f1_score, prediction, golds)}


# ---------------- truy hồi ----------------


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    top = set(ranked[:k])
    return sum(1 for g in set(gold) if g in top) / len(set(gold))


def all_recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    return float(set(gold) <= set(ranked[:k]))


def mrr(ranked: list[str], gold: list[str]) -> float:
    if not gold:
        return float("nan")
    g = set(gold)
    for i, d in enumerate(ranked):
        if d in g:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    g = set(gold)
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in g)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(g), k)))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(ranked: list[str], context_doc_ids: list[str], gold: list[str], ks=(2, 5, 10)) -> dict:
    """``ranked``: id tài liệu theo thứ tự truy hồi; ``context_doc_ids``: id có trong ngữ cảnh thực sự đưa vào LLM.

    ``evidence_recall`` (tính trên ngữ cảnh đã đóng gói) là độ đo chính để so sánh giữa các
    phương pháp trả về đơn vị khác nhau (chunk / tóm tắt / thực thể / đường đi đồ thị).
    """
    out = {
        "evidence_recall": recall_at_k(context_doc_ids, gold, len(context_doc_ids) or 1),
        "evidence_all": all_recall_at_k(context_doc_ids, gold, len(context_doc_ids) or 1),
        "mrr": mrr(ranked, gold),
    }
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        out[f"all_recall@{k}"] = all_recall_at_k(ranked, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, gold, k)
    return out
