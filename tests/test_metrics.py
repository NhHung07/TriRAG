"""Kiểm thử độ đo câu trả lời (khớp script chính thức HotpotQA) và độ đo truy hồi."""

from __future__ import annotations

import math

import pytest

from trirag.eval.metrics import (
    all_recall_at_k,
    answer_metrics,
    exact_match,
    f1_score,
    mrr,
    ndcg_at_k,
    normalize_answer,
    recall_at_k,
    retrieval_metrics,
)


def test_normalize_and_exact_match():
    assert normalize_answer("The Eiffel Tower, Paris!") == "eiffel tower paris"
    assert exact_match("The United States", "united states") == 1.0
    assert exact_match("Paris", "London") == 0.0


def test_hotpotqa_f1_special_and_overlap():
    # Quy tắc đặc biệt của HotpotQA: yes/no/noanswer không khớp một phần với từ khác
    assert f1_score("yes, it is", "yes") == 0.0
    assert f1_score("yes", "yes") == 1.0
    assert f1_score("no", "yes") == 0.0
    # Khớp một phần token: pred="new york city" (3), gold="new york" (2), chung=2 -> P=2/3, R=1 -> F1=0.8
    assert pytest.approx(f1_score("New York City", "New York"), rel=1e-5) == 0.8
    am = answer_metrics("barack obama", ["President Barack Obama", "Barack Hussein Obama"])
    assert am["em"] == 0.0
    assert pytest.approx(am["f1"], rel=1e-5) == 0.8


def test_retrieval_metrics_hand_computed():
    ranked = ["d3", "d1", "d4", "d2", "d5"]
    gold = ["d1", "d2"]
    # Ở k=2: chỉ có d1 -> recall@2 = 0.5, all_recall@2 = 0.0
    assert pytest.approx(recall_at_k(ranked, gold, 2)) == 0.5
    assert all_recall_at_k(ranked, gold, 2) == 0.0
    # Ở k=4: có cả d1 và d2 -> recall@4 = 1.0, all_recall@4 = 1.0
    assert recall_at_k(ranked, gold, 4) == 1.0
    assert all_recall_at_k(ranked, gold, 4) == 1.0
    # MRR: tài liệu vàng đầu tiên (d1) ở vị trí index=1 (hạng 2) -> MRR = 1/2
    assert pytest.approx(mrr(ranked, gold)) == 0.5
    # nDCG@4: d1 ở i=1 (1/log2(3)), d2 ở i=3 (1/log2(5)); IDCG = 1/log2(2) + 1/log2(3)
    expected_dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    expected_idcg = 1.0 + 1.0 / math.log2(3)
    assert pytest.approx(ndcg_at_k(ranked, gold, 4)) == expected_dcg / expected_idcg

    rm = retrieval_metrics(ranked, context_doc_ids=["d3", "d1"], gold=gold, ks=(2, 5))
    assert pytest.approx(rm["evidence_recall"]) == 0.5
    assert rm["evidence_all"] == 0.0
    assert rm["all_recall@5"] == 1.0
