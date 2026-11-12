#!/usr/bin/env python3
"""Phân loại tự động và lấy mẫu các trường hợp trả lời sai (4 nhóm lỗi trong Mục 6 đề xuất)."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from trirag.config import DEFAULTS, load_config
from trirag.eval.summary import load_results, to_markdown
from trirag.utils import PROJECT_ROOT

ERROR_CATEGORIES = {
    "missing_evidence": "Truy hồi thiếu hoàn toàn bằng chứng (evidence_recall = 0)",
    "constraint_or_hop_omission": "Bỏ sót điều kiện / thiếu bước suy luận (0 < evidence_recall < 1)",
    "wrong_distractor": "Truy hồi nhầm tài liệu nhiễu (không có tài liệu vàng)",
    "generation_or_hallucination": "Truy hồi đủ bằng chứng (evidence_all = 1) nhưng LLM trả lời sai / ảo giác",
}


def classify_error(rec: dict) -> str:
    rm = rec.get("retrieval_metrics") or {}
    ev_rec = rm.get("evidence_recall", 0.0)
    ev_all = rm.get("evidence_all", 0.0)
    if ev_all == 1.0:
        return "generation_or_hallucination"
    if ev_rec > 0.0:
        return "constraint_or_hop_omission"
    if rec.get("context_doc_ids"):
        return "wrong_distractor"
    return "missing_evidence"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="", help="Đường dẫn tới file YAML cấu hình")
    ap.add_argument("--exp", default="", help="Tên experiment")
    ap.add_argument("--sample-size", type=int, default=20, help="Số câu sai lấy mẫu mỗi phương pháp")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.config:
        cfg = load_config(args.config)
    elif args.exp:
        cfg = dict(DEFAULTS, experiment=args.exp)
        cfg["paths"] = {k: str(PROJECT_ROOT / v) for k, v in DEFAULTS["paths"].items()}
    else:
        ap.error("Cần truyền --config hoặc --exp")

    res_dir = Path(cfg["paths"]["results_dir"]) / cfg["experiment"]
    all_res = load_results(res_dir)
    rng = random.Random(args.seed)

    dist_rows: list[dict] = []
    md_lines = [f"# Phân tích lỗi — `{cfg['experiment']}`\n"]

    for method, recs in all_res.items():
        wrong = [r for r in recs if (r.get("answer_metrics") or {}).get("em", 1.0) == 0.0 and (r.get("answer_metrics") or {}).get("f1", 1.0) < 0.5]
        counts = Counter(classify_error(r) for r in wrong)
        n_wrong = len(wrong) or 1
        dist_rows.append({
            "method": method,
            "n_total": len(recs),
            "n_wrong": len(wrong),
            **{k: f"{counts[k]} ({counts[k] * 100 / n_wrong:.1f}%)" for k in ERROR_CATEGORIES},
        })

        sampled = rng.sample(wrong, min(args.sample_size, len(wrong)))
        md_lines.append(f"## Phương pháp: `{method}` ({len(wrong)}/{len(recs)} câu F1 < 0.5)\n")
        for i, r in enumerate(sampled, 1):
            cat = classify_error(r)
            md_lines.append(
                f"### {i}. `{r['qid']}` — **{cat}** ({ERROR_CATEGORIES[cat]})\n"
                f"- **Câu hỏi**: {r['question']}\n"
                f"- **Đáp án chuẩn**: `{r.get('answers')}` | **Dự đoán**: `{r.get('prediction')}` (F1={r.get('answer_metrics', {}).get('f1', 0):.2f})\n"
                f"- **Gold docs**: `{r.get('gold_doc_ids')}` | **Context docs**: `{r.get('context_doc_ids')}`\n"
            )

    header_table = "## Phân bố các loại lỗi theo phương pháp\n\n" + to_markdown(
        dist_rows, ["method", "n_total", "n_wrong"] + list(ERROR_CATEGORIES.keys())
    )
    md_lines.insert(1, header_table)
    out_path = res_dir / "error_analysis.md"
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(header_table)
    print(f"Đã lưu chi tiết mẫu lỗi tại: {out_path}")


if __name__ == "__main__":
    main()
