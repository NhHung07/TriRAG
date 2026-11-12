#!/usr/bin/env python3
"""Chạy đánh giá truy hồi + sinh câu trả lời (hoặc chấm cặp bằng LLM-as-a-judge)."""

from __future__ import annotations

import argparse
import logging

from trirag.config import load_config
from trirag.eval.judge import judge_pair, win_rates
from trirag.eval.runner import ExperimentRunner
from trirag.eval.summary import MAIN_COLS, summary_tables, to_markdown
from trirag.utils import load_env, read_jsonl, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_eval")


def run_pairwise_judge(runner: ExperimentRunner, method_a: str, method_b: str, limit: int | None = None) -> dict:
    recs_a = {r["qid"]: r for r in read_jsonl(runner.results_dir / f"{method_a}.jsonl")}
    recs_b = {r["qid"]: r for r in read_jsonl(runner.results_dir / f"{method_b}.jsonl")}
    qids = [q for q in recs_a if q in recs_b]
    if limit:
        qids = qids[:limit]
    judge_llm = runner.c.judge_llm if runner.cfg.get("judge") else runner.c.answer_llm
    results = []
    for qid in qids:
        ra, rb = recs_a[qid], recs_b[qid]
        res = judge_pair(judge_llm, ra["question"], ra.get("prediction", ""), rb.get("prediction", ""))
        results.append({"qid": qid, "question": ra["question"], **res})
    summary = {"method_a": method_a, "method_b": method_b, "n_questions": len(results), "win_rates": win_rates(results)}
    out_file = runner.results_dir / f"judge_{method_a}_vs_{method_b}.json"
    write_json(out_file, {"summary": summary, "details": results})
    log.info("Kết quả LLM-judge (%s vs %s) lưu tại %s: %s", method_a, method_b, out_file, summary["win_rates"])
    return summary


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Đường dẫn tới file YAML cấu hình")
    ap.add_argument("--methods", default="", help="Danh sách phương pháp cách nhau bởi dấu phẩy (mặc định: tất cả)")
    ap.add_argument("--limit", type=int, default=None, help="Giới hạn số câu hỏi chạy thử")
    ap.add_argument("--retrieval-only", action="store_true", help="Chỉ chạy truy hồi, không gọi LLM sinh câu trả lời")
    ap.add_argument("--judge-pair", default="", help="So sánh cặp 2 phương pháp bằng LLM judge, VD: lightrag,dense")
    args = ap.parse_args()

    cfg = load_config(args.config)
    runner = ExperimentRunner(cfg)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or list(cfg["methods"].keys())
    for m in methods:
        runner.run(m, limit=args.limit, retrieval_only=args.retrieval_only)

    tables = summary_tables(cfg)
    print("\n=== BẢNG CHẤT LƯỢNG TRUY HỒI & CÂU TRẢ LỜI ===")
    print(to_markdown(tables["main"], ["method", "n"] + MAIN_COLS + ["context_tokens", "retrieval_ms_p50", "generation_ms_p50"]))
    print("=== BẢNG CHI PHÍ LẬP CHỈ MỤC ===")
    print(to_markdown(tables["cost"]))

    if args.judge_pair:
        ma, mb = [x.strip() for x in args.judge_pair.split(",")]
        run_pairwise_judge(runner, ma, mb, limit=args.limit)


if __name__ == "__main__":
    main()
