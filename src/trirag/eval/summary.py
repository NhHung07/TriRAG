"""Tổng hợp kết quả: bảng so sánh chất lượng + chi phí cho từng phương pháp."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..utils import read_json, read_jsonl

MAIN_COLS = ["em", "f1", "evidence_recall", "evidence_all", "recall@5", "all_recall@5", "mrr", "ndcg@10"]


def _mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def summarize_records(records: list[dict]) -> dict:
    row: dict[str, float] = {"n": len(records)}
    am = defaultdict(list)
    for r in records:
        for k, v in (r.get("answer_metrics") or {}).items():
            am[k].append(v)
        for k, v in (r.get("retrieval_metrics") or {}).items():
            am[k].append(v)
    for k, vals in am.items():
        row[k] = _mean(vals)
    row["context_tokens"] = _mean([r.get("context_tokens") for r in records])
    rt = [r["retrieval_s"] for r in records if "retrieval_s" in r]
    gt = [r["generation_s"] for r in records if "generation_s" in r and not r.get("gen_cached")]
    row["retrieval_ms_p50"] = float(np.median(rt) * 1000) if rt else float("nan")
    row["generation_ms_p50"] = float(np.median(gt) * 1000) if gt else float("nan")
    return row


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    return {p.stem: list(read_jsonl(p)) for p in sorted(results_dir.glob("*.jsonl"))}


def index_cost(index_root: Path, method: str) -> dict:
    stats = [read_json(p) for p in (index_root / method).glob("*/index_stats.json")]
    if not stats:
        return {}
    return {
        "index_s": sum(s["index_seconds"] for s in stats),
        "index_llm_calls": sum(s["llm"].get("calls", 0) for s in stats),
        "index_llm_in_tok": sum(s["llm"].get("input_tokens", 0) for s in stats),
        "index_llm_out_tok": sum(s["llm"].get("output_tokens", 0) for s in stats),
        "index_mb": sum(s.get("index_size_mb", 0) for s in stats),
    }


def summary_tables(cfg: dict, group_by: str | None = None) -> dict[str, list[dict]]:
    results_dir = Path(cfg["paths"]["results_dir"]) / cfg["experiment"]
    index_root = Path(cfg["paths"]["index_dir"]) / cfg["experiment"]
    res = load_results(results_dir)
    main, cost, grouped = [], [], []
    for method, recs in res.items():
        main.append({"method": method, **summarize_records(recs)})
        cost.append({"method": method, **index_cost(index_root, method)})
        if group_by:
            groups = defaultdict(list)
            for r in recs:
                groups[str(r.get("meta", {}).get(group_by))].append(r)
            for g, rs in sorted(groups.items()):
                grouped.append({"method": method, group_by: g, **summarize_records(rs)})
    return {"main": main, "cost": cost, "grouped": grouped}


def to_markdown(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(trống)_\n"
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    cols = [c for c in cols if any(c in r for r in rows)]

    def fmt(c, v):
        if isinstance(v, float):
            if math.isnan(v):
                return "–"
            if _is_pct(c):
                return f"{v * 100:.1f}"
            return f"{v:,.1f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(c, r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _is_pct(col: str) -> bool:
    return col in ("em", "f1", "mrr") or col.startswith(("recall@", "all_recall@", "ndcg@", "evidence_"))
