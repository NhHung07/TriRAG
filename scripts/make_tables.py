#!/usr/bin/env python3
"""Xuất bảng tổng hợp kết quả (Markdown + LaTeX) cho báo cáo."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from trirag.config import DEFAULTS, load_config
from trirag.eval.summary import MAIN_COLS, _is_pct, summary_tables, to_markdown
from trirag.utils import PROJECT_ROOT


def to_latex(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return ""
    cols = [c for c in cols if any(c in r for r in rows)]

    def fmt(c: str, v) -> str:
        if isinstance(v, float):
            if math.isnan(v):
                return "--"
            return f"{v * 100:.1f}" if _is_pct(c) else f"{v:,.1f}"
        return str(v).replace("_", "\\_")

    header = " & ".join(c.replace("_", "\\_").replace("@", "@") for c in cols) + " \\\\"
    body = [" & ".join(fmt(c, r.get(c, "")) for c in cols) + " \\\\" for r in rows]
    spec = "l" + "r" * (len(cols) - 1)
    return "\n".join([f"\\begin{{tabular}}{{{spec}}}", "\\toprule", header, "\\midrule", *body, "\\bottomrule", "\\end{tabular}"]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="", help="Đường dẫn tới file YAML cấu hình")
    ap.add_argument("--exp", default="", help="Tên experiment (nếu không truyền --config)")
    ap.add_argument("--group-by", default="type", help="Trường trong meta để phân tách (VD: type = bridge/comparison)")
    args = ap.parse_args()

    if args.config:
        cfg = load_config(args.config)
    elif args.exp:
        cfg = dict(DEFAULTS, experiment=args.exp)
        cfg["paths"] = {k: str(PROJECT_ROOT / v) for k, v in DEFAULTS["paths"].items()}
    else:
        ap.error("Cần truyền --config hoặc --exp")

    tables = summary_tables(cfg, group_by=args.group_by or None)
    main_cols = ["method", "n"] + MAIN_COLS + ["context_tokens", "retrieval_ms_p50", "generation_ms_p50"]
    cost_cols = ["method", "index_s", "index_llm_calls", "index_llm_in_tok", "index_llm_out_tok", "index_mb"]

    md_parts = [
        f"# Kết quả thực nghiệm: `{cfg['experiment']}`\n",
        "## 1. Chất lượng truy hồi & sinh câu trả lời\n",
        to_markdown(tables["main"], main_cols),
        "## 2. Chi phí lập chỉ mục\n",
        to_markdown(tables["cost"], cost_cols),
    ]
    if tables["grouped"]:
        md_parts.extend([
            f"## 3. Phân tích theo `{args.group_by}`\n",
            to_markdown(tables["grouped"], ["method", args.group_by, "n", "em", "f1", "evidence_recall", "evidence_all"]),
        ])
    md_text = "\n".join(md_parts)
    tex_text = (
        "% Bảng chất lượng chính\n"
        + to_latex(tables["main"], main_cols)
        + "\n% Bảng chi phí lập chỉ mục\n"
        + to_latex(tables["cost"], cost_cols)
    )

    out_dir = Path(cfg["paths"]["results_dir"]) / cfg["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(md_text, encoding="utf-8")
    (out_dir / "summary.tex").write_text(tex_text, encoding="utf-8")
    print(md_text)


if __name__ == "__main__":
    main()
